# HAO AI Lab FP4 FA4 versus TK MXFP4 FA4

Date: 2026-07-22

This note compares:

1. The HAO AI Lab CuTe-DSL kernel pinned at commit
   `9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247` in
   `/workspace/codebases/flash-attention-fp4`.
2. Our authoritative frozen TK MXFP4 production artifact, SHA256
   `678f238c0c37666a49df6949961f28dabef092b1e24f2f8de274b5c503432a51`,
   and its matching source snapshot under
   `results/mxfp4_fa4_production_corr_pv_ex2_20260717T221047Z/source_snapshot`.
3. The HAO-inspired local rewrite currently under design. This third item is
   not a landed or benchmarked replacement for production.

The distinction matters because the current working tree contains many later
experiments. Claims about "our current kernel" below refer to the frozen
production artifact unless a section explicitly says candidate or experiment.

## Short answer

HAO did not obtain more TMEM and did not eliminate the requirement that
block-scale operands reach TMEM. It uses the same 512-column SM100/SM103 TMEM
allocation. Its central choice is how those columns change ownership over
time:

```text
HAO:       S0 | S1 | O0 | O1
columns: 0-127 128-255 256-383 384-511
```

`O0` and `O1` are permanent output accumulators. After a score bank has been
read into registers, HAO reuses that retired score bank for the packed P
payload and the short-lived Q/K/P/V scale pages. It then makes one tensor
issue warp obey a deterministic order in which PV consumes the old P before
the following QK overwrites the bank.

Our frozen production kernel instead uses:

```text
TK:        S0 | S1 | O | permanent scale slab
columns: 0-127 128-255 256-383 384-511
```

It keeps P in shared memory and keeps explicit Q/K/P/V scale regions in TMEM.
To obtain a second logical output accumulator without another 128 columns, it
borrows a retired score bank. That bank therefore has to pass through score,
P-publication, temporary-output, correction, and reuse lifetimes. This is the
"hot plate" ownership problem that led to the larger readiness protocol.

HAO's deep pipeline is primarily a shared-memory K/V prefetch ring. It is not
a deeper TMEM pipeline: TMEM still contains only two score banks and two
output banks. HAO also handles two M128 query tiles in one CTA, while our
frozen production route handles one. At the measured causal S4096/H64 case,
that gives our route 2048 one-query CTAs and about 14 physical service waves
versus the TK BF16 route's 1024 logical two-query jobs and about 7 service
rounds. The timed BF16 persistent route executes those jobs with 148 worker
CTAs. Our FP4 work per wave is cheaper, but the doubled wave count reverses
the advantage.

Finally, HAO's fastest reported mode is not full FP4 QKPV. It uses NVFP4 for
QK and plain FP8 for P/V, which removes P/V block-scale traffic. Its full
NVFP4 QKPV path is only around parity with BF16 at some shapes and wins mainly
at long context on GB300. HAO therefore reached the same fundamental
conclusion as us: full FP4 PV scale handling and online P construction are
hard to hide. Its cleaner TMEM ownership and two-query packaging reduce the
damage; they do not make the cost disappear.

## Terminology

- **Score bank (`S0` or `S1`)**: a 128-column FP32 TMEM region holding one
  128x128 QK result.
- **Output bank (`O0` or `O1`)**: a 128-column FP32 TMEM region holding a
  128x128 PV accumulator for one query tile.
- **Query stage**: one independent M128 query tile inside a CTA. HAO uses two.
- **K/V pipeline stage**: one shared-memory ring position used by the loader
  and tensor issuer. In HAO, the ring positions hold alternating K and V
  items, not complete K/V pairs.
- **Scale page**: the compact TMEM layout consumed by a block-scaled TCGEN MMA.
  The original scale tensor can live in global or shared memory, but the
  instruction-facing fragment must be copied to TMEM before issue.
- **P publication split**: making the first fraction of packed P visible to
  PV before the remaining fraction is finished. This is separate from query
  staging and from the K/V ring depth.
- **Quartering in our production route**: the online P path processes each
  N128 score row as four N32 MX groups. It does not mean the production QK
  operation is four independent 128x32 score MMAs.
- **Full FP4**: Q, K, P, and V all use FP4 payloads with block scales. HAO's
  `NVFP4+FP8` mode is not full FP4 because P and V are plain FP8.

## Architecture at a glance

| Property | HAO pinned kernel | Frozen TK production |
|---|---|---|
| Logical score tile | M128 x N128 | M128 x N128 |
| Q/K head dimension | 128 in published suite | 192 |
| V/output head dimension | 128 in published suite | 128 |
| QK instruction K extent | 128 in published suite | Dqk=192, three K64 commands |
| PV instruction K extent | 128 | 128, two K64 commands |
| Query tiles per CTA | 2 | 1 |
| CTA group | 1 CTA; 2-CTA MMA explicitly rejected | 1 CTA in production |
| Threads | 512, 16 warps | 384, 12 warps |
| TMEM allocation | 512 columns | 512 columns |
| Permanent score banks | 2 | 2 |
| Permanent output banks | 2 | 1 |
| Second output state | Dedicated `O1` | Retired score bank aliases as spare output |
| Permanent scale slab | None | 128 columns |
| P payload location | Packed into upper half of retired score TMEM | Two-stage shared-memory P ring |
| P/V scale location at issue | Low pages of retired score bank | Dedicated P/V TMEM scale slots |
| Q/K scale location at issue | Low pages of opposite score bank | Dedicated Q/K TMEM scale slots |
| Physical K payload slots | Dynamic alternating K/V ring | 2 |
| Physical K scale slots | Same alternating ring generation | 2 |
| Physical V payload slots | Same alternating ring generation | 2 |
| P payload slots | Two TMEM score-bank overlays | 2 SMEM slots |
| Tensor issue ownership | One warp, W12 | Elected issue path in producer WG |
| Main issue policy | Fixed `PV(stage,n) -> QK(stage,n+1)` order | Explicit readiness/hot-plate protocol |
| Default scheduling | Static persistent for noncausal fixed length | H64 production locked to full grid |
| Online softmax | Real online softmax | Real online softmax |
| P quantization | Fused log-domain group quant by default | Fused score-derived per-N32 E8M0 quant |
| Published fastest PV mode | Plain FP8 P/V, no block scales | Full E2M1 plus E8M0 block-scaled PV |

## What HAO does

### Tile and CTA packaging

HAO fixes `m_block_size=128`, `n_block_size=128`, and normally sets
`q_stage=2`. Its CTA tile is therefore M256 x N128, represented as two
independent M128 query stages. QK remains M128 x N128 per stage and PV remains
M128 x Dvo per stage. Both QK and PV accumulate in FP32.

The CTA is not a two-CTA cluster. HAO derives `use_2cta_instrs` from an M256
QK MMA tile and asserts that it is false. The two-query result comes from one
512-thread CTA owning two query stages, not from cluster-wide tensor MMA.

This packaging is important at high head count. The same K/V tile traversal
services two query tiles, and the launch needs roughly half as many CTAs as a
one-query implementation for the same number of M128 query tiles.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:119-175` and `:624-630`.

### Warp specialization

HAO assigns all 16 warps statically:

| Warps | Role |
|---:|---|
| 0-3 | Stage-0 score read, online softmax, and P construction |
| 4-7 | Stage-1 score read, online softmax, and P construction |
| 8-11 | Output correction/rescale |
| 12 | Sole QK/PV TCGEN issue warp and scale SMEM-to-TMEM copies |
| 13 | Epilogue and output store |
| 14 | Q/K/V and scale load producer |
| 15 | Reserved; the non-TMA fallback can use it as a second loader |

The benefit is not merely more warps. Each role has a small ownership
contract, and only W12 mutates the tensor-command stream. The score readers
can notify the exact scale copy that needs the retired bank instead of forcing
the whole CTA to rendezvous at every score fraction.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:210-255`.

### Exact TMEM layout

For N128 and Dvo128, all 512 columns are assigned as follows:

```text
column       0                 128               256               384       512
             |-----------------|-----------------|-----------------|----------|
base owner   |       S0        |       S1        |       O0        |    O1    |
             |-----------------|-----------------|-----------------|----------|
```

The output banks are permanent throughout the key loop. The score banks are
time-multiplexed:

```text
S stage while live:
    [                 FP32 QK score, 128 columns                  ]

Same bank after all score readers have acquired it:
    [ transient scale pages | spare | packed FP4 P, upper 64 cols ]

After dependent PV has consumed P:
    bank may become the next FP32 QK score again
```

The packed P offsets are 64 columns into each score bank, so P0 starts at
column 64 and P1 at column 192. Q/K scales are cross-staggered into the low
part of the opposite score bank. P/V scales use the low part of the current
retired score bank immediately before PV.

There is no fifth permanent region for scales. This is the main TMEM trick:
scales occupy score storage only during the short interval in which TCGEN
must see them.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:219-255` and `:1568-1700`.

### The score-bank lifecycle

For one query stage, the conceptual lifecycle is:

```text
QK writes S_s
      |
      v
softmax warps load all required score fragments to registers
      |
      +--> signal that the overlapping scale area is reusable
      |
      v
mask + row max + correction + exp2 + row sum + P quantization
      |
      +--> write packed P into upper half of S_s
      +--> write P scales to SMEM
      |
      v
W12 copies P/V scales into low pages of S_s
      |
      v
PV consumes packed P and those scale pages, accumulating into permanent O_s
      |
      v
QK may overwrite S_s with the next score generation
```

The score-read fence is concrete. After the TMEM-to-register score load, the
softmax warp executes the TMEM-load visibility fence and arrives on the
opposite stage's Q/K-scale-load barrier. That arrival tells W12 that a scale
copy may use the retired score area without destroying unread scores.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:3627-3648`.

### Deterministic tensor issue order

HAO does not run a general scheduler that polls many possible score owners.
The one issue warp follows a fixed stream. In steady state, each stage does:

```text
wait for V item
wait for first P publication and corrected output ownership
copy P and V scales: SMEM -> retired score-bank TMEM pages
issue PV(stage, n)
wait for next K item
wait for the opposite score bank's scale-copy lease
copy Q and K scales: SMEM -> opposite score-bank TMEM pages
issue QK(stage, n+1)
```

Across both stages, the stream is effectively:

```text
PV0(n) -> QK0(n+1) -> PV1(n) -> QK1(n+1) -> repeat
```

This order is also a memory proof. For a given stage, PV has consumed the P
overlay before QK overwrites that score bank. HAO's source explicitly says it
does not need another old-S reuse wait before QK because the QK is issued
after the dependent PV on the same warp.

That does not mean there are no barriers. W12 still waits for score-read,
P-ready, output-rescaled, scale-copy, and load-pipeline events. The important
difference is that the event graph is static and point-to-point rather than a
runtime choice among several possible bank owners.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:2423-2637`, especially
`:2589-2593`.

### P is published before it is completely stored

The P store is split into two readiness events:

1. Store an initial fraction of P and signal `P_full`.
2. Continue storing the remainder and signal `P_full_2`.
3. PV starts from the first signal; its issue helper embeds the second wait
   before it reaches the still-unpublished K fragments.

The default split depends on the PV type:

- Block-scaled FP4 PV defaults to 1/2 because a K128 FP4 PV has only two K64
  issue chunks.
- Plain FP8 and BF16 paths generally use a 3/4 first publication.

This is the precise meaning of HAO's partial or early P handoff. It is not a
third TMEM score slot. It overlaps the tail of P packing/storing with the
front of PV issue.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:1099-1124` and `:3841-3870`.

### Shared-memory pipeline depth

HAO computes `kv_stage` from a 227 KiB per-block shared-memory budget. Fixed
storage includes barriers, row correction state, two Q tiles, two Q-scale
tiles, two P-scale snapshots, and a two-stage output-store buffer. The
remaining bytes are divided by the size of one K/V ring item.

K and V payload storage aliases physically when they have the same element
type or when K is narrower than V. Each ring position holds exactly one item:

```text
stream item 0 = K0
stream item 1 = V0
stream item 2 = K1
stream item 3 = V1
...
```

Payload and its scale tensor use the same pipeline position and generation.
Consequently, `kv_stage=13` means 13 alternating items, about 6.5 K/V pairs;
it does not mean 13 complete K/V pairs.

For d128, applying the source formula gives the following derived depths:

| QK format | PV format | Derived `kv_stage` | Approximate complete K/V pairs resident |
|---|---|---:|---:|
| NVFP4 | NVFP4 | 13 | 6.5 |
| NVFP4 | MXFP8 | 7 | 3.5 |
| NVFP4 | BF16 | 4 | 2 |
| NVFP4 | plain FP8 | About 8 by bytes, deliberately capped to 4 | 2 |

These numbers are source-formula derivations, not compiler-emitted resource
reports. The actual selected depth can change with head dimension, output
layout, type, and environment overrides.

The deep ring hides global-to-shared K/V and scale loads. It does not add
score or output capacity in TMEM.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:370-424`, `:1520-1539`, and
`:1999-2214`.

### HAO quantization formats

HAO permits the QK and PV sides to use different block formats.

| Tensor | Full NVFP4 QKPV mode | Other supported mode | When quantized |
|---|---|---|---|
| Q | E2M1 payload, E4M3 scale per 16 | MXFP8 E4M3 plus E8M0 per 32 | Before timed forward |
| K | E2M1 payload, E4M3 scale per 16 | MXFP8 E4M3 plus E8M0 per 32 | Before timed forward |
| V | E2M1 payload, E4M3 scale per 16 | BF16, plain FP8, or MXFP8 E4M3 plus E8M0 per 32 | Before timed forward |
| P | E2M1 payload, E4M3 scale per 16 | BF16, plain FP8, or MXFP8 E4M3 plus E8M0 per 32 | Online inside forward |

The benchmark uses FlashInfer's `nvfp4_quantize` for Q/K and full-NVFP4 V.
It uses `mxfp8_quantize` for MXFP8 V. Plain FP8 V is a direct E4M3 cast and
has no block-scale tensor. The Q/K/V conversion is outside the timed forward
call.

Unlike our Q/K input contract, HAO's benchmark interface does not carry an
additional FP32 per-head `q_sg` and `k_sg`; it passes one as the quantizer's
global scale argument and relies on the adaptive block scales.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:137-152` and
`flash_attn/cute/benchmarks/bench_fp4.py:555-655`.

### HAO online P quantization

HAO performs a real online softmax. It loads FP32 scores, applies masking,
updates the running row maximum, rescales the previous output accumulator,
computes exponentials and the row sum, and quantizes P for PV.

Its default block-scaled-P path uses a log-domain fusion. For a scale group
`g` and already softmax-scaled score `s_i`:

```text
m_g  = max(s_i in group g)
bias = m_g - log2(Pmax)
P_i  = exp2(s_i - bias)
SF_g = exp2(bias)
```

`Pmax` is 6 for E2M1 and 448 for E4M3. This is algebraically equivalent to
computing all exponentials, finding their group maximum, dividing by that
maximum, and then quantizing, but it removes the per-group divide and the
post-exp scale pass. The row sum is reconstructed as:

```text
row_sum = sum_g SF_g * sum(P_i in group g)
```

For an E8M0 scale, HAO rounds the log-domain bias upward so the quantized P
does not exceed the payload type's maximum. P scales are packed in registers,
written to shared memory, then copied by W12 to transient TMEM scale pages.
Packed P itself is written directly from registers to the retired score bank.

HAO also contains an optional software pipeline that interleaves P conversion
and TMEM stores, but it is disabled by default. The log-domain quantizer is
enabled by default.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:3110-3295` and `:3707-3773`.

### Exponential path

On SM100, HAO can emulate part of exp2 with arithmetic instructions so FP32
FMA work overlaps tensor MMA better. On SM103, the code disables that
emulation by default because hardware MUFU.EX2 throughput is doubled and the
emulation loses. SM103 can also use `tcgen05.ld.red` to combine score loading
with a row-max reduction. Masked or causal iterations still fall back to a
software maximum because the hardware reduction saw values before masking.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:81-103`, `:300-335`, and
`:3627-3673`.

### Scheduler and launch behavior

For fixed-length noncausal attention, HAO defaults to a static persistent
scheduler. The grid is capped at the SM count, and each CTA advances through
work by adding the grid dimension to its logical tile index. The launch asks
for at least one block per SM.

This is not universal. Causal and local attention select
`SingleTileLPTScheduler`, and variable-length attention selects a different
scheduler. Therefore HAO's published default noncausal result should not be
described as proof that its causal route has the same persistent topology.

Source: `flash_attn/cute/flash_fwd_sm100_fp4.py:1038-1055`, `:1320-1327`, and
`flash_attn/cute/tile_scheduler.py:287-369`.

### Published performance and what it actually proves

HAO's README says full QKVP NVFP4 was 0.84x to 0.95x BF16 on B200. On its
GB300 noncausal suite, full NVFP4 QKPV reaches 1725 TFLOPS versus 1533 TFLOPS
for BF16 at B1/S32768/H24/D128, about 1.125x. At B1/S4096/H24/D128 it is 1291
versus 1322 TFLOPS, so it remains slightly slower.

The fastest published GB300 mode is NVFP4 QK plus plain FP8 PV: 2677 TFLOPS
versus 1533 TFLOPS at B1/S32768/H24/D128, about 1.75x. That mode avoids P and
V block-scale pages entirely. It is the strongest evidence that block-scaled
PV, rather than FP4 QK, is the difficult part.

HAO reports full-NVFP4 output cosine similarity around 0.981 on its suite.
MXFP8 PV is around 0.99 and plain-FP8 PV is generally around 0.99. Those
numbers are tied to HAO's formats and reference setup and are not a numerical
comparison with our per-32 E8M0 path.

Source: `flash_attn/cute/README.md:1-26`, `:72-106`, and `:161-177`.

## What our frozen production kernel does

### Frozen identity and measured shape

The authoritative route is:

```text
dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix
```

The frozen H64 analysis is causal B1/S4096/H64 with Q/K head dimension 192
and V/output head dimension 128. Its instantiated kernel template is
M128/N128/Dqk192/Dvo128. Dqk192 is three K64 QK issue chunks; it is not three
K-ring slots. The timing lock's generic topology object contains `dim: 128`,
but the instantiated symbol, source constants, and profile analysis all make
the asymmetric Dqk192/Dvo128 contract explicit; those are authoritative.

The route launches one M128 query tile per CTA. Although scheduler machinery
is compiled into the family, the authoritative H64 host route passes
`persistent_launch_argument=false`, disallows persistence for this route, and
selects the full-grid launch.

Source:
`results/mxfp4_fa4_production_corr_pv_ex2_20260717T221047Z/h64_production_bf16_reproduction_timing_lock.json`
and frozen `fwd_host_dispatch.inc:6121-6122`.

### Warp specialization

The production configuration uses three warpgroups, or 12 warps:

| Warpgroup | Main responsibility |
|---:|---|
| WG0 | Output correction/rescale, output handling, and epilogue work |
| WG1 | Four score-reader/online-softmax/P-quantization warps |
| WG2 | Q/K/V producer work and the elected tensor-issue path |

Compared with HAO, more unrelated responsibilities share each warpgroup. In
particular, loading and tensor issue are not isolated into separate W14 and
W12 roles, and output correction and epilogue are more tightly coupled.

Source: frozen `fwd_streaming_kernel.inc:5193-5215` and the production config
in frozen `fwd_configs.inc:1450-1456`.

### Exact TMEM layout

For the frozen production specialization, the 512 columns are:

```text
columns       use
0..127        score bank S0
128..255      score bank S1
256..383      one permanent FP32 output accumulator O
384..399      Q E4M3 scale page
400..415      K E4M3 scale page
416..431      P E8M0 scale epoch 0
432..447      P E8M0 scale epoch 1
448..479      V E8M0 scale epoch 0
480..511      V E8M0 scale epoch 1
```

The route advertises dual output accumulation, but it does not allocate two
permanent output banks. The helper returns the score bank corresponding to
`score_idx % SCORE_TMEM_SLOTS` as the spare output tensor. In other words:

```text
logical output A -> permanent O at columns 256..383
logical output B -> whichever retired score bank is leased as temporary O
```

These two logical output states belong to the same single M128 query tile;
they are not HAO-style query stages. They let successive PV/correction states
overlap inside one query traversal. HAO's O0 and O1 instead belong to two
different M128 query tiles owned by the same CTA.

Source: frozen `fwd_streaming_kernel.inc:3853-3931`, `:4888-4931`, and
`:10497-10510`.

### The local hot-plate lifecycle

Our score-bank state machine is longer than HAO's:

```text
QK writes FP32 score S_i
      |
      v
WG1 reads score in four N32 groups
      |
      +--> real mask/max/exp/sum/P quant
      +--> P payload is published to a separate SMEM P slot
      +--> P scale is published to a dedicated TMEM scale epoch
      |
      v
the retired score bank may become a temporary FP32 output accumulator
      |
      v
WG0 correction/rescale and output ownership complete
      |
      v
PV and output merge/release complete
      |
      v
only now may QK overwrite the bank with S_(i+2)
```

Quartering reduced the amount of P work that must finish before publication,
but it did not create another physical score or output bank. The same score
bank still transitions through more roles than HAO's bank because HAO never
uses `S0` or `S1` as an output accumulator.

This is why the issue cannot be described simply as "P and QK write the same
place." P payload is in shared memory in our route. The conflict is that the
score bank is also the only available storage for the second FP32 output
state, and QK cannot reclaim it while that output state is live.

### Shared-memory and scale staging depths

The frozen production route has:

- two physical K payload slots;
- two physical K-scale slots;
- two V payload slots;
- two V-scale TMEM epochs;
- two P payload slots in shared memory;
- two P-scale TMEM epochs;
- two score TMEM banks.

For logical key tile `n`, the K payload and K scale both use slot `n % 2` and
readiness generation `floor(n/2) % 2`. The value `Dqk/64=3` is the number of
K64 scale/MMA chunks within one QK tile, not a third prefetched tile.

This is shallower than HAO's dynamically sized alternating K/V ring. However,
our exact H64 timeline found V already about 2.29 microseconds ahead at the PV
wait, while observed V/P/PV handoff waits were only one 96-224 ns polling
quantum. Therefore increasing V prefetch depth alone is not supported as the
main H64 fix. A deeper aliased ring is valuable if it simplifies ownership
and isolates the producer, not because the frozen profile says V arrives late.

Source:
`results/mxfp4_fa4_upstream_strategy_20260722/k_slot_phase_prebuild_proof.md`
and the stage attribution in
`h64_existing_profile_analysis_20260719.json`.

### Synchronization model

The production kernel contains explicit event families for score arrival,
score copy completion, correction, P payload and scale publication, P-stage
reuse, P-scale reuse, V payload and scale readiness, output reuse, temporary
output reuse, and hot-plate release epochs.

That source complexity should not be misreported as "barrier stalls dominate
runtime." The exact NCU capture reports an average barrier stall ratio of
0.20 for FP4 versus 0.14 for BF16. The larger symptoms are low eligible-warp
density, long scoreboard/wait exposure, and only 7.02 percent tensor-pipe
activity. The event graph matters because it leaves WGs with no independent
work and starves the issue stream, not because every lost cycle is classified
as a barrier stall.

Frozen source: `fwd_streaming_kernel.inc:5537-5552` and related readiness
sites. Dynamic evidence:
`h64_existing_profile_analysis_20260719.json`.

### Our quantization formats

The local path called MXFP4 is a hybrid rather than one uniform format:

| Tensor | Payload | Block scale | Extra scale | Quantization time |
|---|---|---|---|---|
| Q | E2M1 | E4M3 per 16 | FP32 per-head `q_sg` | Before timed forward |
| K | E2M1 | E4M3 per 16 | FP32 per-head `k_sg` | Before timed forward |
| V | E2M1 | E8M0 per 32 | None | Before timed forward |
| P | E2M1 | E8M0 per 32 | Online softmax state | Inside timed forward |
| O | BF16 | None | FP32 accumulator and row stats | Inside forward |

Q/K use the v5 NVFP4-style quantizer. The host computes an `amax` per
batch/head, stores `amax / 2688` as the FP32 global scalar, and packs E4M3
block scales. V uses the local MXFP4 v3 row quantizer and a K-major layout for
PV. P is generated online to match the E2M1 plus E8M0-per-32 PV instruction.

Source: frozen `fp4_pv_experiments.py:1762-1817`, `:2508-2556`,
`:3435-3454`, and `:15787-15902`; global QK scaling is applied in frozen
`fwd_device_helpers.inc:1-13`.

### Our online P quantization

For each row, the production path divides N128 into four N32 groups. It finds
the maximum raw score in each group and derives an E8M0 scale in log space:

```text
block_scale_log2 = log2(P_scale_constant)
                   + (block_max - row_max) * softmax_scale_log2
```

The frozen route selects the floor conversion for E8M0. It then derives:

```text
log2 coefficient  = log2(6 * P_scale_constant) - E8M0 exponent
reciprocal weight = exp2(E8M0 exponent) / (6 * P_scale_constant)
```

The log2 coefficient is folded into the exp2 input while eight E2M1 values
are packed into each 32-bit word. The reciprocal weight reconstructs each
group's contribution to the online row sum. Thus the route already fuses
softmax exponential, group scaling, P quantization, packing, and weighted
row-sum work. It is not a fake-P or sigmoid experiment.

The payload is published to one of two shared-memory P slots. Its E8M0 scale
is written to one of the dedicated TMEM P-scale epochs. Early publication and
reuse folding reduce some waits, but the second logical output still extends
the score-bank lifetime.

Source: frozen `fwd_streaming_kernel.inc:27628-27674` and the subsequent pack
loop.

### Exponential path in the frozen artifact

The frozen production SASS contains 257 `MUFU.EX2` sites. Later working-tree
branches added ALU/e2e exp2 experiments, but the authoritative route name and
frozen SASS do not prove that those later helpers are in the production
binary. The comparison must therefore treat the production path as the MUFU
version.

Source: `h64_existing_profile_analysis_20260719.json` under `static_sass`.

### Measured H64 behavior

At causal B1/S4096/H64:

| Metric | Frozen FP4 production | TK BF16 comparison |
|---|---:|---:|
| Protected median wall time | 380.288 us | 326.624 us, persistent route |
| M128 query tiles per CTA/job | 1 | 2 |
| Logical M128 query tiles | 2048 | 2048 |
| Logical jobs | 2048 one-query jobs | 1024 two-query jobs |
| Timed launch physical CTAs | 2048 full-grid CTAs | 148 persistent workers |
| Service-wave ceiling | 14 | 7 |
| Normalized time per service wave | 27.16 us | 46.66 us |
| NCU tensor-pipe active | 7.02% | 54.02%, full-grid surrogate |
| Eligible warps per scheduler cycle | 0.38 | 0.58 |
| Dynamic shared memory | 100,352 B | 231,424 B |
| Registers per thread | 168 | 128 |
| TMEM columns | 512 | 512 |

The strongest interpretation is:

1. FP4 tensor arithmetic is not intrinsically slower here. One FP4 service
   wave is about 41.8 percent cheaper than one BF16 wave.
2. The one-query packaging needs twice as many critical waves.
3. Within each FP4 wave, the serial score -> online softmax -> exp2/P pack ->
   correction/publication chain leaves too few eligible warps and starves
   TCGEN issue.
4. The run is not primarily DRAM, L2, TMA, occupancy-capacity, V-arrival, or
   final-output-store bound.

Removing the final output store saved only about 9-10 us in an impossible
ceiling experiment, so the epilogue store cannot explain the roughly 54 us
wall-time gap. Historical intrusive H8 markers identify exp2/P packing as the
largest P sub-block, but those marker spans cannot be summed into H64 wall
time.

Source:
`results/mxfp4_fa4_production_corr_pv_ex2_20260717T221047Z/h64_existing_profile_analysis_20260719.json`.

## Direct TMEM comparison

### HAO

```text
Permanent allocation:

  S0 128 + S1 128 + O0 128 + O1 128 = 512 columns

Transient use after score read:

  retired S low pages   = current Q/K or P/V scale fragments
  retired S upper half  = packed P
  O0/O1                 = never borrowed during the key loop
```

### TK production

```text
Permanent allocation:

  S0 128 + S1 128 + O 128 + scale slab 128 = 512 columns

Transient use after score read:

  P payload             = separate SMEM ring
  retired S bank        = second logical FP32 output accumulator
  scale slab            = Q/K/P/V pages remain explicitly allocated
```

### The actual difference

Both implementations know that a scale is needed in TMEM only at TCGEN
issue. HAO fully commits to temporal aliasing and spends the recovered 128
columns on a permanent second output. Our production path spends those 128
columns on stable scale slots and then aliases the second output into score
storage.

This produces opposite complexity:

```text
HAO must perfectly order transient scale/P use inside a retired score bank.
TK must perfectly order temporary output use inside a retired score bank.
```

HAO's problem has a shorter lifetime because P and scales are dead once PV is
issued. Our temporary FP32 output survives through correction/rescale and
merge/release, so the score-reuse edge is later and harder to predict.

## Direct pipeline comparison

### HAO has three independent forms of overlap

1. **Two query stages**: S0/O0 and S1/O1 are independent query owners.
2. **Deep alternating K/V SMEM ring**: global loads run several K/V items
   ahead of tensor issue.
3. **Partial P publication**: PV can consume the first P fraction before the
   last fraction is stored.

These should not be collapsed into one "pipeline depth" number.

### TK production has shallower storage and more dynamic ownership

1. Two score banks overlap score production and consumption.
2. K, K-scale, V, P, and P-scale storage are all depth two in the relevant
   production path.
3. Four N32 P groups support progressive construction/publication.
4. The second logical output borrows a score bank, so the next QK depends on
   a larger set of release conditions.

The local scheduler experiments attempted to exploit QK being one logical
step ahead of PV. They often made control flow or lifetimes worse because the
next QK had nowhere legal to write while the corresponding score bank still
held temporary output state. Polling the release more often cannot create
storage or independent work; it only changes how the wait is expressed.

## Direct quantization comparison

| Question | HAO full NVFP4 | TK production |
|---|---|---|
| Q/K payload | E2M1 | E2M1 |
| Q/K block scale | E4M3 per 16 | E4M3 per 16 |
| Q/K global scalar | No explicit per-head scalar in benchmark interface | FP32 scalar per batch/head |
| P/V payload | E2M1 | E2M1 |
| P/V block scale | E4M3 per 16 | E8M0 per 32 |
| P group count across N128 | 8 groups per row | 4 groups per row |
| P scale precision | Mantissa-bearing E4M3 | Exponent-only E8M0 |
| P quant method | Default fused log-domain group quant | Fused score-derived log-domain group quant |
| P payload destination | Retired score TMEM upper half | Shared memory |
| P scale path | Registers -> SMEM -> transient TMEM | Quantizer -> dedicated TMEM epoch on the direct-scale route |
| Prequantized tensors excluded from timing | Q/K/V | Q/K/V |
| Online work included in timing | Softmax and P construction | Softmax and P construction |

Our per-32 scale choice halves the number of P/V scale groups relative to
HAO's full NVFP4 path, but it does not make scale TMEM occupancy zero. TCGEN
still consumes a fixed-layout scale fragment, and our production allocation
reserves whole 16/32-column pages for alternating epochs. This is why merely
coarsening the mathematical scale granularity did not automatically release
a useful 128-column output bank.

The formats also have different error behavior. HAO's full NVFP4 uses more,
more precise P/V scales. Our E8M0-per-32 route has fewer scale values and less
scale precision. Performance and accuracy must be compared separately.

## Benchmark methodology is not yet apples-to-apples

HAO's default benchmark:

- uses `triton.testing.do_bench` with a 25 ms repetition window, 10 ms warmup,
  and median reporting;
- sleeps 0.8 seconds between shapes to recover clock/thermal state;
- defaults to noncausal attention;
- covers B1/S256/H16, B1/S1024/H16, B4/S4096/H16,
  B1/S32768/H16, B4/S4096/H32, and H12/H24 long-context shapes;
- publishes primarily d128 results;
- times already-quantized Q/K/V inputs.

Source: `flash_attn/cute/benchmarks/bench_fp4.py:711-813` and `:898-956`.

Our Python benchmark:

- uses CUDA-event samples, currently defaulting to two warmups and five timed
  iterations unless a protected runner overrides them;
- can report both CuTe-DSL BF16 FA4 and TK BF16 FA4;
- also excludes Q/K/V quantization from the timed call;
- has historically focused on causal Dqk192/Dvo128 shapes, including
  S4096/H64.

Source: frozen `fp4_pv_experiments.py:3597-3690` and `:3845-3870`.

Raw HAO TFLOPS and our frozen microseconds therefore cannot be placed in one
speedup table without rerunning the same shape, causal mode, input data,
clock policy, cooldown, warmup, and baseline implementations. HAO also needs
an isolated environment here because the pinned code requires
`nvidia-cutlass-dsl>=4.4.2` and `quack-kernels>=0.4.0`, while the shared
environment freeze recorded 4.4.1 and 0.2.10.

## What HAO confirms, and what it does not

### Confirmed structural lessons

1. Two score banks plus two permanent output banks fit in 512 columns if all
   instruction-facing scale pages are placed transiently in retired score
   storage.
2. Packed FP4 P can also fit in the upper half of a retired FP32 score bank.
3. The reliable reuse proof is a fixed one-warp command order, not a heuristic
   guess about whether QK is probably one tile ahead.
4. Two M128 query tiles per CTA are important for saturated high-head shapes.
5. A deep K/V pipeline can live in shared memory without increasing TMEM
   score/output depth.
6. Early P publication is still useful, but it is only one part of the full
   schedule.
7. Full FP4 PV remains materially harder than FP4 QK plus BF16/FP8 PV.

### Not confirmed

1. HAO does not prove that block scales can be passed directly from shared
   memory to block-scaled TCGEN MMA. It still copies them to TMEM.
2. HAO does not have a third score bank or triple-buffered output in TMEM.
3. HAO does not use 2-CTA tensor MMA for this kernel.
4. HAO does not make full FP4 universally faster than BF16.
5. HAO's deep K/V ring does not by itself solve our temporary-output lifetime.
6. HAO's published noncausal Dqk128/Dvo128 performance does not directly
   predict our causal Dqk192/Dvo128 H64 performance.

## The local HAO-inspired candidate

The current design contract deliberately changes the ownership model instead
of adding another policy to the production scheduler.

Its intended TMEM map is exactly:

```text
S0 [0,128) | S1 [128,256) | O0 [256,384) | O1 [384,512)
```

It retains our input and numerical choices:

- Dqk192 and Dvo128;
- Q/K E2M1 plus E4M3 per-16 scales and FP32 head scalars;
- V and online P as E2M1 plus E8M0 per-32 scales;
- our ALU-oriented online softmax/P recurrence where enabled;
- causal adjacent-query tail handling.

It borrows HAO's ownership choices:

- 16 explicit warp roles;
- two query stages;
- permanent O0/O1;
- packed P in the upper half of retired score TMEM;
- Q/K and P/V scale pages in the low half of retired score TMEM;
- one tensor issue warp with a fixed command stream;
- one alternating, physically aliased K/V payload-and-scale ring.

The initial correctness build is intentionally depth two. D3, D4, and D5 are
separate resource experiments after correctness, not assumptions baked into
the first build. The source-derived dynamic shared-memory sizes are 93,184 B,
107,520 B, 120,832 B, and 135,168 B respectively before linked static shared
memory.

The lifecycle and no-cycle proof are recorded in:

`results/mxfp4_fa4_upstream_strategy_20260722/upstream_stage_local_candidate_lifecycle_20260722.md`.

As of this note, that document is a design gate. An early implementation was
withheld because it still used separate K/V allocations and placeholder
correction/epilogue code. It must not be described as landed, correct, or
faster until the source contract, build, correctness suite, and benchmark
matrix pass.

## Recommended comparison matrix after the candidate lands

Run the same pre-generated BF16 source tensors through all providers and keep
Q/K/V quantization outside every timed call. For each cell, record wall time,
correctness, launch topology, registers, shared memory, TMEM columns, tensor
activity, eligible warps, and source-counter attribution.

Required providers:

1. Frozen TK MXFP4 production artifact.
2. HAO-inspired local full-FP4 candidate.
3. HAO pinned kernel in an isolated dependency environment.
4. TK BF16 FA4, both full-grid and persistent where supported.
5. CuTe-DSL BF16 FA4.
6. Local FP4-QK plus plain-FP8-PV candidate, if implemented.

Required batch/sequence/head cells should include HAO's published list plus
our high-head stress cells:

```text
B1/S256/H16
B1/S1024/H16
B1/S2048/H16
B1/S4096/H24
B1/S4096/H32
B1/S4096/H64
B1/S8192/H32
B1/S16384/H32
B1/S32768/H24
B1/S32768/H32
B4/S4096/H16
B4/S4096/H32
```

Run causal and noncausal as separate suites. First reproduce each provider's
native dimension contract: HAO Dqk128/Dvo128 and local Dqk192/Dvo128. A true
equal-dimension comparison additionally requires either a valid HAO
Dqk192/Dvo128 specialization or a valid local Dqk128/Dvo128 specialization;
do not silently label either native run as equal work. Do not compare a
persistent noncausal provider with a full-grid causal provider without
showing the topology difference.

## Bottom line

HAO and our investigation reached the same hardware-level fact: block-scaled
FP4 scale fragments consume TMEM at MMA issue, and 512 columns do not permit
two score banks, two output banks, and a permanent scale slab at once.

HAO's solution is to remove the permanent slab. It stores transient scales
and packed P inside score banks after the score has been read, keeps two real
output banks, packages two queries per CTA, and uses a deterministic one-warp
`PV -> QK` command stream. Its deep K/V pipeline is in shared memory.

Our frozen production solution preserves a scale slab, moves P to shared
memory, and borrows a score bank as the second output. That makes the score
bank's lifetime longer and the reuse protocol substantially more complex.
At H64, it also leaves us with one query per CTA and twice the BF16 service
waves. Those structural choices now matter more than another local exp2 or
barrier micro-optimization.

The direct next experiment is therefore the standalone S0/S1/O0/O1 rewrite,
not another heuristic layered onto the existing hot-plate scheduler. It must
retain our per-32 numerical path initially so the result isolates ownership
and packaging. After that works, FP8 PV is the highest-value alternate mode
because HAO's own results show that removing block-scaled PV traffic opens the
largest remaining performance margin.

## Source index

HAO pinned source:

- `flash_attn/cute/flash_fwd_sm100_fp4.py:119-255`: tile, query stages, warp
  roles, and TMEM offsets.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:370-424`: shared-memory budget and
  derived K/V ring depth.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:624-630`: one-CTA requirement and P
  TMEM source.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:1080-1124`: mbarrier families and P
  split.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:1520-1700`: K/V aliasing and scale
  placement in score banks.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:1999-2214`: alternating K/V producer.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:2423-2637`: QK/PV issue loop.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:3110-3295`: fused P quantizers.
- `flash_attn/cute/flash_fwd_sm100_fp4.py:3627-3870`: score load, retirement
  signal, P construction dispatch, and split P store.
- `flash_attn/cute/benchmarks/bench_fp4.py:555-655`: host quantization.
- `flash_attn/cute/benchmarks/bench_fp4.py:711-813`: timing and shape suite.
- `flash_attn/cute/README.md:72-106`: GB300 performance table.
- `flash_attn/cute/README.md:161-177`: precision table and exact formats.

Frozen TK source/evidence:

- `results/mxfp4_fa4_production_corr_pv_ex2_20260717T221047Z/source_snapshot/fp4_fa4_fwd/fwd_host_dispatch.inc:6121-6122`:
  exact route instantiation and full-grid launch argument.
- Frozen `fwd_streaming_kernel.inc:3853-3931`: score/output/scale widths.
- Frozen `fwd_streaming_kernel.inc:4888-4931`: P/V scale epochs and TMEM end.
- Frozen `fwd_streaming_kernel.inc:5193-5215`: production warpgroup roles.
- Frozen `fwd_streaming_kernel.inc:5537-5552`: readiness objects.
- Frozen `fwd_streaming_kernel.inc:10497-10510`: score-bank output alias.
- Frozen `fwd_streaming_kernel.inc:27628-27674`: per-N32 E8M0 P scale math.
- Frozen `fp4_pv_experiments.py:1762-1817`: Q/K input contract.
- Frozen `fp4_pv_experiments.py:2508-2556`: Q/K v5 quantization and head scalar.
- Frozen `fp4_pv_experiments.py:15787-15902`: V MXFP4 quantization.
- `results/mxfp4_fa4_production_corr_pv_ex2_20260717T221047Z/h64_production_bf16_reproduction_timing_lock.json`:
  artifact identity and launch topology.
- `results/mxfp4_fa4_production_corr_pv_ex2_20260717T221047Z/h64_existing_profile_analysis_20260719.json`:
  protected timing, NCU counters, static SASS, stage attribution, and causal
  synthesis.
- `results/mxfp4_fa4_upstream_strategy_20260722/k_slot_phase_prebuild_proof.md`:
  exact local K/K-scale slot and phase proof.
- `results/mxfp4_fa4_upstream_strategy_20260722/upstream_stage_local_candidate_lifecycle_20260722.md`:
  unlanded HAO-inspired candidate contract.

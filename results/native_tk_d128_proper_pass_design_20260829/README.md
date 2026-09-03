# Native ThunderKittens D128 causal-GQA backward proper-pass design

Date: 2026-08-29

## Verdict

A proper native ThunderKittens (TK) D128 causal-GQA backward is implementable,
but it is new kernel development. No recovered native TK kernel is currently
faster than CuTe DSL for the Llama-8B geometry.

The old native D128 fallback is the wrong topology: it uses a KV-head grid,
serializes the four query heads belonging to each KV head, performs work on
fully masked causal tiles, uses BF16 HMMA rather than SM100 `tcgen05`, and
spills heavily. The D64 v385 experiment has the right K128/Q128 ownership and
genuine `tcgen05` primitives, but it is single-buffered, repeatedly synchronizes
the full CTA, and already spills 120 bytes per thread at D64. Widening it with a
D128 typedef would not be a credible implementation.

The bounded native pass should instead copy the authenticated CuTe topology:
one 512-thread CTA per `(batch, query_head, K128 tile)`, two compute warp-groups,
one tensor-issue warp, one TMA-load warp, four dQ drain warps, K/V persistence,
Q/dO double buffering, and the full 512-column TMEM map. The initial semantics
should retain CuTe's BF16 split-query-head dK/dV partials and ratio-4 FP32
reducer. A direct BF16 TMA `RED.ADD` dK/dV variant is the most credible later
way to remove the partial workspace and reducer.

Production remains on CuTe DSL unless the native public backward boundary—not
merely its main kernel—passes the acceptance checks below.

## Scope and evidence boundary

This was a read-only design and ISA audit. It did not edit a kernel, build a new
candidate, launch a benchmark, touch a GPU job, or change cluster state. The
only repository changes are this receipt and its `summary.json`.

The active worktree at audit time was:

```text
/workspace/codebases/pv/fp4_matmul_monolithic_tk_20260828
branch: codex/monolithic-tk-fa4-train-20260828
HEAD:   1ede7dc715a58ea91165997b78d420439a3585b3
```

Terms used here are deliberately strict:

- **Native TK** means CUDA/C++ source that directly programs ThunderKittens
  types and SM100 operations.
- **CuTe DSL** means the generated CUTLASS CuTe DSL backward, even when the
  harness or generated module lives under `tk_fa4`.
- **Verified** means read from authenticated source, PTX/CUBIN, SASS, a fresh
  matched receipt, or device properties.
- **Proposed** means the native design derived from those facts. It is not a
  performance result.

Static PTX/SASS instruction counts below are instruction sites, not dynamic
execution counts. No NCU profile was collected in this audit, so they do not
establish stall percentages, issue utilization, or memory-bandwidth use.

## Authenticated CuTe D128 target

The exact generated source used for this audit was reconstructed by the active
patch loader and authenticated before compilation:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Generated CuTe control source | 221,230 | `cfbd3ad27e5188d39c475abc238b57b5331fc7e631054a7075c7993150c70764` |
| SM100a CUBIN | 114,440 | `4963de5a48e41aea1e4cf6163d1682611becdcb199a779099a7ddfc14c12bc6c` |
| SM100a PTX | 184,037 | `89c04ad597dfc038c5ded5d0348497384b0ce30dd06526f91cbe82e71c21a793` |
| SASS dump | 1,126,725 | `52eb4ec8f72b76f93dc555217df890ec3d1d325bf3b84f5a0d0315aced13e931` |

The compiler artifacts are diagnostic files under
`/tmp/fa4_d128_exact_isa_20260829`; their hashes, rather than those temporary
paths, are the durable identity.

The compiled instance is the current low-precision backward for
`B=2, S=4096, Hq=32, Hkv=8, D=128, causal`. It consumes E4M3 Q, K, V, and dO,
precomputed FP32 log-sum-exp and dP-row-sum statistics, and writes BF16 dQ, dK,
and dV. It is not a pure-FP4 backward. Its principal settings are:

| Setting | Verified value |
| --- | --- |
| Macro tile | K128 x Q128 x D128 |
| Grid | `(32, 32, 2)` = `(S / 128, Hq, B)` |
| Block | 512 threads / 16 warps |
| CTA residency | One CTA per SM |
| Q stages | 2 |
| dO stages | 2 |
| dK/dV accumulation stages | 1 |
| P handoff | Shared-memory E4M3, reused for dS computation |
| dS | E4M3 with lift 256 |
| Exponential | Native EX2 (`period=0`) |
| Softmax beta | `(1 / sqrt(128)) / 16` for Q/K inputs published at 4x each |
| dQ output | Direct compact BF16 TMA reduction |
| dK/dV output | BF16 per-query-head partials plus fused ratio-4 reducer |

### Verified CuTe resources and ISA

The main CUBIN reports `REG=128`, `STACK=64`, `LOCAL=0`, and 1,024 bytes of
static/reserved shared memory. PTX requests 512 threads and one CTA per SM.
The generated shared structure has the following exact offsets:

| Region | Offset | Bytes |
| --- | ---: | ---: |
| Pipeline barriers/control, followed by alignment | 0 | 1,024 |
| K, one E4M3 K128 x D128 tile | 1,024 | 16,384 |
| V, one E4M3 K128 x D128 tile | 17,408 | 16,384 |
| Q, two E4M3 Q128 x D128 stages | 33,792 | 32,768 |
| dO, two E4M3 Q128 x D128 stages | 66,560 | 32,768 |
| P, one E4M3 K128 x Q128 stage | 99,328 | 16,384 |
| dS, one E4M3 K128 x Q128 stage | 115,712 | 16,384 |
| dQ drain, two BF16 Q128 x N32 stages | 132,096 | 16,384 |
| LSE | 148,480 | 512 |
| Alignment gap | 148,992 | 512 |
| dP row sum | 149,504 | 512 |
| Tail alignment | 150,016 | 512 |

CuTe rounds the structure to its strictest 1,024-byte member alignment, so the
exact dynamic shared allocation is **150,528 bytes**. Together with the CUBIN's
1,024 static/reserved bytes, the main kernel has a 151,552-byte shared-memory
envelope. The local GB200 reports 233,472 bytes of shared memory and 65,536
registers per SM; the register split alone therefore makes this a one-CTA
design.

The generated PTX contains 40 `tcgen05.mma` sites, 64 native EX2 sites, six
bulk tensor-load sites, and four tensor-reduction sites. The ELF records 24
mbarriers. These are static sites, but they authenticate that the target is an
SM100 `tcgen05`/TMEM/TMA pipeline rather than a relabeled CUDA fallback.

The exact warp and TMEM ownership is:

| Owner | Function |
| --- | --- |
| Warps 0-3 | dQ TMEM drain, BF16 conversion, and TMA reduction |
| Warps 4-11 | Two compute warp-groups for softmax, P, dP correction, and dS |
| Warp 12 | `tcgen05` tensor-issue producer |
| Warp 13 | TMA load producer |
| Warps 14-15 | Spare/control roles |

| TMEM columns | Lifetime |
| --- | --- |
| 0-127 | Persistent dK accumulation |
| 128-255 | Persistent dV accumulation |
| 256-383 | dP, then time-aliased by dQ after dS publication |
| 384-511 | Score S while compute produces P |

The separate fused convert/GQA-reduce kernel uses 256 threads, 40 registers,
zero stack, and 1,024 static/reserved shared bytes.

## D64 v385: useful primitives, unsafe D128 schedule

The genuine native TK D64 source is:

- [`v385_d64_gqa_e4m3_k128q128.cuh`](../../tk_fa4/native_gqa_tk_bwd/v385_d64_gqa_e4m3_k128q128.cuh), SHA-256
  `11ee92a96d883c15af41c8a23b552350b594000ffee152da45b38516ecca4d70`;
- [`v385_d64_gqa_e4m3_k128q128.cu`](../../tk_fa4/native_gqa_tk_bwd/v385_d64_gqa_e4m3_k128q128.cu), SHA-256
  `74fde07b11c5b1954c6c12bea02ecd052c2feb63f162dc04be8e3ae01c3a25d0`.

Its authenticated diagnostic binary is 2,283,880 bytes with SHA-256
`0b62572a70b2575a984630d0703c81a0490dda616fb8b2515f030131c5a7bc09`.
The v385 main kernel reports `REG=128`, `STACK=120`, `LOCAL=0`, and
`SHARED=67,696` bytes.

v385 gets several important choices right: K128/Q128 ownership, a 512-thread
CTA, an Hq grid, persistent K/V, corrected E4M3 `tcgen05` AB and AtB helpers,
direct BF16 TMA reduction, and eight compute warps. Those helpers are the safe
material to carry forward.

Its schedule is not safe to widen:

- Q and dO use one shared buffer instead of a producer/consumer pipeline.
- Full-CTA `__syncthreads()` calls separate the steady-state phases.
- Eight compute warps retain wide score/P/dS fragments at a requested 160
  registers per thread, producing the observed 120-byte stack at D64.
- dP, score, and dQ are serialized through overlapping TMEM rather than using
  the 512-column D128 overlap plan.
- Direct BF16 dK/dV adds change GQA reduction order and are not yet validated
  for D128.

The prior 5.864 ms native measurement belongs to v384, not v385. No timing is
claimed for v385 in this receipt. For context, the authenticated D64 CuTe cd57
control has source SHA-256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`,
uses K128/Q128, block 512, `REG=128`, `STACK=0`, 76,800 dynamic plus 1,024
static/reserved shared bytes, and measured 3.232 ms for its main path at the
matched B16 geometry. Its `run(reset=True)` public boundary measured 3.345824
ms in the fallback receipt. That difference in boundary matters.

## Legacy native D128 negative baseline

The surviving generic native implementation and lineage are documented in
the [native D128 provenance audit](../native_tk_d128_provenance_audit_20260829/README.md).
The relevant source is:

- [`fa4_bwd_unified_sm100.cuh`](../../tk_fa4/deprecated/fa4_bwd_unified_sm100.cuh);
- [`fa4_bwd_dkdv_sm100.cuh`](../../tk_fa4/deprecated/fa4_bwd_dkdv_sm100.cuh);
- [`fa4_bwd_dq_sm100.cuh`](../../tk_fa4/deprecated/fa4_bwd_dq_sm100.cuh);
- [`deprecated/tk_fa4.cu`](../../tk_fa4/deprecated/tk_fa4.cu).

Causal GQA fails both optimized native gates because they require noncausal
attention and equal Q/KV head counts. The selected generic causal D128 main
kernel uses a 256-thread block, `REG=255`, `STACK=352`, and `SHARED=100,352`.
Its grid is `(S / 128, Hkv, B)`, only 256 CTAs at the B1 target. Each CTA owns
one KV head/K128 tile, then serially processes four query heads and all Q16
subtiles. It reconstructs causal masks instead of pruning the fully masked
query prefix. Its static ISA uses BF16 HMMA sites rather than `tcgen05`/TMEM.

A fresh matched B1 public-boundary measurement was:

| Backend | Median | Correctness |
| --- | ---: | --- |
| Legacy native TK generic BF16 causal GQA | 6.578432 ms | finite; dQ/dK/dV cosine at least 0.9999957 vs CuTe |
| Exact-BF16 CuTe DSL | 0.479504 ms | reference |

The legacy boundary was 13.719243x slower. This is a negative topology
baseline, not a starting point for optimization.

## Proposed proper native D128 schedule

### CTA ownership and traversal

Use one CTA for `(batch, query_head, key_tile)`:

```text
grid.x = ceil_div(sequence, 128)
grid.y = query_heads
grid.z = batch
kv_head = query_head / 4
```

At B2/S4096/Hq32 this gives 2,048 main CTAs, four times the parallel query-head
ownership of the legacy Hkv grid. Each CTA loads its K128 and V128 tile once,
then traverses causal Q128 tiles from its key-tile index through the end of the
sequence. For S4096, the average CTA visits 16.5 query tiles. This preserves
CuTe's causal pruning and K/V reuse.

Do not replace this with a CTA per KV head: serializing four query heads is the
dominant legacy failure. Do not begin with a four-CTA cluster: every CTA needs
one SM's register/TMEM/shared envelope, so cluster reduction adds scheduling
and DSM risk without solving the main loop.

### Warp roles

Use 16 warps with explicit, non-overlapping jobs:

| Warps | Proposed native role |
| --- | --- |
| 0-3 | dQ TMEM-to-BF16 drain; publish Q128 in four N32 chunks |
| 4-7 | Compute WG0, responsible for one half of K rows |
| 8-11 | Compute WG1, responsible for the other half of K rows |
| 12 | Sole `tcgen05` issue warp and MMA completion producer |
| 13 | Bulk-TMA K/V/Q/dO load producer |
| 14 | FP32 LSE/dPsum async-copy producer and causal control |
| 15 | dQ TMA-store leader and final dK/dV epilogue/store control |

Only initialization, TMEM allocation, and final teardown may use a CTA-wide
barrier. The steady-state loop must use named mbarriers between load, MMA,
compute, and drain roles. This is the central change from v385.

### Per-query-tile pipeline

For each valid Q128 tile:

1. Warp 13 prefetches the next Q and dO stages while warp 14 publishes the
   matching statistics.
2. Warp 12 issues `S = K @ Q^T` into TMEM columns 384-511.
3. Compute WGs apply scale, causal mask, LSE correction, native EX2, and the
   retained probability lift, then publish E4M3 P to shared memory.
4. Warp 12 issues `dV += P @ dO` into persistent TMEM columns 128-255 and
   `dP = V @ dO^T` into columns 256-383.
5. Compute WGs consume dP and P, subtract the precomputed dP row sum, apply the
   softmax scale, and publish lifted E4M3 dS to shared memory.
6. After dP is dead, warp 12 issues `dK += dS @ Q` into columns 0-127 and
   `dQ = dS^T @ K` over the dP alias in columns 256-383.
7. Warps 0-3 convert dQ to two-stage BF16 N32 shared tiles while warp 15 issues
   TMA `RED.ADD` to the caller-zeroed compact dQ.

dK and dV remain in TMEM for the complete query-tile traversal. At loop exit,
the Q/dO shared stages are dead and may be reused as BF16 epilogue staging; no
additional peak shared allocation is needed.

### TMEM, shared memory, and registers

The first correct implementation should use the exact CuTe TMEM map and
150,528-byte dynamic shared layout documented above. Do not begin by aliasing
P with dS or score with dP: those aliases remove the overlap needed to hide
MMA/compute latency. A later proven schedule may target roughly 133-136 KiB by
shortening lifetimes, but shared-memory reduction alone cannot create a second
resident CTA because the register and TMEM budgets already limit occupancy.

Use SM100 register reconfiguration with this role budget:

```text
4 reduce warps  * 152 registers/thread * 32 threads = 19,456
8 compute warps * 128 registers/thread * 32 threads = 32,768
4 role warps    *  96 registers/thread * 32 threads = 12,288
total                                                   64,512
GB200 registers per SM                                  65,536
```

The critical native advantage must be lexical fragment lifetime and
`STACK=0`, not a larger register request. Score, P, dP, and dS fragment objects
must not remain simultaneously live across pipeline waits. A candidate that
repeats v385's local stack traffic is not a proper pass.

## ABI and GQA reduction plan

### Public tensor and descriptor ABI

Accept the model-native contiguous BSHD storage without a materializing
transpose:

```text
Q, dO: [B, S, Hq,  D] E4M3
K, V:  [B, S, Hkv, D] E4M3
LSE, dPsum: [B, Hq, 1, S] FP32
dQ: [B, S, Hq,  D] BF16, zeroed before the main kernel
dK, dV: [B, S, Hkv, D] BF16
```

The TMA descriptor should expose the existing physical storage as logical
`[S, D, query_head_within_group, kv_head, B]`; this is the same no-copy layout
used by the authenticated CuTe wrapper. Pass raw pointers, dimensions, and
strides through the native extension rather than constructing CuTe/DLPack
wrappers. Pointer alignment and tensor identity checks belong in the host
cache, outside the timed per-layer hot path.

The current producer's exact statistic sign and scale convention must be
carried through unchanged. In particular, do not infer a new sign from the
friendly `LSE`/`dPsum` names: the retained direct-workspace producer already
publishes the corrected pages expected by the backward. Authenticate those
pages against the CuTe control before replacing the wrapper ABI.

### Baseline: split-Hq BF16 partials

The semantic baseline matches CuTe:

- the main CTA owns one query head and writes one BF16 K128 x D128 dK partial
  and one dV partial;
- partial layout is `[B, Hq, S, D]` for each gradient;
- a 256-thread reducer maps `q_head -> kv_head`, sums the four partials in
  FP32, and converts once to BF16 `[B, Hkv, S, D]`;
- the current D128 sequence block is 64 and the reduction vector width is 4.

For B2/S4096/Hq32/D128, dK and dV partials consume 64 MiB each. With the two
FP32 statistic pages, the exact workspace is 136,314,880 bytes (130 MiB). This
is intentionally retained for the first correctness result.

### Performance variant: direct BF16 TMA add

After the baseline is correct, add an opt-in path that clears final dK/dV and
has the four query-head CTAs issue BF16 TMA `RED.ADD` directly to their shared
KV-head destination. The D64 direct-TMA mechanism is already authenticated and
is the safest primitive to copy. At ratio four, each final element receives
four partial contributions.

This variant removes the 128 MiB B2 partial workspace, its readback, and the
second kernel. It also changes accumulation order and introduces repeated BF16
rounding, so it may not silently replace the baseline. Its zeroing cost must be
included in public-boundary timing, and its numerical result must pass the
separate checks below.

## Dependency on D64 v387

No v387 file existed when this D128 resource audit began. Untracked v387 files
appeared concurrently in the shared worktree before this receipt's final
validation; they belong to separate active work and were not treated here as a
runtime-validated result. The D128 pass is therefore conditionally dependent
on v387's eventual runtime evidence, not on its filename or presence.

The concurrent compile-only snapshot is nevertheless useful evidence:

| v387 artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `v387_d64_gqa_e4m3_async_pipeline.cuh` | 20,994 | `037459345c4e6da0f4a0e49a909e75231a00642b8ed131a388b71cf16c48c290` |
| `v387_d64_gqa_e4m3_async_pipeline.cu` | 6,849 | `9fcad6b67f6e2533ad38d96d94e42b57b50a545a14c33879262fe865ba3eb362` |
| `Makefile.v387` | 1,023 | `b394ed92d54e7f7698a572876d19329b960523f42bfe33fcaff56012ba267734` |
| Diagnostic extension | 2,743,440 | `58fce638e8d62c4d5d6fb5f243d3e6d2e09577885d828870cd57a06439d27c71` |
| SM100a CUBIN | 1,510,752 | `f972db1da9b5d77e0d1c32b55aa2018ee6243f8f92e3d1e06d7636b3313469ac` |

Its main symbol compiles to `REG=128`, `STACK=0`, `LOCAL=0`, and 101,600
shared bytes, including 1,024 bytes beyond the 100,576-byte user allocation.
Its source has one prologue `__syncthreads()` and no steady-state CTA-wide
sync. The isolated SASS slice has 2,776 static instructions, including 40
`UTCQMMA`, 32 EX2, 12 TMA loads, three TMA reductions, and no LDL/STL sites.
This is compile/disassembly evidence only: no S128/S4096 correctness or timing
had run because all local GPUs were occupied.

The current register reconfiguration is feasible and matches the D128 budget:
compute warps 0-7 retain the compiler's 128-register allocation, publication
warps 8-11 request 152, and control warps 12-15 return registers at 96:

```text
8 * 128 * 32 + 4 * 152 * 32 + 4 * 96 * 32 = 64,512 registers
GB200 registers per SM                         = 65,536 registers
```

This removes the register-budget blocker, but compilation alone still does not
validate the pipeline's gradient semantics or latency.

Use v387 as the primitive base only if its stable D64 build demonstrates all
of:

1. corrected E4M3 `tcgen05` AB and AtB descriptors;
2. K128/Q128 ownership and causal pruning;
3. named-mbarrier steady-state staging rather than loop-wide
   `__syncthreads()`;
4. a register-reconfiguration sum no greater than 65,536 registers per CTA;
5. `STACK=0` and `LOCAL=0` in the final CUBIN; and
6. validated dQ/dK/dV semantics.

Even then, create a new D128 sibling source. Do not patch the TK submodule and
do not instantiate v387 with `D=128` unchanged. The D128 sibling must add the
full 512-column TMEM ownership, two-stage Q/dO loads, separate score/dP overlap,
four-warp dQ drain, and the split-GQA reducer ABI. If v387 does not satisfy the
six conditions, the safest base is the corrected MMA/TMA helpers in
[`native_gqa_tk_bwd_pipelined.cuh`](../../tk_fa4/native_gqa_tk_bwd/native_gqa_tk_bwd_pipelined.cuh),
not the v385 main loop.

## Acceptance checks

These checks prevent another source-label or main-kernel-only result from
reaching the 8B training route.

### 1. Provenance and compile resources

- The executed symbol must come from native TK CUDA/C++ and its CUBIN hash must
  be recorded; a CuTe-generated symbol cannot be labeled native TK.
- Compile for SM100a with block 512 and a K128/Q128/D128 tile.
- CUBIN must report `STACK=0` and `LOCAL=0`; `REG` must be no greater than 128
  at the launch-resource level.
- Dynamic shared memory must be no greater than 150,528 bytes and total
  dynamic plus static/reserved shared memory no greater than 151,552 bytes.
- Disassembly must contain the intended `tcgen05`, TMA, TMEM-load, and
  mbarrier operations. The steady-state loop must not contain a full-CTA
  barrier between every math phase.

### 2. Isolated correctness

Run matched B1 and B2 `S4096/Hq32/Hkv8/D128/causal` inputs against the
authenticated current CuTe low-precision control:

- all dQ/dK/dV values must be finite;
- each gradient cosine must be at least 0.9999;
- each relative L2 must be at most 0.02;
- each norm ratio must be in `[0.995, 1.005]`; and
- metrics versus exact BF16 must not regress the CuTe low-precision control by
  more than 2% relative or 0.001 absolute, whichever is larger.

For direct dK/dV add, additionally compare against the accepted partial route:
each dK/dV cosine must be at least 0.999 and relative-L2 degradation no greater
than 0.01. Preserve both routes if direct-add is faster but does not meet this
numerical condition.

### 3. Isolated performance

- Benchmark the same GPU, tensors, precomputed statistics, clock state, and
  reset semantics with interleaved CuTe/native samples.
- Report main-kernel timing for diagnosis and public-boundary timing for the
  claim. The native boundary must include output clears and the reducer or
  direct-add zeroing that its ABI requires.
- The partial route must be no more than 1% slower at either B1 or B2 and must
  beat the CuTe public boundary by at least 3% at B2 before it replaces CuTe.
- A direct-add result may be promoted only if it passes correctness and beats
  the accepted partial route after including zeroing.

### 4. 8B integration

Only after the isolated boundary passes should the route enter the saturated
Llama-8B B2 comparison. Record per-layer attention backward, aggregate model
backward, step latency, tokens/s/GPU, memory, and finite loss/gradient norms.
The model claim uses aggregate step speed, not the isolated main-kernel ratio.
Until then, additional GPUs do not make the existing native D128 fallback
faster; they only replicate a slow kernel.

## Bottom line

D128 is the right native target because it is the actual 8B geometry, but the
existing TK code is not already faster there. A credible pass is bounded and
structural: validate v387's primitive layer, implement the CuTe-derived
K128/Q128 two-WG pipeline as genuine native TK, preserve split-GQA semantics
first, eliminate all stack traffic, then test direct dK/dV reduction. If that
public boundary does not beat CuTe, use CuTe for the remaining 8B runs and
report the native attempt honestly.

# Native TK D128 causal-GQA backward optimization

Date: 2026-08-29

This receipt covers the isolated SM100 backward shape used by the Llama-8B
attention path: B1/B2, S4096, Hq32/Hkv8, causal, D128.  It does not claim an
end-to-end training speedup.  All accepted GPU measurements used one visible
NVIDIA GB200 and caller-owned BF16 outputs semantically reset inside each
measured backward boundary.  Most routes physically clear all outputs; the
single-writer owner4 route directly overwrites complete dK/dV tiles and clears
only additive dQ.

## Verified numerical and statistics ABI

The native TK kernels consume represented E4M3 Q/K/V/dO tensors with encoding
scale 4.  They publish additive BF16 dQ/dK/dV with output encoding scale 4 and
use a `1/256` gradient epilogue scale.

The two compared implementations do **not** consume the same LSE page:

- native TK: `lstat = 8 - LSE * log2(e)`
- retained generated CuTe D128: `lstat = -LSE * log2(e)`
- both: `dstat = -16 * sum(O * dO)`

The accepted comparison constructs a separate statistics state for each
implementation.  An earlier CuTe control run that supplied native TK's `+8`
lift to both implementations is invalid and is not used below.

The native candidates v422 through v426 and the corrected v428 passed the
deterministic S128 causal reference gate under the represented-E4M3 diagnostic
envelope, produced finite/nontrivial gradients, and produced exact zeros for
zero dO.  The broad gate thresholds (`cosine >= 0.99`, relative L2 `<= 0.15`,
norm ratio in `[0.85, 1.15]`) are diagnostic, not the measured quality claim;
the measured aggregate quality is reported below.

## Accepted rotated timing matrix

Protocol: 10 warmups, 41 CUDA-event samples per boundary, rotated execution
order, S4096, and reset-inclusive backward boundaries.  Medians are primary;
p25/p75 are included because the generated CuTe samples occasionally contain
a roughly 1 ms outlier while their central distribution remains tight.

| Candidate | B1 median (us) | B1 p25/p75 (us) | B2 median (us) | B2 p25/p75 (us) |
|---|---:|---:|---:|---:|
| generated CuTe P0 | 382.080 | 379.968 / 384.960 | 626.848 | 623.392 / 628.288 |
| TK v422 exact | 509.984 | 506.880 / 511.264 | 941.664 | 939.616 / 943.520 |
| TK v423 D2/P2 | 532.224 | 530.816 / 533.856 | 984.672 | 982.816 / 987.200 |
| TK v424 exact schedule | 478.336 | 477.056 / 479.904 | 886.528 | 884.000 / 887.360 |
| TK v425 D1/P2 | 528.192 | 526.720 / 529.152 | 978.464 | 976.704 / 980.000 |
| TK v426 v424+D2/P2 | 494.976 | 493.888 / 496.544 | 919.712 | 918.208 / 922.304 |

v424 improves v422 by 1.066x at B1 and 1.062x at B2 without changing the
numerical method.  It nevertheless remains 25.2% slower than CuTe at B1 and
41.4% slower at B2 (equivalently, CuTe/TK speed ratios of 0.799 and 0.707).

Aggregate quality versus exact BF16 attention on the same represented-E4M3
inputs:

| Candidate | B1 cosine / rel-L2 / norm | B2 cosine / rel-L2 / norm |
|---|---|---|
| generated CuTe P0 | 0.999664 / 0.025909 / 0.999606 | 0.999668 / 0.025761 / 0.999341 |
| TK v424 exact | 0.999649 / 0.026488 / 0.999661 | 0.999659 / 0.026119 / 0.999445 |
| TK v426 D2/P2 | 0.999648 / 0.026523 / 0.999487 | 0.999660 / 0.026081 / 0.999373 |
| TK v425 D1/P2 | 0.998737 / 0.054817 / 0.976804 | 0.998779 / 0.054212 / 0.976463 |

## Exponential approximation result

The forward-derived degree/period approximations do not close this backward
gap in the current schedule:

- D2/P2 halves static EX2 sites from 64 to 32, but adds ALU work and makes the
  v424 schedule 3.5% slower at B1 and 3.7% slower at B2.  Its numerical change
  is negligible relative to the represented-E4M3 probability payload.
- D1/P2 is also slower and materially worsens the gradient norm and relative
  L2 error.

Therefore v424 retains native EX2.  This is evidence about this specific
backward schedule, not a claim that forward SFU approximations are generally
unhelpful.

## Verified schedule progress and remaining gap

v424 makes two useful scheduling changes over v422:

1. It releases the score TMEM page after every compute warp has loaded the
   second score half, allowing EX2 and the E4M3 shared-P store to continue from
   registers while the issuer starts the next score MMA.
2. It submits dP(next) at the collective dQ-drained point, before waiting for
   current dK/dV solely for input-stage reuse.

Static selected-SASS accounting explains why this was not enough:

| Role | generated CuTe instructions | TK v422 | TK v424 |
|---|---:|---:|---:|
| compute body | 1,654 | 1,960 | 1,641 |
| reducer/publication body | 350 | 1,274 | 1,274 |
| whole selected body | 3,520 | 4,832 | 4,520 |

The v424 compute body is already approximately CuTe-sized.  The large static
remainder is gradient publication: TK's generic D32 drain expands to 24 small
`LDTM.x4` loads, 192 BF16 packs, 48 shared stores, and 12 additive TMA sites;
CuTe uses four `LDTM.x32` loads, 64 BF16 packs, 16 vector shared stores, and
four additive TMA sites.  TK also has substantially more synchronization and
control sites.  These are static counts across role-specialized paths, so they
localize the optimization target but do not by themselves prove dynamic stall
fractions.

## Direct owner-x32 publication (v429)

v429 replaces the generic gradient load with the audited owner-aligned
`LDTM.x32` mapping and converts directly into the existing BF16 D32 double
buffer.  It applies this path uniformly to dQ, dK, and dV and avoids v428's
FP32 shared-memory round trip.  The final vectorized build passes the B1 and B2
S128 reference, finite/nontrivial, and exact-zero-dO gates.

The source-level simplification is not sufficient by itself.  ptxas still
emits 192 BF16 pack operations, and the selected body falls only from 4,520 to
4,496 instructions.  In a fresh rotated run it is effectively tied with
v424:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 380.096 / 376.896 / 384.960 | 622.304 / 618.336 / 625.568 |
| TK v424 | 481.696 / 478.208 / 483.104 | 887.584 / 885.920 / 890.368 |
| TK v429 final | 478.848 / 476.608 / 482.976 | 886.976 / 884.064 / 888.768 |

That is a 0.59% B1 improvement and a 0.07% B2 improvement over v424.  dP
still waits until all four serial chunk conversions/publications have reached
the final dQ load, so the critical lifetime is materially unchanged.

The CuTe-parity follow-up is therefore a full dQ *register* evacuation: pack
the first D32 fragment, retain the other three raw fragments, release the
aliased TMEM page, and only then publish the four chunks.  This requires about
112 data registers per reducer lane and a 152-register reducer allocation,
which fits one CTA at 64,512 of 65,536 SM registers when compute/control roles
use 128/96 registers.

## Dynamic-counter availability

An Nsight Compute 2025.3.1 run reached the profiler-gated v424 launch on a
separate GB200 but failed with `ERR_NVGPUCTRPERM`.  No system permissions were
changed and no `.ncu-rep` was produced.  Achieved occupancy, eligible/issued
warps, tensor/HBM throughput, and dynamic stall fractions are therefore
unverified.  Static evidence still shows that both kernels have 512 threads,
128 compiled registers, about 150 KiB shared memory, 64 EX2 sites, and derive
to one CTA/SM; it cannot by itself be presented as a dynamic stall profile.

## Rejected optimization attempts

- v427 replaces nested TK issue helpers with fixed-issuer raw helpers.  Its
  selected SASS is instruction-for-instruction identical to v424.  The
  remaining TCGEN election/replay envelope is generated by ptxas, so this is a
  source-level no-op.
- The first v428 binary omitted `tensor_load_wait()` after raw x32 TMEM loads.
  That binary and every measurement made from it are invalid.
- Corrected v428 fully evacuates dQ to a 64 KiB FP32 shared tile before BF16
  conversion.  It is numerically equivalent but slower than v424: 492.576 us
  versus 480.224 us at B1, and 903.776 us versus 888.096 us at B2, in a fresh
  rotated run.  Early dP release does not repay the extra FP32 shared-memory
  store/reload.

v429 confirms that audited x32 ownership plus direct BF16 conversion is
correct but does not shorten the dQ lifetime enough.  Compute-warp dQ
publication remains lower priority because those warps overlap the next
score/P work while the reducers drain dQ.

## Full dQ evacuation (v430/v431)

v430 tested the most direct CuTe-style lifetime change: retain the complete
dQ accumulator payload in reducer registers, release the aliased TMEM page,
then convert and publish all four D32 chunks.  Its selected SASS had the
intended ordering but ptxas could not retain the payload under the one-CTA/SM
register budget: the kernel used a 264-byte stack with 260/264 bytes of spill
stores/loads.  It was rejected statically and never launched.

v431 instead evacuates all four dQ chunks directly to one 128x128 BF16 shared
buffer.  Only after all four reducer warps finish the complete evacuation does
the tensor issuer submit dP(next).  This costs 32 KiB of shared memory but
avoids v428's FP32 round trip and v430's spills.  The immutable selected kernel
uses `REG128`, `STACK0`, `LOCAL0`, zero spills, and 183,600 bytes of shared
memory; its 4,304 decoded instructions are 216 fewer than v424.  An independent
static audit verified every raw x32 TMEM load wait, the complete-dQ drain before
alias reuse, and the shared-buffer read-wait before overwrite.

Both B1 and B2 S128 gates passed, including finite/nontrivial gradients and
exact zero gradients for zero dO.  The B2 dQ/dK/dV cosines were
`0.999300/0.999297/0.999707` with relative-L2
`0.037434/0.037493/0.024232`; these are small-shape diagnostics rather than
the S4096 quality claim.

The fresh matched S4096 matrix is:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 393.472 / 389.568 / 398.784 | 644.448 / 641.664 / 650.656 |
| TK v424 | 480.352 / 477.664 / 482.912 | 889.888 / 887.744 / 892.064 |
| TK v429 | 477.792 / 475.360 / 481.536 | 885.472 / 882.720 / 888.480 |
| TK v431 | 453.088 / 451.360 / 454.176 | 834.112 / 832.256 / 835.360 |

v431 is 1.060x faster than v424 at B1 and 1.067x at B2, with essentially
unchanged aggregate quality (B1 cosine/rel-L2/norm
`0.999649/0.026486/0.999635`; B2
`0.999659/0.026115/0.999454`).  It nevertheless remains 15.2% slower than
CuTe at B1 and 29.4% slower at B2.  The result validates early full-dQ
evacuation as a material scheduling lever; it does not yet justify changing
the production route.

The remaining publication path still issues four D32 additive TMAs for each
of dQ, dK, and dV.  v432 tests whether that chunking is itself the bottleneck.

## Full-width gradient publication (v432)

v432 replaces the twelve D32 publication sites with one reusable 128x128
BF16 shared tile and one full-width BSHD additive TMA for each of dQ, dK, and
dV.  The source indexes the full tile with its own 128-byte swizzle; it does
not reinterpret v431's four independently swizzled D32 tiles.  A single proxy
fence follows the complete four-chunk fill.  Before every overwrite, reducer
warp 0/lane 0 waits until the previous TMA has stopped reading and releases a
four-warp reuse barrier.

The first exploratory build retained a proxy fence inside the unrolled
four-chunk fill and spilled 40 bytes in the dK/dV tail.  It was rejected
statically and never launched.  Hoisting the fence and serializing only the
outer four-chunk drain loop produced the accepted artifact: `REG128`,
`STACK0`, `LOCAL0`, zero spills, and 167,200 bytes shared.  Its selected SASS
has 3,624 instructions, three `UTMAREDG.5D.ADD` sites, 12 `MEMBAR`, 13
`FENCE`, and 15 `WARPSYNC`, versus v431's 4,304 instructions and twelve
additive TMA sites.  An independent audit verified the full-tile swizzle,
BSHD descriptor coordinates, all dQ-to-dK-to-dV phases for every positive
iteration count, and same-lane TMA issue/read-wait/completion ownership.

Both B1 and B2 S128 reference/finite/nontrivial/exact-zero gates passed.  A
fresh matched S4096 rotation shows that the large static simplification is
not a material latency win:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 376.704 / 374.592 / 383.136 | 620.160 / 615.040 / 622.336 |
| TK v431 | 456.480 / 453.888 / 457.376 | 836.768 / 833.984 / 838.688 |
| TK v432 | 456.192 / 454.496 / 458.144 | 834.656 / 832.480 / 836.256 |

v432 is only 0.06% faster than v431 at B1 and 0.25% at B2, well inside the
central-distribution width.  Aggregate quality is unchanged: v432 B1
cosine/rel-L2/norm is `0.999649/0.026487/0.999650`, and B2 is
`0.999659/0.026116/0.999453`.  This rules out source-site count as the main
remaining limiter in isolation.  A plausible interpretation is that the
coarser 32 KiB reductions and surrounding critical lifetimes offset the
instruction/control reduction; dynamic counters remain unavailable, so that
mechanism is an inference rather than a measured stall attribution.

## Equal-work CTA raster (v433)

v433 changes only the two CTA-coordinate sources and the corresponding launch
grid: query head becomes the fast `grid.x` dimension and key tile becomes
`grid.y`.  This schedules the 32 equal-length query-head CTAs for one causal
key tile together instead of presenting a scheduler wave with the full
32-to-1 iteration range for one head.  The selected SASS differs from v432 in
exactly two instructions (`CTAID.X`/`CTAID.Y` are exchanged); the other 3,622
instructions are byte-identical.  Resources remain `REG128`, `STACK0`,
`LOCAL0`, zero spills, and 167,200 bytes shared.

Both B1 and B2 S128 reference/finite/nontrivial/exact-zero gates passed under
the same diagnostic envelope as v432.  Two independent 10-warmup/41-sample
S4096 rotations reproduced the gain:

| Rotation | Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---|---:|---:|
| R1 | generated CuTe P0 | 374.080 / 372.512 / 375.616 | 619.104 / 615.328 / 624.000 |
| R1 | TK v432 | 457.696 / 454.496 / 459.744 | 835.872 / 832.864 / 837.824 |
| R1 | TK v433 | 405.600 / 403.328 / 407.872 | 782.976 / 780.480 / 784.704 |
| R2 | generated CuTe P0 | 383.936 / 376.480 / 388.192 | 633.984 / 629.696 / 640.224 |
| R2 | TK v432 | 459.136 / 455.104 / 460.544 | 835.968 / 833.376 / 838.400 |
| R2 | TK v433 | 407.744 / 405.152 / 410.496 | 783.008 / 780.800 / 785.696 |

The v433/v432 latency ratios are 0.8862/0.8881 at B1 and 0.9367/0.9366 at
B2: a reproduced 11.2--11.4% B1 reduction and 6.3% B2 reduction with no
numerical-method change.  v433 remains 6.2--8.4% slower than generated CuTe
at B1 and 23.5--26.6% slower at B2 in these rotations, so it is the leading
native candidate but not yet the production route.

## Rejected paired-query-head cluster (v434)

v434 tests the smallest GQA-aware transfer from the historical BF16 donor.
Two adjacent query-head CTAs form a `2x1x1` cluster; rank 0 multicasts K/V,
both ranks retain private Q/dO/statistics and dQ, and rank 0 combines rank 1's
BF16 dK/dV tile through DSM before the only pair-level additive TMA.  The
selected kernel is `REG128`, `STACK0`, `LOCAL0`, zero spills, and 167,232
bytes shared.  It has two multicast loads, eight vector DSM loads, and three
full-width gradient publications.

The B1/B2 S128 gates passed, including exact-zero dO.  The matched S4096
rotation rejects the schedule for performance:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 375.264 / 371.264 / 378.208 | 619.392 / 616.096 / 623.648 |
| TK v433 | 407.552 / 404.704 / 409.952 | 784.704 / 781.728 / 786.560 |
| TK v434 cluster2 | 448.672 / 446.080 / 450.848 | 867.872 / 865.536 / 870.464 |

v434 is 10.1% slower than v433 at B1 and 10.6% slower at B2.  Its output
metrics remain equivalent to v433 within additive-BF16 ordering noise.  The
verified traffic saving is therefore outweighed by cluster/DSM synchronization
and pair-combine work; the candidate is retained as a negative result and is
not routed.

## Flattened batch/head raster (v435)

v435 tests whether v433 leaves a B2 scheduler bubble by flattening batch and
query head into `grid.x`: `(B*Hq, S/128, 1)`.  It decodes query head with
`&31` and batch with `>>5`; the substantive body, instruction count, resource
use, and numerical method remain v433-equivalent.  B1/B2 S128 gates passed.

The matched S4096 rotation is neutral:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 375.392 / 372.640 / 381.696 | 617.952 / 616.128 / 622.912 |
| TK v433 | 407.712 / 404.416 / 409.248 | 784.640 / 782.144 / 786.528 |
| TK v435 batch-head | 406.720 / 404.000 / 407.840 | 784.416 / 782.208 / 785.856 |

The v435/v433 ratios are 0.9976 at B1 and 0.9997 at B2, inside the paired
distribution width.  The missing B2 performance is therefore not explained
by keeping the two batches in separate `grid.z` slabs.

## Exact-S4096 and runtime accumulation predicate (v436)

v436 transfers two bounded mechanisms from the authenticated BF16 TK donor.
S4096 dispatches a compile-time 32-query-tile kernel; all other valid sequence
lengths fall back to unchanged v433.  The first K32 E4M3 dV and dK MMAs use a
runtime accumulate predicate, while later chunks preserve the corrected
`2*chunk` B-descriptor offset.  The selected kernel remains `REG128`,
`STACK0`, `LOCAL0`, zero spills, and 167,200 bytes shared.

Static selected SASS falls from 3,624 to 3,512 instructions (-3.09%), with
28 versus 36 tensor-MMA sites, 229 versus 243 branches, and the dynamic
sequence load/division removed.  The two first-MMAs lower with predicate
`UP0`; subsequent accumulations retain `UPT`.

Authenticated B1/B2 S4096 validation passed the mandatory S128 reference,
target finite/nontrivial, and full-shape exact-zero gates.  The matched
10-warmup/41-sample rotation measures a small but clear win:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 371.872 / 368.864 / 375.840 | 618.624 / 615.392 / 622.496 |
| TK v433 | 407.296 / 402.848 / 408.832 | 785.056 / 781.696 / 786.272 |
| TK v436 | 400.800 / 397.632 / 402.272 | 772.608 / 769.728 / 775.648 |

v436 is 1.59% faster than v433 at both batches, with v433-equivalent output
metrics.  At this point in the lineage it is the fastest single-query-head
native candidate and the best B1 route, but remains 7.8% slower than generated
CuTe at B1 and 24.9% slower at B2 in this rotation.

A later four-way rotation independently repeats the v436/v433 reduction at
1.95% for B1 and 1.71% for B2 (401.632 versus 409.600 us and 773.088 versus
786.560 us respectively).

## B2 sequential two-query-head ownership (v437)

v437 removes v434's cluster and DSM costs.  At B2, one CTA owns two adjacent
query heads from the same KV head, loads K/V once, uses one monotonic semaphore
phase index across both heads, publishes dQ independently for each head, and
retains FP32 dK/dV in TMEM until both heads have accumulated.  It then drains
and publishes dK/dV once per pair.  B1 dispatches unchanged v433.

The owner kernel remains `REG128`, `STACK0`, `LOCAL0`, zero spills, and
167,200 bytes shared.  B2 has 1,024 CTAs (6.74 one-CTA/SM waves) instead of
v433's 2,048; total attention-tile work is unchanged.  Authenticated B2 S128
and S4096 validation passed the reference/finite/nontrivial/exact-zero gates.

The matched 10-warmup/41-sample rotation is:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 382.496 / 378.496 / 385.888 | 625.376 / 622.880 / 629.984 |
| TK v433 | 409.600 / 407.520 / 411.072 | 786.560 / 784.352 / 788.384 |
| TK v436 | 401.632 / 399.808 / 404.448 | 773.088 / 770.848 / 774.752 |
| TK v437 owner2 | 406.176 / 404.896 / 408.160 | 712.672 / 711.136 / 713.952 |

At B2, owner2 is 9.39% faster than v433 and 7.81% faster than v436, with
equivalent output metrics.  It reduces the native/CuTe gap to 14.0% latency
(CuTe/native speed ratio 0.8775).  The B1 value is the v433 fallback plus
rotation noise, not an owner2 result.  This verifies that single-CTA GQA
ownership is useful when B2 retains enough waves; the next composition is the
v436 exact/predicated inner schedule inside owner2.

## Exact owner2 composition and B1 ownership (v438/v439)

v438 composes the exact-S4096/runtime-first-accumulate inner schedule with
v437's B2 owner2 schedule.  The selected kernel remains `REG128`, `STACK0`,
`LOCAL0`, zero spills, and 167,200 bytes in the same total shared-resource
envelope.  Static selected SASS falls from 3,704 instructions in v437 to
3,592, and tensor-MMA sites fall from 36 to 28.  Authenticated full B2/S4096
validation passed the mandatory S128 reference, finite/nontrivial, and
poisoned-output exact-zero-dO gates.

The matched composition rotation is:

| Candidate | B1 median / p25 / p75 (us) | B2 median / p25 / p75 (us) |
|---|---:|---:|
| generated CuTe P0 | 380.352 / 376.576 / 386.144 | 625.056 / 622.272 / 629.344 |
| TK v436 | 403.584 / 402.272 / 405.280 | 775.072 / 773.568 / 776.704 |
| TK v437 owner2 | 407.104 / 404.800 / 409.024 | 712.704 / 711.968 / 715.424 |
| TK v438 exact owner2 | 401.120 / 399.520 / 403.072 | 709.472 / 707.616 / 710.400 |

The exact inner schedule is a small 0.45% B2 improvement over owner2 rather
than a second large step.  v438's B1 value is the v436 fallback.

v439 then enables the same two-query-head exact owner kernel at B1.  Its
512-CTA grid provides 3.37 one-CTA/SM waves, and its selected body is
byte-identical to v438 after the function-name line.  Full B1/S4096
validation passed, including exact zero gradients after all three outputs
were deliberately prefilled with ones.  In two independent matched rotations
the native timing was stable while generated CuTe shifted across runs:

| Rotation | generated CuTe P0 (us) | TK v436 fallback (us) | TK v439 B1 owner2 (us) | CuTe / v439 |
|---|---:|---:|---:|---:|
| R1 | 370.080 | 400.608 | 377.152 | 0.9812x |
| R2 | 396.160 | 401.312 | 379.680 | 1.0434x |

Thus v439 is reproducibly 5.7--6.2% faster than the prior native B1 route and
lands near CuTe parity: it loses one rotation by 1.9% and wins the other by
4.3%.  The cross-process CuTe shift prevents a stronger universal speedup
claim; the stable native medians do establish that the historical BF16
persistent-work idea transferred successfully to this D128 GQA shape.

## Owner4 and unique-writer dK/dV (v440--v442)

v440 extends ownership to all four query heads associated with one KV head.
At B2/S4096 this reduces the grid to 512 CTAs, K/V loads to one per
`(batch, kv_head, key_tile)`, and dK/dV publication to one pair of tiles per
owner.  It remains `REG128` with no stack, local memory, or selected-kernel
spills.  A matched rotation measured 711.424 us for v440 versus 711.520 us
for v438; owner4 alone is neutral because the traffic reduction is offset by
fewer waves and slightly worse triangular work balance.

Owner4 also makes each dK/dV output tile single-writer.  v441 exploits that
property: dQ remains an additive TMA and is physically cleared, while dK/dV
use two plain direct TMAs and are not pre-cleared.  The selected SASS has
3,568 instructions and exactly one additive plus two direct full-gradient
stores.  An independent source audit proved the owner-to-KV-head mapping is
bijective and that every full dK/dV destination tile is overwritten exactly
once.  Full B2/S4096 validation passed with outputs prefilled before the
zero-dO gate, ruling out stale-output dependence.

Two matched rotations reproduce the native result:

| Rotation | generated CuTe P0 (us) | TK v438 fallback (us) | TK v441 direct owner4 (us) | CuTe / v441 |
|---|---:|---:|---:|---:|
| R1 | 619.296 | 711.040 | 698.944 | 0.8860x |
| R2 | 640.448 | 713.504 | 698.432 | 0.9170x |

v441 is 1.7--2.1% faster than the exact owner2 route and stable at about
698--699 us, but it remains 9--13% slower than generated CuTe at B2.  Its B2
aggregate cosine/relative-L2/norm is
`0.999663/0.025966/0.999439`, equivalent to the preceding native route and
slightly closer to the represented-input BF16 reference than the different
BF16 addition order in v438.

v442 tests owner4/direct dK/dV at B1.  It is valid and finite, but the
256-CTA/1.68-wave grid measures 386.080 us versus v439 at 379.680 us in the
same rotation.  It is rejected for B1; two-head ownership is the better
occupancy/reuse point there.

## Exact rounded-P register reuse (v443/v445)

v443 keeps the exact E4M3 words already rounded and published to shared
memory for dV in sixteen packed 32-bit registers per compute lane.  After dP
completes, dS expands those same words rather than reloading and converting P
from shared memory.  The shared P tile remains coordinate-correct and is still
the dV MMA operand; no pre-rounding FP32 probability is retained and the
numerical payload is unchanged.  At B2/S4096 this removes a logical 16 KiB
shared-P read for each of 33,792 causal attention-tile interactions, or about
528 MiB of aggregate shared reads across the launch.

The selected owner4 body remains `REG128`, `STACK0`, `LOCAL0`, zero-spill, and
167,200 bytes shared.  Its static instruction count falls from 3,568 in v441
to 3,424.  Full B2/S4096 validation passed the mandatory S128 reference,
finite/nontrivial, and poisoned-output exact-zero gates.  Two matched
rotations reproduce the latency change:

| Rotation | generated CuTe P0 (us) | v441 owner4 direct (us) | v443 compact P (us) | CuTe / v443 |
|---|---:|---:|---:|---:|
| R1 | 644.608 | 698.464 | 589.344 | 1.0938x |
| R2 | 647.904 | 699.360 | 589.280 | 1.0995x |

Thus compact rounded-P reuse reduces v441 latency by about 15.6% and moves
the B2 native route ahead of generated CuTe in both rotations.  v443's B2
aggregate cosine/relative-L2/norm is
`0.999663/0.025966/0.999439`; those values are indistinguishable from v441,
including separately identical dK/dV metrics and only additive-order noise in
dQ.

v445 applies the same mechanism to v439's B1 owner2 schedule.  The selected
body has 3,440 instructions versus v439's 3,592, with the same `REG128`,
zero-spill, and shared-memory envelope.  The final hash-bound artifact passed
the full B1/S4096 gate and measured 313.824 us standalone.  Two matched
rotations using that artifact and the later unified candidate matrix give:

| Rotation | generated CuTe P0 (us) | v439 owner2 (us) | v445 compact owner2 (us) | CuTe / v445 |
|---|---:|---:|---:|---:|
| R2 | 371.680 | 377.504 | 313.664 | 1.1850x |
| R4 | 378.400 | -- | 313.984 | 1.2052x |

The native median is stable while the fresh generated CuTe process moves.
The conservative claim is therefore a reproduced 1.185--1.205x isolated
speedup over this CuTe boundary, not the larger ratio from any single control
process.  v445's aggregate cosine/relative-L2/norm is
`0.999651/0.026432/0.999673`, numerically equivalent to v439.

## BF16-donor follow-ups and final schedule selection (v446--v453)

The authenticated historical BF16 donor suggested two bounded instruction
schedules after compact-P reuse:

- v446 packs the score-scale/lstat affine into FP32x2 operations.  It passes
  the numerical gate and reduces selected static instructions from 3,424 to
  3,248, but ptxas tail-merges the two probability-half paths: static
  `FFMA/MUFU/FMNMX` counts change from `128/128/128` to `0/64/64` with only
  32 `FFMA2` sites.  The merged path executes dynamically twice and regresses
  to 609.824 us versus v443 at 591.776 us in the same rotation.  Compile-time
  half specialization (v448) and ptxas `--dont-merge-basicblocks` (v450)
  both produce instruction words byte-identical to v446, so this transfer is
  rejected.
- v447 moves the first dP TMEM load after the completed-MMA wait and mandatory
  tensor-proxy acquire but before the previous dQ/dK waits.  Its
  `tensor_load_wait()` remains at the original dS consumer.  Resources and
  the complete 108-opcode histogram are identical to v443; only the LDTM
  position changes.  It passes the full B2 gate and is slightly but
  consistently faster in two matched rotations: 589.984 versus 591.776 us,
  then 589.856 versus 592.352 us.  The 0.30--0.42% gain is retained.

The compact-P mechanism also changes enough work that owner count was
rechecked rather than inherited from the noncompact lineage.  The final
matched matrix is:

| B | Candidate | Median / p25 / p75 (us) | CuTe / candidate |
|---:|---|---:|---:|
| 1 | generated CuTe P0 | 378.400 / 375.872 / 381.088 | 1.0000x |
| 1 | v445 owner2 compact | 313.984 / 313.184 / 316.512 | 1.2052x |
| 1 | v451 owner4 compact/direct | 314.944 / 314.240 / 316.032 | 1.2015x |
| 1 | v453 owner2 compact + dP preissue | 313.888 / 313.120 / 314.720 | 1.2055x |
| 2 | generated CuTe P0 | 621.728 / 619.840 / 625.024 | 1.0000x |
| 2 | v443 owner4 compact/direct | 592.352 / 591.168 / 593.824 | 1.0496x |
| 2 | v447 owner4 compact/direct + dP preissue | 589.856 / 589.056 / 590.880 | 1.0540x |
| 2 | v452 owner2 compact/additive | 632.320 / 631.200 / 634.656 | 0.9832x |

At B1, owner4 and dP preissue are inside the central-distribution width and
do not improve on v445 robustly; v445 remains the simpler selected route.  At
B2, owner2's extra waves do not repay duplicated K/V work and additive dK/dV
publication, while v447's small preissue gain repeats.  The selected pair is
therefore v445 for B1/S4096 and v447 for B2/S4096.  This is still an isolated
backward result on represented E4M3 inputs, not an end-to-end training claim.

## Frozen unified native route (v454)

v454 is a thin dispatcher over the two selected exact routes and the frozen
dynamic-sequence fallbacks.  B1/S4096 dispatches to v445's owner2 compact-P
kernel; B2/S4096 dispatches to v447's owner4 compact-P/direct-dK-dV kernel
with the first dP half preissued.  Nonexact B1 and B2 shapes retain v436 and
v437 respectively.  The public gradient boundary remains
`bfloat16_additive`: the out entry point logically resets all outputs, while
exact B2 physically clears only additive dQ because its unique writers
completely overwrite dK/dV.  Every selected body is `REG128`, `STACK0`,
`LOCAL0`, zero-spill, and uses 167,200 bytes of total shared memory.

Both hash-bound full gates passed, including the mandatory S128 reference,
finite/nontrivial S4096 outputs, and poisoned-output exact-zero check.  A
fresh matched 41-sample rotation against generated CuTe measured:

| B | Boundary | Median / p25 / p75 (us) | CuTe / v454 |
|---:|---|---:|---:|
| 1 | generated CuTe P0 | 381.376 / 377.728 / 388.928 | 1.0000x |
| 1 | v454 unified native TK | 315.360 / 310.528 / 318.208 | 1.2093x |
| 2 | generated CuTe P0 | 632.832 / 629.600 / 640.256 | 1.0000x |
| 2 | v454 unified native TK | 591.904 / 588.064 / 594.368 | 1.0691x |

The generated CuTe sample sets each contain one approximately 1 ms outlier,
so the median and interquartile ranges, rather than means, define the
comparison.  v454's aggregate cosine/relative-L2/norm is
`0.999651/0.026432/0.999673` at B1 and
`0.999663/0.025966/0.999439` at B2.  These exactly reproduce the selected
v445/v447 payloads.  This freezes an isolated D128 GQA backward candidate for
integration; it does not yet establish an end-to-end transformer speedup.

## Split-dV overlap and checkpoint route (v455--v458)

v455 shortens the B2 dV critical path without changing its E4M3 probability
payload.  The compute producer now publishes the first 64 probability columns
as soon as their shared stores and proxy fence are complete.  The tensor
issuer consumes that half for dV chunks 0--1, issues score(next) once the
second score half has released the aliased TMEM page, then consumes the second
probability half for dV chunks 2--3.  Only the second dV half commits
`dv_ready`, so consumers still observe completion of all four K32 commands.
The dP preissue and both dS consumers retain their v447 ordering.

The selected v455 body remains `REG128`, `STACK0`, `LOCAL0`, zero-spill, and
uses 167,200 bytes of total shared memory.  Its 3,448 selected instructions
are 24 more than v447's 3,424; the gain therefore comes from the changed
producer/consumer overlap, not static instruction removal.  The full B2 gate
passed.  In a fresh matched rotation:

| B2 boundary | Median / p25 / p75 (us) | CuTe / candidate |
|---|---:|---:|
| generated CuTe P0 | 617.056 / 614.112 / 623.392 | 1.0000x |
| v447 compact P + dP preissue | 590.752 / 587.872 / 592.352 | 1.0445x |
| v455 split dV | 577.632 / 575.744 / 579.104 | 1.0683x |

The CuTe set contains one 1,388.768 us outlier, so the median and
interquartile range are the appropriate comparison.  v455 is 2.27% faster
than v447 by median.  Their aggregate cosine/relative-L2/norm values are,
respectively, `0.999662857/0.025965809/0.999438740` and
`0.999662857/0.025965812/0.999438741`; dK and dV metrics are exactly equal in
the receipt and dQ differs only at additive-BF16 ordering scale.

Two follow-ups were numerically valid but rejected for latency.  v456 keeps
v447's schedule and interleaves four FP32x2 EX2 operations with each pair of
E4M3 packs and one vector shared store; it measures 591.968 us standalone.
v457 composes that cadence with split dV and measures 585.120 us standalone,
versus 578.592 us for v455 under the same full-validator timing procedure.
These standalone figures reject the transformations but are not substituted
for the matched CuTe matrix above.

v458 is the checkpoint thin dispatcher: B1/S4096 selects unchanged v445, B2/S4096
selects v455, and nonexact B1/B2 retain v436/v437.  Its embedded exact-route
SASS is byte-identical to the frozen standalone v445 and v455 streams, and
both v458 full gates passed (standalone medians 314.368 us at B1 and 579.808
us at B2).  Consequently the selected isolated speedups are inherited from
the hash-bound matched measurements of those exact bodies: 1.2093x for the
v445 B1 body in the v454 rotation and 1.0683x for the v455 B2 body in the R6
rotation.  No matched timing result is being inferred from merely similar
source: the embedded machine-code streams were checked byte-for-byte.  This
still does not establish a full-model or end-to-end training speedup.

## Quarter-dP pipeline and selected route (v459--v464)

The next BF16-donor transfer was the quarter-width dP pipeline.  The exact
historical donor is commit `ae44880835899937e90922d3a434fb6b3b6e7cf5`,
whose exported name is `v352_pipeline_first_dp_quarter_loads_internal` and
whose own comment calls it V339.  Commit `17cef0b` describes a different
next-Q/dO experiment as "V352 backward"; it is not the quarter-load donor.
No preserved timing receipt was found for the quarter-load donor, so only its
source schedule, not a historical speedup, is treated as verified evidence.

v459 transferred the donor's first-half x16 owner mapping but used an empty
register-consuming inline-assembly anchor.  ptxas legally sank the chunk-1
LDTMs to `0x6c70/0x6c80`, after chunk-0 LDS/scalar work had begun, so the
intended overlap did not exist in machine code and v459 was not GPU-tested.
v460 replaces that ineffective anchor with only
`tensor_before_thread_sync()`.  The selected SASS now issues chunk 0 at
`0x6360/0x6370`, chunk 1 at `0x6500/0x6510`, and begins chunk-0 LDS/FADD at
`0x6530/0x65b0`.  It remains `REG128`, `STACK0`, `LOCAL0`, zero-spill, one
barrier, 167,200 bytes total shared memory, and 3,448 instructions.  v461's
`__syncwarp()`-only spelling compiled to the same selected instruction stream;
v462's fence-plus-warp-sync spelling added a redundant NOP and was rejected in
favor of the more precise tensor-proxy anchor in v460.

Two independent 41-sample rotations reproduced the first-half result:

| Rotation | Boundary | Median / p25 / p75 (us) | CuTe / candidate |
|---|---|---:|---:|
| R7 | generated CuTe P0 | 624.096 / 621.024 / 629.664 | 1.0000x |
| R7 | v455 split dV | 579.616 / 576.960 / 581.248 | 1.0767x |
| R7 | v460 first-half quarter dP | 574.432 / 572.576 / 577.216 | 1.0865x |
| R8 | generated CuTe P0 | 617.536 / 611.328 / 622.048 | 1.0000x |
| R8 | v455 split dV | 579.232 / 576.640 / 580.256 | 1.0661x |
| R8 | v460 first-half quarter dP | 575.104 / 572.864 / 576.416 | 1.0738x |

Thus v460 is 0.90% and 0.71% faster than v455 by median in the two
rotations.  Its aggregate cosine/relative-L2/norm remains
`0.999662857/0.02596581/0.99943874`; dK/dV are exactly unchanged from v455 and
dQ differs only at additive-order scale.

v463 is the selected thin dispatcher.  B1/S4096 still embeds v445, B2/S4096
embeds v460, and nonexact B1/B2 still use v436/v437.  The embedded v445 and
v460 selected streams are byte-identical to their standalone streams.  Both
hash-bound full gates passed, including the mandatory S128 reference,
finite/nontrivial S4096 gradients, and poisoned-output exact-zero check;
standalone medians were 315.232 us at B1 and 576.544 us at B2.  Exact B2 still
physically clears only additive dQ because its unique writers fully overwrite
dK/dV; all other routes clear dQ/dK/dV.  The public ABI remains
`bfloat16_additive`.

v464 tested the obvious extension to quarter-split the second dP half as well.
Its compiled ordering and numerical gate both passed with unchanged resources
and instruction count, but two matched rotations rejected it:

| Rotation | Boundary | Median / p25 / p75 (us) | CuTe / candidate |
|---|---|---:|---:|
| R9 | generated CuTe P0 | 621.376 / 617.472 / 626.720 | 1.0000x |
| R9 | v460 first-half quarter dP | 574.848 / 573.024 / 577.792 | 1.0809x |
| R9 | v464 all-quarter dP | 579.744 / 577.376 / 582.560 | 1.0718x |
| R10 | generated CuTe P0 | 620.736 / 616.384 / 623.488 | 1.0000x |
| R10 | v460 first-half quarter dP | 576.576 / 572.800 / 578.624 | 1.0766x |
| R10 | v464 all-quarter dP | 580.736 / 578.752 / 582.304 | 1.0689x |

v464 is 0.85% and 0.72% slower than v460.  The second half has no intervening
dQ/dK waits to hide its first x16 load, so the extra split/fence is overhead
rather than useful latency hiding.  At this checkpoint v460/v463 remained
selected.  The isolated exact-shape speedups were 1.2093x for the unchanged
v445 B1 body and 1.0738--1.0865x for the v460 B2 body in its two fresh
rotations.  The later synchronization-elision pass below supersedes only the
B2 route.

## Historical BF16 TK donor boundary

The remembered BF16 TK win is authenticated, but it is not the current
shape.  Commit `780af75723816258737229e36ec72d05b6114ec4` contains a causal
MHA route at B1/S4096/H8 with Dqk=192, Dv=128 and a two-CTA cluster.  Its two
strict rotations measured 355.935 versus 389.312 us and 359.520 versus
399.424 us against CuTe (TK/CuTe 0.914 and 0.900).  The accepted artifact hash
is `82f961849d81cca64748bf59649943d60074d194814fa87025995d2135d9752a`;
the artifact is no longer present, so this is preserved hash-bound evidence,
not a fresh rehash.  The final broad BF16 matrix was W19/L16/U0.

Those historical measurements alone did not establish a D128 GQA speedup.
In particular, their owner-Q split filled an underoccupied MHA grid, whereas
this target already has several CTA waves; v434 confirmed that a direct
two-CTA GQA transplant regresses by about 10%.  The successful transfer was
instead single-CTA persistent ownership across GQA query heads, followed by
exact rounded-P register reuse.  That composition is what produces the
current v445/v455 isolated wins.  The packed-affine experiment also shows why
static instruction reduction cannot be assumed to transfer: ptxas changed
the dynamic path shape and made v446 slower despite a smaller selected body.

## Transitive completion elision and selected route (v465--v470)

The BF16 audit did not uncover a second hidden D128-GQA implementation.  Its
remaining useful lesson was to re-audit completion edges rather than preserve
every inherited barrier.  Two barriers in v460 had become redundant after the
compact-P and split-dV transformations:

- v465 removes `probability_consumed`.  dS consumes the retained per-warp
  compact E4M3 words, while dV is the only remaining reader of shared P.  The
  half-1 `dv_ready` commit follows all four K32 dV commands, so waiting the old
  iteration's `dv_ready` already proves that shared P can be overwritten.
- v466 removes `stats_consumed` but retains `stats_ready`.  The eight compute
  warps are the only lstat/dstat readers and collectively publish `ds_ready`
  after those reads.  Tensor issue waits `ds_ready`, completes dK/dV, and only
  then publishes the per-stage `operand_consumed`; the loader already waits
  that same old-stage epoch before refilling Q/dO and stats.  This is a strict
  transitive dependency, including across query-head boundaries because
  `work` is monotonic.
- v468 removes the tensor issuer's explicit `dv_ready` wait immediately after
  its `dk_ready` wait.  The same lane issues both dV halves before dK, and a
  later `tcgen05.commit` tracks all prior same-thread asynchronous TCGEN work;
  dK completion therefore proves dV completion for operand-stage reuse.  The
  compute-side `dv_ready` wait protecting shared P and the final drain remain.
- v469 removes the compute-side old `dq_ready` wait immediately before its
  retained old `dk_ready` wait.  The same tensor lane issues dQ and its commit
  before dK and its commit, so later dK completion proves both consumers have
  stopped reading old shared dS.  `dq_ready` remains initialized and committed
  for the reducers, and every `dq_drained` reuse edge remains intact.

All four changes were independently source-audited and passed the full B2/S4096
gate, including the mandatory S128 reference, finite/nontrivial S4096
gradients, and exact zero gradients for zero dO.  Aggregate quality versus the
exact BF16 reference is unchanged at cosine/relative-L2/norm
`0.999662857/0.02596581/0.99943874`.

v465 remains `REG128`, `STACK0`, `LOCAL0`, and zero-spill; it reduces the
selected body from 3,448 to 3,432 instructions and total shared allocation
from 167,200 to 167,184 bytes.  v466 remains at the same register/spill class,
reduces the body further to 3,376 instructions, and uses 167,168 bytes.  Its
28 tensor-MMA sites, 13 TMEM loads, and arithmetic payload are unchanged from
v465; the static reduction is synchronization/control code.  v468 and v469
remain `REG128`, `STACK0`, `LOCAL0`, zero-spill, and 167,168 bytes shared while
reducing the selected body to 3,368 and 3,360 instructions, respectively.
v469 retains 28 tensor-MMA sites, seven TCGEN completion commits, and 13 TMEM
loads.  Its exact rounded-E4M3 numerical payload is unchanged.

The native-to-native improvements reproduced in two rotations each:

| Rotation | Boundary | Median / p25 / p75 (us) | CuTe / candidate |
|---|---|---:|---:|
| R11 | generated CuTe P0 | 614.880 / 610.656 / 620.672 | 1.0000x |
| R11 | v460 probability barrier | 575.040 / 572.672 / 577.056 | 1.0693x |
| R11 | v465 dV-gated P reuse | 574.496 / 573.216 / 575.936 | 1.0703x |
| R12 | generated CuTe P0 | 613.024 / 608.512 / 618.048 | 1.0000x |
| R12 | v460 probability barrier | 575.776 / 572.992 / 577.280 | 1.0647x |
| R12 | v465 dV-gated P reuse | 574.912 / 573.184 / 576.832 | 1.0663x |
| R13 | generated CuTe P0 | 608.480 / 605.120 / 614.400 | 1.0000x |
| R13 | v465 retained stats barrier | 576.384 / 573.280 / 577.408 | 1.0557x |
| R13 | v466 operand-gated stats reuse | 572.640 / 570.336 / 574.144 | 1.0626x |
| R14 | generated CuTe P0 | 604.064 / 602.528 / 608.640 | 1.0000x |
| R14 | v465 retained stats barrier | 576.512 / 574.112 / 578.048 | 1.0478x |
| R14 | v466 operand-gated stats reuse | 573.216 / 570.560 / 575.360 | 1.0538x |
| R19 | generated CuTe P0 | 608.032 / 605.056 / 612.000 | 1.0000x |
| R19 | v466 retained dominated waits | 573.824 / 571.776 / 575.104 | 1.0596x |
| R19 | v469 later-dK-gated reuse | 571.296 / 569.728 / 573.312 | 1.0643x |
| R20 | generated CuTe P0 | 608.224 / 605.152 / 611.008 | 1.0000x |
| R20 | v466 retained dominated waits | 573.792 / 572.704 / 575.616 | 1.0600x |
| R20 | v469 later-dK-gated reuse | 572.096 / 569.760 / 573.536 | 1.0632x |

Thus v465 improves v460 by 0.09--0.15% in its matched rotations, and v466
improves v465 by 0.57--0.65%.  The CuTe-relative ratio is reported per fresh
rotation rather than spliced across runs because the generated control median
varied between experiment groups.  The stacked v468/v469 completion cleanup
improves frozen v466 by 0.30--0.44% in its final hash-bound R19/R20 rotations.

v470 supersedes v467 as the selected thin dispatcher: B1/S4096 embeds
unchanged v445, B2/S4096 embeds v469, and nonexact B1/B2 retain v436/v437.
The embedded v445 and v469 SASS streams are byte-identical to their standalone
streams at 3,440 and 3,360 instructions.  Both hash-bound full gates passed;
the unified artifact's gate medians were 315.200 us at B1 and 573.120 us at
B2.  These gate medians are not substituted for a matched matrix.  Full-model
integration and end-to-end training speedup remain unmeasured.

## Head-boundary score overlap and selected route (v471--v473)

v471 tested whether the old shared-P reuse wait could be hidden behind the
next half-0 score-to-probability calculation.  The selected SASS put both
score loads, the affine/clamp/EX2 work, and the exact E4M3 pack before the
`dv_ready` poll, with all shared-P stores after it.  It remained `REG128`,
`STACK0`, `LOCAL0`, zero-spill, and used 167,168 bytes shared.  Its apparent
3,360-to-3,256 instruction reduction was ptxas merging duplicated half-1
branch bodies, not removal of dynamic arithmetic.  The longer packed-P live
range regressed both matched rotations, so v471 is rejected and its source is
not retained:

| Rotation | Boundary | Median / p25 / p75 (us) |
|---|---|---:|
| R21 | generated CuTe P0 | 604.480 / 601.568 / 608.128 |
| R21 | v469 later-dK-gated reuse | 571.456 / 569.152 / 574.016 |
| R21 | v471 pack before dV-reuse wait | 581.504 / 579.008 / 582.912 |
| R22 | generated CuTe P0 | 612.800 / 607.616 / 616.384 |
| R22 | v469 later-dK-gated reuse | 574.880 / 571.168 / 576.352 |
| R22 | v471 pack before dV-reuse wait | 583.072 / 580.864 / 585.696 |

v472 instead changes only the three query-head boundaries in each owner-4
CTA.  After the old score page is consumed and the next Q stage is ready, the
tensor issuer submits score(next head) before waiting for reducers to drain
old dQ.  Score occupies TMEM `[384,512)`, disjoint from old dQ.  The retained
`dq_drained` acquire remains immediately before dP(next head), because dP and
dQ alias TMEM `[0,128)`.  A selected-SASS audit verified the fast and retry
paths as score wait, query wait/acquire, score MMA/commit, dQ-drain
wait/acquire, then dP MMA/commit.  The result remains `REG128`, `STACK0`,
`LOCAL0`, zero-spill, and 167,168 bytes shared.  It has 3,368 instructions,
28 tensor-MMA sites, seven completion commits, and 13 TMEM loads; the eight
instructions over v469 are control/address materialization rather than math.

The full B2/S4096 gate passed the documented S128 diagnostic envelope,
finite/nontrivial S4096 gradients, and exact zero gradients for zero dO.  dK
and dV are unchanged from v469; dQ varies only at the existing additive-TMA
reduction nondeterminism scale.  Two matched rotations confirm the modest
head-boundary win:

| Rotation | Boundary | Median / p25 / p75 (us) | CuTe / candidate |
|---|---|---:|---:|
| R23 | generated CuTe P0 | 613.120 / 608.192 / 617.920 | 1.0000x |
| R23 | v469 later-dK-gated reuse | 573.856 / 571.232 / 576.224 | 1.0684x |
| R23 | v472 score before dQ drain | 572.256 / 569.504 / 574.624 | 1.0714x |
| R24 | generated CuTe P0 | 624.608 / 618.752 / 630.144 | 1.0000x |
| R24 | v469 later-dK-gated reuse | 573.536 / 570.816 / 575.232 | 1.0890x |
| R24 | v472 score before dQ drain | 569.760 / 567.328 / 573.344 | 1.0963x |

v472 improves v469 by 0.28% and 0.66% in R23/R24.  v473 is the selected thin
dispatcher: B1/S4096 embeds unchanged v445, B2/S4096 embeds v472, and
nonexact B1/B2 retain v436/v437.  The embedded v445 and v472 selected SASS
streams are byte-identical to their standalone streams at 3,440 and 3,368
instructions.  Both hash-bound full gates passed; the reset-inclusive gate
medians were 316.256 us at B1 and 572.128 us at B2.  These are isolated
kernel results.  Full-model integration and end-to-end training speedup remain
unmeasured.

For v465--v473, a "selected SASS stream" hash is SHA-256 over the raw
`cuobjdump --dump-sass` text after the selected `Function :` line through the
line immediately before the next `Function :` header, including its trailing
newline.  The standalone/embedded identity checks use that same extraction.

## Historical BF16 donor boundary

A fresh history audit did not find a hidden exact-BF16 D128/GQA TK kernel
that beat the CuTe control.  The remembered exact-BF16 TK win is real but is
a different shape: commit `780af757` used B1/S4096/H8, Dqk192/Dv128 MHA and
measured 355.935 versus 389.312 us and 359.520 versus 399.424 us in two
strict rotations.  Its broad W19/L16/U0 schedule supplied the batched dQ
drain idea below, but cannot be reported as D128/GQA evidence.  The historical
D64 GQA result was an E4M3-input/BF16-output low-precision kernel, not exact
BF16; at B16/S4096 it measured 3.729 versus 3.749 ms against its matched
native-EX2 CuTe control, while a different CuTe period-2 control measured
3.356 ms.  These distinctions resolve the apparent conflict with the present
D128 results.

## Split dQ lifetime and selected route (v475--v483)

The remaining D128 B2 bubble was the aliased dQ/dP TMEM page.  v475 tried a
broader cross-head score/dP pipeline and regressed v472 in both rotations.
v476 split the TMEM-capture and shared-publication barriers but left a
collective branch in the generated schedule and regressed by 1.66%.  Both are
rejected; their source is not retained.

v477 peels the final D32 load from the serial publication loop.  Reducers
announce `dq_tmem_drained` after that load completes, allowing dP to reuse the
TMEM page while the last BF16 conversion/store finishes.  v478 captures D2
and D3 with paired x32 loads and one wait before the same release.  v480 then
keeps D1 as sixteen exact packed BF16 register words and defers its shared
stores until after release.  Finally, v482 also retains D0 as sixteen packed
words: its complete per-lane source payload is 32 packed plus 64 raw words,
and it releases TMEM after the paired D2/D3 load but before every D0--D3
shared store.  The later `dq_drained` barrier still protects shared
publication.  Every retained candidate uses the same elementwise
`1/256`, RN-BF16 conversion and addresses as v472.

| Rotation | Control/candidate medians (us) | Matched result |
|---|---|---:|
| R25 | v472 572.320; v475 574.592 | v475 0.40% slower |
| R26 | v472 572.512; v475 575.520 | v475 0.53% slower |
| R27 | v472 572.192; v476 581.664 | v476 1.66% slower |
| R28 | v472 571.936; v477 568.896 | v477 0.53% faster |
| R29 | v472 571.680; v477 569.792 | v477 0.33% faster |
| R30 | v477 570.976; v478 569.984 | v478 0.17% faster |
| R31 | v477 568.992; v478 566.848 | v478 0.38% faster |
| R32 | v478 568.960; v480 539.840 | v480 5.39% faster |
| R33 | v478 568.864; v480 539.808 | v480 5.38% faster |
| R34 | CuTe 620.032; v480 540.224; v482 514.272 | v482 1.2057x CuTe |
| R35 | CuTe 639.328; v480 540.064; v482 515.040 | v482 1.2413x CuTe |

The v482 aggregate quality against exact BF16 on represented E4M3 inputs is
unchanged from its parents: R34 cosine/relative-L2/norm are
`0.9996628565/0.025965816/0.999438741`.  Its authenticated full gate also
passed exact-zero dO, the S128 reference envelope, and finite/nontrivial
B2/S4096 gradients.  The selected kernel is `REG128`, `STACK0`, `LOCAL0`,
zero-spill, with 167,184 bytes total shared allocation and 3,504 decoded
instructions.  A v481 attempt to force later conversion with an opaque scale
was constant-folded by ptxas, added eight instructions, and did not move the
conversion boundary; it was rejected statically and its source is not
retained.

v483 is the new thin selected dispatcher.  It preserves the byte-identical
v445 B1/S4096 kernel, embeds byte-identical v482 only for B2/S4096, and keeps
v436/v437 for nonexact B1/B2.  Both authenticated full gates passed; their
reset-inclusive medians were 314.912 us at B1 and 517.824 us at B2.  Those
gate medians are integrity checks, not a replacement for the matched R34/R35
matrix.  The result is still isolated backward-kernel evidence; full-model
integration and end-to-end training speedup remain unmeasured.

## Authenticated evidence

- Correct ABI matrix:
  `/tmp/d128_native_compare_output_20260829/native_vs_cute_correct_stats.json`
  (53,917 bytes,
  `9d77dc85ac6b5cd520103320a30df114cd088cc2d09052753537d670d7de4a0c`)
- Corrected v424/v428 matrix:
  `/tmp/d128_native_compare_output_20260829/v424_vs_v428_corrected.json`
  (27,122 bytes,
  `0c4d9fc70d868bc29142a93f1bcf32b5dd353dfba04ffc4bb555ef1a1b059777`)
- Final v424/v429 matrix:
  `/tmp/d128_native_compare_output_20260829/v424_vs_v429_final.json`
  (27,172 bytes,
  `aa93391f24376d1d266c8ec47548f9989037975566533c8f0bdf31b50ec234d3`)
- Matrix harness: `/tmp/d128_native_compare_20260829.py`
  (`f4c7af316b900f8ea08dbadda1a5c4a106000b0451fb186435f9e5252e0f6f9f`)
- Original multi-candidate specification:
  `/tmp/d128_native_specs_20260829.json`
  (`0c78701eba340d156d9393d7af9b977b2f9f169e5f56039426cdffed9c996a63`)
- Corrected v428 binary:
  `0048cb29609683436076576da11b9d883ca92260345402303d9fd58723e6a6fa`
- Corrected v428 header:
  `27654850adbb7e0fadd875a011413fb9db20e3fb4e5eadfd092d133c5c5823c3`
- Final v429 binary:
  `61fa08ba36153274bf92ca2c03950267bcaef064a3dd406425a77be6015ce628`
- Final v429 header:
  `db6cd6d27f46c2b2cb96eb971e0b8570911a327550fc8825948dbe52f08c82dc`
- Final v429 selected-SASS stream:
  `0e544221eacaa22fb1b9049cb2cfc207f1567d0ab9db88707d355e08f9a74092`
- Final v424/v429/v431 matrix:
  `/tmp/d128_native_compare_output_20260829/v424_v429_v431_final.json`
  (36,972 bytes,
  `45b6bef0e4a331c210ace108d1a545cbe09c72d6254953f4aeb0ab91f14b36db`)
- Final v431 binary:
  `0c91456c8e986640940388df1791d9c2371273e5d5ae2d56e25e25395bb4878a`
- Final v431 source:
  `32b2639c3e7e95a9bec15ec32d076501a97a29296673721d1d36fc6673ad4f96`
- Final v431 selected-SASS stream:
  `228e41fcec5b95ff97024fc53313310f384758fba5bb6211f8d015ab06b92045`
- Final v431/v432 matrix:
  `/tmp/d128_native_compare_output_20260829/v431_v432_final.json`
  (28,041 bytes,
  `80b926f28a78aa5e605c06eaf2ab0d370f6fd99cbd141ef706fccd1b6e8cf3ff`)
- Final v432 binary:
  `28bac3b7388ebddf3771100b19947abc4486fd1982cd9646a87484ee275d460a`
- Final v432 source/header:
  `4542cd312b07476951f741214f319c825585433364c3f8201d4a453994ccb493` /
  `2baf023ca28455f60add1ffb156f8f2abad383602e05e52509c3726a39b5107a`
- Final v432 selected-SASS stream:
  `5bca5450f0cfa531304c4d41533b15f4152266f07d2bbef4b7622791558cb33d`
- Final v433 R1 matrix:
  `/tmp/d128_native_compare_output_20260829/v432_v433_head_fast_final.json`
  (`8bbe19c3c82eeb2da18763878c72cebb8a9e22cdcd0effd6f4fd07afd78a3f31`)
- Final v433 R2 matrix:
  `/tmp/d128_native_compare_output_20260829/v432_v433_head_fast_r2.json`
  (`cb6aa02beb135cf4edf0bee6ae00756a57510397b8a47757066ecb1badf8bac0`)
- Final v433 binary/source/header:
  `3906438d9d8b6bb094833a835ff05e7d0f25167504f5f6dfc36f02eb04ac12d6` /
  `e7a6522478c1843764296e3ca2947f0caddeb00591a862db2956263cd0cf86a4` /
  `2a932ce1a0a44373128ab8ab371e953c7e4da1fc12019578ac180b8b9901670d`
- Final v433 selected-SASS stream:
  `89aa518a109aadc56d85bf64652c4a378d135b35e4ec629efa172c5e639fbf3e`
- Final v434 matrix:
  `/tmp/d128_native_compare_output_20260829/v433_v434_cluster_final.json`
  (`82490c9e4727b84a189402567595868838051eb1bd3372f31401a15680271f80`)
- Final v434 binary/source/header:
  `5520be3e5a04f686f6b36248f849965f30c443921a8490f2b9b8599728fe25fd` /
  `da81dcf735091126d7ad1c044133cf51e5d242bb681ee07e6c5e0dba57945a2f` /
  `366fd6a0216d2f9a5d33b8b08eccc78e9c07b1653fd796e02a38a2080f634ba5`
- Final v435 matrix:
  `/tmp/d128_native_compare_output_20260829/v433_v435_batch_head_final.json`
  (`a6c280f8fcfa332fcab81875794f0f62f3500f6a566d35b80711f50a4c689d01`)
- Final v435 binary/source/header:
  `2836596bb8b5f6daf4498a6cab81f55505ac93c29658d1e10dc9277dfa087705` /
  `68a74a204814d9d82c8261dbf4e2fdb3e032925bf1a80bbff66074740a22804d` /
  `b50db2c6d034e04783eb6eb23801b70f607e5708c97490bc380553efd9106e57`
- Final v436 matrix:
  `/tmp/d128_native_compare_output_20260829/v433_v436_exact_runtime_final.json`
  (`d653bbf4d38d1e852dc4712a8b8ddbaac5e6ccb3133b81599f3ad28a69e0a9e6`)
- Final v436 binary/source/header:
  `c1735445444d2802c558a4433425170402a588acbc5d0b3a7403a22601208227` /
  `f659597b503226bbdbd09f5897130d58a918907319abae478b668033e31fe973` /
  `1ba7739da73592cc1af106bd2fe020d1cf43c2e1ef379a841c1e8dbde138602d`
- Final v436 selected-SASS stream:
  `a5bed8316ee1fdaabff39096ad79be9cd30ce791466e3b93d2bf2e7f8628b0f3`
- Final v433/v436/v437 matrix:
  `/tmp/d128_native_compare_output_20260829/v433_v436_v437_owner2_final.json`
  (`ce2d5c29388e6c2af5ad7ea16b59ab4a89a0a18b31c2333b0af8225d9f7da616`)
- Final v437 binary/source/header:
  `d7044bf00e05ce66b67c8dd2d7f67dc5151f5f539f75d7ec05970a01b9b4369f` /
  `574d64344f98fd83bcecdcf153818c91613d4941d15db44b824cc6565e95cffb` /
  `8688f89753b8868fab11b102cc1ba85f2ed0952309adc214ce116dba33a926b0d`
- Final v438 composition matrix:
  `/tmp/d128_native_compare_output_20260829/v436_v437_v438_composed_final.json`
  (35,321 bytes,
  `8144af3dd5af76770af2363d4eb16a819a83cab458b142d22d8dfad0277c2ccc`)
- Final v438 binary/source/header and selected SASS:
  `adbe08c8b2465c5e51d5fe22d57288b2fd89f8e3729031cd85e60a81f263ff68` /
  `03e5f2ca9463480b015b4f2bf8632e176a8a72b93d17a613fb313e95fd7760bc` /
  `9c5ced5f34e83fc0ca55eeae7d57fc597100ef54f058c615c0e24de9646e826c` /
  `8b192ab634a93f7647c0aa9610d52ca85e3df819b7d55309cca93d46b903a3c`
- Final v438/v440 owner-count matrix:
  `/tmp/d128_native_compare_output_20260829/v438_v440_owner_compare.json`
  (25,525 bytes,
  `678f1c81a51aebb5df04214c3e9d9520b87a706e4e5f22026fbe41e1a3d6139a`)
- Final v439/v441 R1 matrix:
  `/tmp/d128_native_compare_output_20260829/v439_v441_final.json`
  (25,757 bytes,
  `fbb88d6532fe24847815e63c5f9c808b8421c60ea0519219a4d637a7b21aa21d`)
- Final v439/v441/v442 R2 matrix:
  `/tmp/d128_native_compare_output_20260829/v439_v441_v442_r2.json`
  (34,471 bytes,
  `e0b157c61a465dfac6c9ea1be0238074a26cbf67e252d688400462b5bec065a3`)
- Full v439/v441/v442 validator receipts:
  `/tmp/d128_native_compare_output_20260829/v439_full_b1_s4096.json`
  (`426f14d0fb957476e18aa4872e46b22fb162abadbb58e6b1fe20938bdaeac2a7`),
  `/tmp/d128_native_compare_output_20260829/v441_full_b2_s4096.json`
  (`f0fc90945a5f17a7edf10e39091cefcab53ec13c21b9ca841e70631c2e22747f`),
  and `/tmp/d128_native_compare_output_20260829/v442_full_b1_s4096.json`
  (`c7ae6a4c3911be4089cfbbcfbb29808db06f6636eea1cf0e24ed163d76500bb4`).
- Final v439 binary/source/header and selected SASS:
  `d9950fc0168965a3df7f3c9b98e62e77b43f7fd0a5b0cad40ed1c81d2987e8d0` /
  `3d4c994342947d56708844a06b3adf6a8db8806bef86e1fc6f3b339b185ab33b` /
  `1995c1feaa6455c8725d77ebb5dbca20390523ce8f5df68addbfbfe0af2be46a` /
  `e93073bf9c7132074e648ab0a27e3aa6e864c724684124d74f1f9548442b05f9`
- Final v440 binary/source/header and selected SASS:
  `50a5ab04309726cc2923256476c3103d66d15663e61611f42a8ca68656073ed9` /
  `e4c3f2d3c203573f63ebd4aaeecb89794257b9990cf74bc7993ff494091ebb3b` /
  `6458ed6b9ee0be2228b4b5b534aeddec876c8dfc46907b273bcffbff1f408ba6` /
  `0b28bb5ea23de9d7ab30bd7e373e9d2c747b3da011ff518c7d85815f754956b4`
- Final v441 binary/source/header and selected SASS:
  `184e1bbeeb3e55b2a450b878bbc18c2b45e62edd6f0829512147c95718b47749` /
  `ede39ef1e10d446e99c982bec43741dd3c2af6848355cc1956ed365c6f8032e5` /
  `e1085a3f358ee4a81e24b543e317fa68ec3d202773dbcef8d4abcaa58f75a684` /
  `cdbfe36b418ca7bb984b9a99cefc90fd2c971e674553db446adaa6af21588f29`
- Final v442 binary/source/header and selected SASS:
  `17b6446eb1c1b61718bea8b84fa26dca615812e0c750dfd903e188afa9a83adb` /
  `4895c642949f2c023e5ebfe190acf0b95e124563f05f7cf3d688c123fb7bf902` /
  `8090a2cd01d576cea615083d9a6dfdaf9762f22c9fcf25fae5f390d10741db12` /
  `993e78c7edfd40385decbf8ad60169b7a575ec058e2753de9e33d1503a82384b`
- v443 compact-P R1/R2 matrices:
  `/tmp/d128_native_compare_output_20260829/v439_v441_v443_r1.json`
  (34,829 bytes,
  `59e0e4600553719ebe11971791b9a11d5dbd6fdcd077e01bddde741bb3672635`)
  and `/tmp/d128_native_compare_output_20260829/v439_v441_v443_r2.json`
  (34,827 bytes,
  `c3c562e444231adc8a509e398e3cd526c546b11a8922a3f2ecfb3e93ad5c239d`).
- v443 full validator receipt:
  `/tmp/d128_native_compare_output_20260829/v443_full_b2_s4096.json`
  (6,396 bytes,
  `e31c4355894ab93a6b9c157afa51d8ce091b7211897529f7371175aa37976315`).
- Final v443 binary/source/header and selected SASS:
  `f5f8f82212eb64aac50121f64b885bfbd940f19379d244684a8f540787d3bf5d` /
  `162d06e8ad3272f09299ffe36d402ff61217a4be05b1aef5d981fbd10f3eba24` /
  `1936a90a5c4084967beda8454b3f5928d50abd277479f14bbf4546dcdd211095` /
  `f65d9d6f7c0cd542c822651142619992b9032032df16b89e80d8b7342c5210fc`.
- Final v445 matched matrix and validator receipt:
  `/tmp/d128_native_compare_output_20260829/v439_v445_compact_b1_final_r2.json`
  (25,891 bytes,
  `ec8a2d0f656d2560ba2b0cb7450ae99e92cc14a5f9b675f97e837f17059b7f3c`)
  and `/tmp/d128_native_compare_output_20260829/v445_final_full_b1_s4096.json`
  (6,915 bytes,
  `bc7ec69eca43a1a8a2434b3d3683c6294a2f5c197c21e1428d205a3c8a410264`).
- Final v445 binary/source/header and selected SASS:
  `8a6c2279d18198fde33ce01a52bdc0ddf166ef367462f9a9ce2d38f8c6411e5b` /
  `253e5cacd7035b74ed14e990e6349972b95e928fff91ef2616059ffd81512e4b` /
  `2dc3f078fd722d19353ba3675a2773943fb7e13a6c3d70ee8f0950a4160d02cc` /
  `3014c168b8d5986ba565c2e1ec075dfa1cea10c161bae1a9885f1dcc212acdcd`.
- v446/v447 matched B2 matrix:
  `/tmp/d128_native_compare_output_20260829/v443_v446_v447_b2_r3.json`
  (24,024 bytes,
  `cae89f334955d827adc634ff8e0b78c8fd465019709b526e12f72c997cf3d1de`).
- v446/v447 full validator receipts:
  `/tmp/d128_native_compare_output_20260829/v446_full_b2_s4096.json`
  (`853169611488c165bcfc7d7fa21bbaefc410f26f586bbde1a62e2023e43459b6`)
  and `/tmp/d128_native_compare_output_20260829/v447_full_b2_s4096.json`
  (`813576c96a1268e129123e271071848bb1d21814fd6fb7eaf33ea9c03ba02e26`).
- Final v447 binary/source/header and selected SASS:
  `acf3dcfa9bca366aee793f81417ffa1f360d83a1c61d091924b7918cf8487a53` /
  `bfab3d7aa30a6573699067015d936a67198b6f963d99345f4ec720729ff530ea` /
  `ede80c1045de4d10bd1d8a6a7ac208f9a81ccf13619bd534ba980df97aa30fdd` /
  `b97a3df02d585dec7b5122ea97472fb3d3e2cfdd953226b23f00e63365df7a87`.
- Final owner/schedule matrix:
  `/tmp/d128_native_compare_output_20260829/compact_owner_schedule_r4.json`
  (62,631 bytes,
  `6b8c4743bde81e817cdc45583425449efd63005fa39708f38c0150b795874e0d`).
- v451/v452/v453 full validator receipts:
  `/tmp/d128_native_compare_output_20260829/v451_full_b1_s4096.json`
  (`2b623f6a4c88bd5c76561f5a2a64fd5740f1b94aa702184455e54620449d899b`),
  `/tmp/d128_native_compare_output_20260829/v452_full_b2_s4096.json`
  (`ef3841256b7a5734e9b39abe37e0297ba13a3386499a64a90b4b97eb0168501c`),
  and `/tmp/d128_native_compare_output_20260829/v453_full_b1_s4096.json`
  (`eba6fa443b96da0ed8c64ff85710d41882f14dfd95f543d197cabc9f7d6dea61`).
- B2-only matrix harness:
  `/tmp/d128_native_compare_b2_20260829.py`
  (7,063 bytes,
  `870c5dc4e98e1543447d2570a5edab470587f8d311a0a87a3ec9d3e07fe1ba57`).
- Final v454 matched CuTe rotation:
  `/tmp/d128_native_compare_output_20260829/v454_vs_cute_final_r5.json`
  (17,774 bytes,
  `4337a10a449a1e79d2bcc63a7fd5773cdb79b084a449d3b3c4739e79329e5248`).
- Final v454 B1/B2 full validator receipts:
  `/tmp/d128_native_compare_output_20260829/v454_full_b1_s4096.json`
  (6,802 bytes,
  `b2ac8280332f86a2239399b1a1a575e83ab1d140f7606a90fd17a9c83a31baaf`)
  and `/tmp/d128_native_compare_output_20260829/v454_full_b2_s4096.json`
  (6,793 bytes,
  `71e500adab99dc08be7e4f20e72fde5f0c7e6bf5f3bc0d7b82da644150ca5c08`).
- Final v454 binary/source/header/Makefile:
  `5a0b6abb021f4dfc8a28c5a23a40ad7d46c45c9506a1febee11e7f840f599dc7` /
  `f01a1e294c79efa0107879b9c80a5a8467d8bf1494395d38e8e85d3f00b73833` /
  `c53d6b9b45a36f6efd568a0e9df7a63c5deef7d308b8e588be9d22fbb2b1365e` /
  `e89ec0f447314b3c557eea6612d02ce699bf27d4b5b39dcb50d77518ebd3a020`.
- v455 full B2 validator receipt and matched v447/v455 R6 matrix:
  `/tmp/d128_native_compare_output_20260829/v455_full_b2_s4096.json`
  (6,987 bytes,
  `58e2f2ae03ba7cfac3a441495fc955feb294dd1bf57d709b9610afa46d57d289`)
  and
  `/tmp/d128_native_compare_output_20260829/v447_v455_split_dv_b2_r6.json`
  (18,084 bytes,
  `d357ba743726d26c31506b155a89d72447a08dd1555a761628e4998191526a40`).
- Final v455 binary/source/header/Makefile and selected SASS:
  `89c18aebde362c1c4a176c1091f79be8f6721688c50ba1cfa08d0d05730746aa` /
  `1868a11a624e10d0cffcf63278528f070e1c2a99595e86aac8550265666d7d4b` /
  `84958345feb36ab0b558c5f71ca18cce6729c3e62920ab9ad121308387a3fda4` /
  `9b4b7c1691e7497ec9c811b7bf44914412c9393754e28a2ca1932a0913faeae4` /
  `a5a4ab95ec77c507bb9be25fe508dd3ec4bfc733f8a6ced00e116cfbf50a0547`.
- Rejected v456/v457 full validator receipts:
  `/tmp/d128_native_compare_output_20260829/v456_full_b2_s4096.json`
  (6,883 bytes,
  `817c26c651e6a45dd8a47e0d0038142f6210e2d7af880277b95ee80981f2b963`)
  and
  `/tmp/d128_native_compare_output_20260829/v457_full_b2_s4096.json`
  (7,323 bytes,
  `52f4cee2784a85f80b747112b9b585e4374b0e103c3f74f8d9b02cc042a40421`).
- Final v456/v457 binaries:
  `a0e3306f310540d5dc612a268e87fdda221f2db319d7ef143817cb7618954eee` /
  `ca4e784e966eacd93728f8e3d93898b118b3350635e458c34aa6eb3923960eb3`.
- v458 full B1/B2 validator receipts:
  `/tmp/d128_native_compare_output_20260829/v458_full_b1_s4096.json`
  (7,217 bytes,
  `8d9c4e2e8a7bb8c5f55595ba087ca80c384ff65174e32c5aecb8026412fb3ad0`)
  and
  `/tmp/d128_native_compare_output_20260829/v458_full_b2_s4096.json`
  (7,201 bytes,
  `6d8a79ee878ca499f94eb48630e3bfa94df3894a3ec02812cfc4fba85986eeb2`).
- Final v458 binary/source/header/Makefile:
  `e8e6794d833e1281bd8c67c9508b5deb4299a5cf655539954f479d476469fc36` /
  `047e2346d5c316f433ef095128e0f23da6a0d95b98d0bc48bf4d7f73ba972c5f` /
  `6d9887fa3ade57387ca66b0deec879d25de4e2aa542183100c3047b217e93a6f` /
  `a7b8720d73dd9c4d72c595a46ed9dd68c1f3ccf0327a816dc39e091d08776811`.
- v460 full B2 validator receipt:
  `/tmp/d128_native_compare_output_20260829/v460_full_b2_s4096.json`
  (7,316 bytes,
  `6feaad17b2f3e68e1f55d080f05f844db68b4d68cf72136755d802f4c3c43689`).
- v455/v460 matched R7 and R8 matrices:
  `/tmp/d128_native_compare_output_20260829/v455_v460_quarter_dp_b2_r7.json`
  (18,794 bytes,
  `e6c621004c4a07605e9e2207ef63eb0f6a220feb151c74d867e4dd6f07bcc79e`)
  and
  `/tmp/d128_native_compare_output_20260829/v455_v460_quarter_dp_b2_r8.json`
  (18,776 bytes,
  `35cba3391707d83539cb32b4143ad25f2d9cfe64ed46bc7323138086322df127`).
- Final v460 binary/source/header/Makefile and selected SASS:
  `d180e470b315cbf5d7a20d3403d69bbf11b6d500112fc4ed72bed55421c92ea7` /
  `1ca679b0c9788dbf8087e0683bdd60f737e4ea246e3104899af9b2fce1d3f95d` /
  `528c131b9540b4408cf08f5979b6ceb937ab9335b444995ec79efb5fb693cce0` /
  `e7a582200c1cf89e9baef809762a6837e0677979480eccdd2b13a92177adc57a` /
  `59fa570840be6b8cbb648dbc4c35829b6bf190eea1f6f81efbdd15cfff6eefa5`.
- v463 full B1/B2 validator receipts:
  `/tmp/d128_native_compare_output_20260829/v463_full_b1_s4096.json`
  (7,539 bytes,
  `6d97e5b43da97b2fb8b0a5fb20c15546ef6bd5d1c45c394131140fc2c300d3c0`)
  and
  `/tmp/d128_native_compare_output_20260829/v463_full_b2_s4096.json`
  (7,528 bytes,
  `9f1e8d2d08470d4911e35ee321d3029525fb368a14138ebe56f9923b4b0734dc`).
- Final v463 binary/source/header/Makefile:
  `e15953fd1fc3b8f658ecbac2b76392b75b89dbf12bd14fe055eae471d6c5ac03` /
  `03a801748b396e0e8e1dd92db06b9aaaa783b4ee79d77a69139f7f7876dd4f89` /
  `343d7fdb39c83afe634fac6ae7c782a19a1e83751330c47694cbad63ca205848` /
  `aabbfa50edc5881e4897487134db4213ccf1e4b2b6ab16cab3884ad7690da09a`.
- Rejected v464 full validator receipt and matched R9/R10 matrices:
  `/tmp/d128_native_compare_output_20260829/v464_full_b2_s4096.json`
  (7,509 bytes,
  `6b4a75d1004a326c706b5446a7ef2c2ec1f12ac29cc2799b7cbb9a6978c3793c`),
  `/tmp/d128_native_compare_output_20260829/v460_v464_all_quarter_b2_r9.json`
  (19,348 bytes,
  `a92f50ec6acdb93fd946c0ef99067a36ece5598b97087456df3ed547b45824be`),
  and
  `/tmp/d128_native_compare_output_20260829/v460_v464_all_quarter_b2_r10.json`
  (19,354 bytes,
  `4dfaee933bad90f95b51908a12b904596ba654555640d2d5343aaaba06f708ab`).
- Final v464 binary/source/header/Makefile and selected SASS:
  `ad8019536c3ace04332de087c0a911f4229c45ebb444d976e32032b30bb19a4e` /
  `ec476abb05d5988962d99418b834c60651a841b386ba9efa0d8056966e2a3690` /
  `7ace2bd29c6ce6e827676ab22b78e188dc4d9809fff994f52d4d9341ec185707` /
  `52f6c3f1e4360e69513ce645ab17c7f7ede5fd041058cd53400714407fce8fdc` /
  `bb1b2146a121783e8feb2b62b9282a59c7ec229928799b1145c7fe55ae13ede0`.
- v465 full B2 validator receipt:
  `/tmp/d128_native_compare_output_20260829/v465_full_b2_s4096.json`
  (7,476 bytes,
  `dacc400552bac8f7de1d2d1361fbb9be19b50061ac2f46fb873cd5353a6ed1f1`).
- v460/v465 matched R11 and R12 matrices:
  `/tmp/d128_native_compare_output_20260829/v460_v465_probability_barrier_b2_r11.json`
  (19,377 bytes,
  `578ea4a156d84fa66389fec3996cc64f70b04143a40e47b55c9c1544f9ce939b`)
  and
  `/tmp/d128_native_compare_output_20260829/v460_v465_probability_barrier_b2_r12.json`
  (19,383 bytes,
  `a677d4f005e041ca994ca7eaf4fbbea9723bb3df53558c0a5abec0fe87e37d79`).
- Final v465 binary/source/header/Makefile and selected SASS stream:
  `b4013d021fc3b8c0af994e4fe60e38c1d768e8299b290adb1d6bb52b66ab2930` /
  `99e113ad4fa89aff625155534b0b5d66349569de453fa0ed68d8de85d7ee1ede` /
  `7b8456694712fe2213dd62da22dd459a302a72d20115c1a70867ebdd2ab7fa37` /
  `934e1f6fc83fa671af1cb7ab8beee888fb29800d073b09aff1f572bb45af2603` /
  `4508669dfd97d7575f652fa673800eb386223a4bafc74cb5d7269c2a06ee63ea`.
- v466 full B2 validator receipt:
  `/tmp/d128_native_compare_output_20260829/v466_full_b2_s4096.json`
  (7,698 bytes,
  `fecd1a2ab41cf7a0e092933973687ff1b891b714097017515eb4b231aabef9e3`).
- v465/v466 matched R13 and R14 matrices:
  `/tmp/d128_native_compare_output_20260829/v465_v466_stats_barrier_b2_r13.json`
  (19,806 bytes,
  `d0542cd5f435c5e098c0f6b90959446539428b3747cfb77eb4dcd5bc8c4e4a34`)
  and
  `/tmp/d128_native_compare_output_20260829/v465_v466_stats_barrier_b2_r14.json`
  (19,785 bytes,
  `5c16f9f89f4ad27daed799b96abb98d3cfd66f485ed896e9eeb91a55a2823c37`).
- Final v466 binary/source/header/Makefile and selected SASS stream:
  `79ac19cdceda54bd505b1b8b0dc7a6b7cb075823df48108ac34e282b0bbde7a3` /
  `1f93fe63e0208a2b98472275f4c00e0f2cd6c2021608054efc53e706cc89ed23` /
  `0cc23c9018fd94ab3ce109492ca7f9542f882ea6a9b81648fbae97ec117afd6c` /
  `c9c4ac0ced61204c518a3a3ff2a6129ce84d9836b32e78be770ed8a5d80875da` /
  `ed023826a26016a6d2d80f0db7bf8585198081bf9436979101d2f6c5e3d9600b`.
- v467 full B1/B2 validator receipts:
  `/tmp/d128_native_compare_output_20260829/v467_full_b1_s4096.json`
  (8,057 bytes,
  `93aef464562a4b90f1dc566d7c546f181c1e01e4ecd805d1dd8b3658934a48fb`)
  and
  `/tmp/d128_native_compare_output_20260829/v467_full_b2_s4096.json`
  (8,039 bytes,
  `84d4ee3ad196be5ccb41afcc64401f7c98896b6cf4a9f924e6bba54898f43620`).
- Final v467 binary/source/header/Makefile:
  `49d7c721216b34f73d79af1681e1ff0e75276e9ce6251c5ea8b5502e2e482018` /
  `460891a4117ddf33b2d25e3aab733eb2b42e89370b12c2a15fe7dc7ef9c1463c` /
  `f8a942b057084d3ae03abde609649350243a7903a766251d9f1af345d359e5a9` /
  `60fbe20481a47171b734c66c21e1a618176ae303b2207c75bbd09f219fce9b77`.
- v467 embedded v445/v466 selected SASS streams:
  `1734b07c039af28b38c717ef283719c5fef5bcf01b9d7ac0e770df61701f6cf7` /
  `ed023826a26016a6d2d80f0db7bf8585198081bf9436979101d2f6c5e3d9600b`;
  both are byte-identical to their frozen standalone streams.
- v468 full B2 validator receipt:
  `/tmp/d128_native_compare_output_20260829/v468_full_b2_s4096.json`
  (7,996 bytes,
  `c2e7787a11a5dd4e0fbd5e0a78bc222a4620b32e4451a0bd3d2f6f6a6934af7c`).
- Final v468 binary/source/header/Makefile and selected SASS stream:
  `f91f2d9c9d0ada529b0e92a1f71ebed5e3485897dbc12a7db0e1d6353a8be587` /
  `c047233e5a1830346445998dedf5affb195d607bb90830a5bcb7ccc2bb716b1d` /
  `316d29de2c589325f685298984fbb6cb6fcd41e39e736f1861f9f45ac6a45a21` /
  `ed0786a9a25422b711334ee84ad8e6cf3ac7f28397ac7ae13e0bfb6cf2e4ff6f` /
  `66376bfd3460c4e946b620c8b1755f8f545991d09b21c9fc4e7f54cd490f2399`.
- v469 final full B2 validator receipt:
  `/tmp/d128_native_compare_output_20260829/v469_final_full_b2_s4096.json`
  (8,288 bytes,
  `b8940fe0eb5b0d13eda6eeb85789b62a629eb44ad014b169a9f7d49d1b00c69e`).
- v466/v469 final matched R19 and R20 matrices:
  `/tmp/d128_native_compare_output_20260829/v466_v469_dominated_waits_final_b2_r19.json`
  (20,672 bytes,
  `c1542ef34b53a62231a8b052b0f5696fb8707e5d5f9da70807ae97a26d4ba698`)
  and
  `/tmp/d128_native_compare_output_20260829/v466_v469_dominated_waits_final_b2_r20.json`
  (20,673 bytes,
  `45aeab9463c9055aee8d420265f3d65cc3d3612ad4904e7426fe6fc6ec088a36`).
- Final v469 binary/source/header/Makefile and selected SASS stream:
  `69cd528ac415343429fe252904736ebb1d2179d0e8c58a87c784f9d8b03059a4` /
  `3dd4610e2b7da4318f8fd4a3d11c676aeaa72287b16850771fb726aa1c8277ba` /
  `aa1ecb0ffee344fef245a8a368d67a2d32f3b59dc70b4f1ba9b6edde1c9c1f6b` /
  `c5822426659d249e045d0e28ffce885f7574a0bfa2745686dc8af5160369a48a` /
  `98b2d6bc784af6dbaab968d5260de0546e9b79df5311602cc3f195d91a40d0de`.
- v470 full B1/B2 validator receipts:
  `/tmp/d128_native_compare_output_20260829/v470_full_b1_s4096.json`
  (8,800 bytes,
  `62a6eddbf2699cd5e45539e4caa23686f653a8fc46a408b2aded6345b0fbbb38`)
  and
  `/tmp/d128_native_compare_output_20260829/v470_full_b2_s4096.json`
  (8,788 bytes,
  `1e43ed3bc68b32e7c33a960da02139c8d203a1c0d4e4f973aa93d0babfe3cf34`).
- Final v470 binary/source/header/Makefile:
  `4f4eeceb0975d2be7b5175f19d73778a6f15262fc617c9adbafac803e0271ef8` /
  `f6907627bee167affa0baa08b70a92d4a140735ec7def362deaadf68673cb579` /
  `84f54a75280056a5a7a557ad1180ab7c3c7ca46fe97cb2f20ec676c8790e4c82` /
  `a891da9d5d6171537591d410aa145992e1839b87e44ef47b53b17805add6e1fc`.
- v470 embedded v445/v469 selected SASS streams:
  `1734b07c039af28b38c717ef283719c5fef5bcf01b9d7ac0e770df61701f6cf7` /
  `98b2d6bc784af6dbaab968d5260de0546e9b79df5311602cc3f195d91a40d0de`;
  both are byte-identical to their standalone streams.
- Rejected v469/v471 matched R21 and R22 matrices:
  `/tmp/d128_native_compare_output_20260829/v469_v471_p0_pack_before_dv_wait_b2_r21.json`
  (21,806 bytes,
  `cf4e5ce25e3958b8f2f76d4517fddb4a9482c5cc7d5dea142b8962203b16a7e7`)
  and
  `/tmp/d128_native_compare_output_20260829/v469_v471_p0_pack_before_dv_wait_b2_r22.json`
  (21,811 bytes,
  `0cac0135e74edc9477e829924f5e173017fdb125ca3bb67c76b6c5ace9728265`).
- Rejected v471 binary/source/header/Makefile and selected SASS stream:
  `50f1f82b1082f051b882f20304e251f7ceb3e046f068605377f67202e2905dde` /
  `bd598ee956afe49f4fd165fdd8ae565823d705560817bc096c57e385162ac9fb` /
  `84dcd6733e39328f5282c54dacd37da0dc97cc5b8e25236f59cd0acf26631137` /
  `4f693a52a8bc53616dbcfb09a002646beff416d2137832d7e2d5455ce5b398d6` /
  `0f546b963b2892ddeab186eb4fcbde89ee3b861ca157657dd99dfa0221b82ba1`.
  The source files are intentionally not retained because both rotations
  rejected the schedule.
- v472 full B2 validator receipt:
  `/tmp/d128_native_compare_output_20260829/v472_full_b2_s4096.json`
  (8,517 bytes,
  `f181ae5f10c27e11a39326e1ac1a6c81b018db0af6d90a5b759780afcac3bfbe`).
- v469/v472 matched R23 and R24 matrices:
  `/tmp/d128_native_compare_output_20260829/v469_v472_head_score_b2_r23.json`
  (21,533 bytes,
  `786574947e1fbf4dd4bf53bb7638f395750e855edf71f02e4c113a987b46c622`)
  and
  `/tmp/d128_native_compare_output_20260829/v469_v472_head_score_b2_r24.json`
  (21,536 bytes,
  `e808ea8891cfb7b2f642d3d4573c85972141786b04c2f011021cd91fc952904b`).
- Final v472 binary/source/header/Makefile and selected SASS stream:
  `749f98c43680d3222513f038e7d400179a596a44c81556fcbac5d235b7ce62f6` /
  `d36d52074a9f2f1a44ed4f6d3fd2fafb18ba9211b7bdc6bba99ea4ef44b847da` /
  `9f6b57b7f41f8583b4d8cc490fb4c897503dc8dad0aa59d31b470be44bbda1f1` /
  `389f146c32571e592892ddc23a5673b37ef3b64b9f07bfc887eeabcaeb4a91eb` /
  `60e0e04420bd2a085553ede5c431e656b85254650be5e48def0022136b79279a`.
- v473 full B1/B2 validator receipts:
  `/tmp/d128_native_compare_output_20260829/v473_full_b1_s4096.json`
  (9,072 bytes,
  `3cfdf4c4899a539eb812bebe45946d2a2101c904e583d217045447222f63d6ae`)
  and
  `/tmp/d128_native_compare_output_20260829/v473_full_b2_s4096.json`
  (9,054 bytes,
  `cd550c572889b893b1760956a187972b5f92ae091e9b1c9b3074386ee998d28e`).
- Final v473 binary/source/header/Makefile:
  `1ac48ab790e12e28ec3c03819d5ecf8396e5d8a463cef4986883a428f9e7d977` /
  `22cf7cdd2b981968ba9f31023fee796c3fc912302274ffbb8b73353349b1c834` /
  `5569cbaa8b7224f4a223602d7a87399b17193304e1a1c4dacbb4c00d06fd7858` /
  `9c9fa07df0ccacfb974a5ff049cfe04fa375b6044bb2d3ea74d02a1d66c35056`.
- v473 embedded v445/v472 selected SASS streams:
  `1734b07c039af28b38c717ef283719c5fef5bcf01b9d7ac0e770df61701f6cf7` /
  `60e0e04420bd2a085553ede5c431e656b85254650be5e48def0022136b79279a`;
  both are byte-identical to their standalone streams.
- Rejected v475 matched R25/R26 matrices:
  `/tmp/d128_native_compare_output_20260829/v472_v475_cross_head_pipeline_b2_r25.json`
  (22,027 bytes,
  `02e29cdd4798b65c4356e427d067c1c2dfbb8b6d2f347e060fcb2209e5095d4d`)
  and
  `/tmp/d128_native_compare_output_20260829/v472_v475_cross_head_pipeline_b2_r26.json`
  (22,033 bytes,
  `1b96b8c12627dbb9aa1c515ada7c0b996e3def71f9a35f9f1d0a0b77dfed6ffe`).
- Rejected v476 full gate and matched R27 receipts:
  `/tmp/d128_native_compare_output_20260829/v476_full_b2_s4096.json`
  (8,810 bytes,
  `8fc91640ec5c49fbe11e0d8ca33eed4fcb66fe7094416d6526b8e4c1b01642c3`)
  and
  `/tmp/d128_native_compare_output_20260829/v472_v476_split_dq_release_b2_r27.json`
  (22,069 bytes,
  `a426561c965784af926e92cc754d902240992fc0ebe5ec7996caf8bab9d06ecf`).
- v477/v478 matched R28--R31 receipts:
  `v472_v477_peeled_split_dq_release_b2_r28.json` (22,276 bytes,
  `e8a2618f42f966cd01531a6c24ba43baec2a956c33e3d8962a02779012e8632e`),
  `v472_v477_peeled_split_dq_release_b2_r29.json` (22,276 bytes,
  `d98341dbdd8c44f766db9a3d61d0c58e9160a61a2e2b3d1347dff21154e1eb88`),
  `v472_v477_v478_two_chunk_release_b2_r30.json` (30,962 bytes,
  `d376824c906a79f56009ecdfc3d5afcf770a70d88a401cbf3ea95c9c9c345880`),
  and `v472_v477_v478_two_chunk_release_b2_r31.json` (30,965 bytes,
  `0e5f3a123245fbe3bd01468bde64106fff0cb237b674b19511562e55568c95fc`).
  All four reside under `/tmp/d128_native_compare_output_20260829/`.
- v480 matched R32/R33 receipts:
  `/tmp/d128_native_compare_output_20260829/v477_v478_v480_deferred_packed_d1_b2_r32.json`
  (31,596 bytes,
  `2c9cb9b86df666211a6b98ecec551c710b6c3ebc9d4b7908815e048266c9c747`)
  and the R33 sibling (31,598 bytes,
  `76bab9ef1e79fc846db335e3063276a7c54b9350d46c555c00e4797c5f466315`).
- v482 matched R34/R35 receipts:
  `/tmp/d128_native_compare_output_20260829/v478_v480_v482_deferred_packed_d0_d1_b2_r34.json`
  (31,781 bytes,
  `a7af9f8dc22aa05e005ce580a217ff9aff6ba0e3863732f50e7f4cd4bddd9de2`)
  and the R35 sibling (31,793 bytes,
  `54c89cbdac12df13f28aaedda30c09ca45b730e5b9d19c9b7d7382a6d250a0d7`).
- Final v482 binary/source/header/Makefile and standalone selected-SASS dump:
  `afea4d06e23c0cb56cdca5970fbeb9da1460682ccd501e2aa8e9cd5494ff0667` /
  `3039d99b22c4559ba209e7de1865dd8f524f33de6289a8cd7ccffb9b400b8eca` /
  `065eb9d3da50b08ec2fbb7d6bbd41da25eb6e5470aac61f9b58bfa25b659fae1` /
  `5290bd0159c533752146a675251ad88fe6c939400e9252855e16376e03bbb569` /
  `f8e022021db064e067b0d28411ff9fd8c9e3189da771a6bcf1a6268708076537`.
- v483 full B1/B2 validator receipts:
  `/tmp/d128_native_compare_output_20260829/v483_full_b1_s4096.json`
  (9,796 bytes,
  `05974f008a624a22727ddeec4704fa87a3668c15e332244442f7402bfa167d9a`)
  and
  `/tmp/d128_native_compare_output_20260829/v483_full_b2_s4096.json`
  (9,777 bytes,
  `5177f1f984065c55535466594a5e625db73835677a37218d6adae79f832678b7`).
- Final v483 binary/source/header/Makefile:
  `faa43c33753e75ec357522d18a78809282fdfe251820de7e5d4eecf91c8d4122` /
  `4121bcc98a243f4e3a42f6c522f3a26a202d3b4372ed4f836ea3fcbf2927b602` /
  `7ccfce00928282b4492043ddbd2c8acdab6a24b3723aee37d0cc858de0cf99e3` /
  `0fcd4717e24d310114b28f39a23f4727f3eb2a3c362342ba31f557d203013942`.
  Direct selected-function diffs verify both embedded v445 and v482 SASS
  streams are byte-identical to their standalone binaries.
- NCU permission-failure receipt:
  `/tmp/d128_ncu_v424_vs_cute_20260829/ncu_attempt_receipt.json`
  (5,360 bytes,
  `0514d931374e77df2e63b18d48d3ec54d0f1527f297aa533931c0e18a5f9cf06`)

v483 is now the selected native D128 integration candidate: B1 freezes v445,
B2 freezes v482, and both exact routes are numerically gated and competitive
under their matched isolated boundaries.  It has not yet been wired into or
measured in full-model training, so no end-to-end speedup is claimed here.

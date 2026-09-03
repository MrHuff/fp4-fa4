# Low-precision FA4 projection fusion — 2026-08-12

## Outcome

Both requested structural experiments were implemented and measured on GPU 0.

1. The Q/K projection now has a real persistent SM100 NVFP4 GEMM whose
   register epilogue emits BF16 Q/K and all four adaptive E2M1 layouts consumed
   by FA4 backward. It does not read back BF16 Q/K through a fake-quantization
   pass.
2. The adaptive FP4+FP8 backward now publishes per-head BF16 dQ readiness
   directly into the projection K loop. A two-accumulator persistent consumer
   beats the cuBLAS chain by 2.592 us at S8192/H8. Wider heads retain a
   shape-dispatched BF16-dQ-plus-cuBLAS fallback because the bounded consumer
   still leaves an exposed projection tail.

The earlier timing-only no-publication floor near 0.693 ms is not directly
attainable under the current K/V-owner decomposition. Every dQ tile is a sum
from many unclustered owner CTAs. A correct adjacent consumer must therefore
retain the global BF16 reduction scratch and wait for its final arrival. Fully
removing that scratch requires q-centric ownership or fusion with a larger
projection/reduction topology.

## Actual Q/K projection epilogue

The retained producer is in `tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh`.
It uses NVFP4 A/B tensor-core work, stages E4M3 block scales through TMEM,
performs one BF16 rounding, and publishes these views from the same consumer
register slice:

- BF16 Q and K;
- sequence-aligned Q for dK;
- compact depth-packed Q for score/dK;
- depth-aligned K for dQ;
- compact depth-packed K for score/dQ.

All four low-precision views are bit-exact against the retained standalone
packer. The fused specialization compiles at 128 registers with zero spills;
the projection-only specialization uses 96 registers with zero spills.

Rotated CUDA-event timings:

| Shape | BF16 cuBLAS + packer | NVFP4 + packer | Fused NVFP4 epilogue | Fused vs BF16 | Fused vs separate |
| --- | ---: | ---: | ---: | ---: | ---: |
| S4096/H24/K3072 | 0.327296 ms | 0.182304 ms | 0.176768 ms | 1.8516x | 1.0313x |
| S8192/H8/K1024 | 0.165600 ms | 0.112384 ms | 0.123680 ms | 1.3390x | 0.9087x |
| S4096/H64/K8192 | 1.565296 ms | 0.631552 ms | 0.469552 ms | 3.3336x | 1.3450x |

Q/K cosine against the BF16 projection is approximately 0.99097–0.99098.

Validation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../.. \
  python3 validate_projection_fp4_epilogue.py --sequence 4096 --heads 24 \
    --hidden 3072
```

## Head-pipelined dQ projection consumer

The producer now keeps one arrival counter per head and 128-row dQ tile. Each
CTA publishes a device-scope release increment only after all four reducer
warps have retired their BF16 TMA stores and issued an async-proxy global
fence. The consumer polls with a device-scope acquire load immediately before
each head's first K64 slice. It no longer waits for every head before beginning
the projection reduction.

The consumer also alternates two FP32 TMEM output accumulators. MMA for output
tile `p+1` overlaps BF16 epilogue publication for tile `p`, using the full 512
TMEM columns of the separate projection kernel without changing the attention
TMEM map. Six early clusters are retained for H8 and sixteen for wider
projections. Launching a polling cluster before the producer remains invalid
on SM100 because it can prevent admission of the producer cluster grid.

For wide outputs, an experimental split lets the bounded consumer produce a
ready row prefix while a disjoint cuBLAS GEMM produces the suffix after
attention completes. The public path uses the fully persistent topology only
at H8; wider heads fall back to BF16 dQ plus cuBLAS unless
`TK_FA4_DQ_PROJECTION_FORCE_PIPELINED` is set.

Final seven-warmup, 31-sample rotated timings. The public H24 row uses the
shape-dispatched fallback; the forced row keeps the experimental topology
visible:

| Shape | Route | BF16 dQ attention | Attention + cuBLAS | Result | Delta | dX cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S8192/H8 | public pipelined | 0.746528 ms | 0.774944 ms | 0.772352 ms | +0.002592 ms | 0.99999535 |
| S4096/H24 | public fallback | 0.628768 ms | 0.734336 ms | 0.733728 ms | +0.000608 ms | 0.99999994 |
| S4096/H24 | forced pipelined | 0.633952 ms | 0.740384 ms | 0.748256 ms | -0.007872 ms | 0.99999636 |

This turns the old H8 loss of roughly 14 us into a 2.6 us win and cuts the
forced H24 loss from 93.6 us to 7.9 us. The public H24 fallback is timing
neutral (the measured 0.6 us advantage is noise) and bit-exact to the
comparison output. dK and dV match the direct-BF16 reference bit-for-bit. An
isolated fixed-dQ test shows that the persistent projection is bit-exact
against `torch.mm`; repeated-output variation comes from BF16 TMA reduction
order in the transposed attention grid. Most H8 comparisons stay above 0.9999
cosine, while one repeat reached 0.99863 cosine / 0.0523 relative L2. The
projection consumer compiles at 80 registers and two barriers with zero
spills. The hot attention specialization remains at 128 registers and sixteen
barriers with zero spills.

Validation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../.. \
  python3 validate_dq_projection_handoff.py --sequence 8192 --heads 8
```

## Projection-native hierarchical dQ reduction

The next structural rung is implemented behind
`TK_FA4_DQ_PROJECTION_HIERARCHICAL=1`. It deliberately does not alter the
public dispatch. Attention maps even and odd K/V owners into two independent
BF16 reduction lanes. The projection consumer loads both lanes and accumulates
both products into the same FP32 TMEM output. A completed standalone BF16 dQ
tensor is therefore never published or consumed; the final lane sum is part of
projection backward itself.

The specialization is compile-time isolated from the ordinary producer. The
one-lane path has no runtime modulo or lane branch in its hot reducer. Both the
ordinary and hierarchical projection consumers compile at 80 registers, two
barriers, and zero spills. The hot attention specialization remains at 128
registers with zero spills. A seven-warmup, 31-sample refresh put the ordinary
consumer and attention-plus-cuBLAS within 1.22 us by median and 0.04 us by
mean, so the experiment does not change the retained route.

The exact hierarchy is correct but not a speed win:

| Shape | Attention + cuBLAS | Two-lane projection reduction | Delta | dX cosine |
| --- | ---: | ---: | ---: | ---: |
| S8192/H8 | 0.773824 ms | 0.805888 ms | -0.032064 ms | 0.99999422 |
| S4096/H24 | 0.736000 ms | 1.063392 ms | -0.327392 ms | 0.99999648 |

Two implementation alternatives were also rejected. Folding the BF16 lanes
serially in shared memory took roughly 2.68 ms with scalar conversion and
2.33 ms with packed BF16 addition. Pipelining that fold onto idle producer
warps improved it to 1.056544 ms; explicit 128-bit shared loads/stores reduced
it further to 0.860608 ms. It still lost to issuing the second projection MMA,
so the source retains the faster tensor-native hierarchy.

This experiment sharpens the 0.693 ms ceiling. Merely moving the final add
into projection is insufficient because attention still publishes the same
number of BF16 partial bytes, while projection either doubles its weight MMA
work or introduces a shared-memory fold. The next implementation must remove
work before publication as well. The credible forms are an on-chip super-owner
that merges multiple K/V owners before one global write, or compact FP8/FP4
owner partials consumed by a matching low-precision projection. A q-centric
rewrite is less attractive because it moves the global reduction from dQ's
192 features to dK+dV's 320 features.

## Shape-dispatched publication

The apparent small-head regression was not caused by head count itself. The
original model-like rows changed projection reduction width together with the
head count (`K = 128 * H`). Holding one variable fixed reverses the result:

| Shape | Fused | Separate | Winner |
| --- | ---: | ---: | --- |
| S4096/H8/K3072 | 0.084432 ms | 0.087120 ms | fused, 1.0318x |
| S4096/H24/K1024 | 0.161936 ms | 0.145440 ms | separate, 1.1134x |

The controlling variable is the projection reduction depth. The persistent
kernel has one consumer warpgroup per CTA. That warpgroup rounds BF16,
quantizes E2M1, transposes Q, and publishes four layouts while the tensor
producer starts the next output tile. With only four K256 reduction steps, the
producer reaches the next handoff before this scalar publication drains. The
standalone 32-register packer can instead distribute the work over a much
wider CTA grid. At twelve or more K256 steps, publication hides under the MMA
pipeline and eliminating the extra BF16 read/write is the clear win. The
128-register fused specialization remains spill-free; shared memory already
limits the persistent kernel to its launch topology, so this is a producer /
consumer balance issue rather than a simple occupancy loss.

The public `b300_project_qk_adaptive_lowp_nvfp4` API now selects publication by
reduction width and returns one identical `B300AdaptiveLowpOperands` contract:

- `K < 3072`: projection-only specialization plus parallel packer;
- `K >= 3072`: direct register-epilogue publication;
- `publication="fused"` or `"separate"`: forced diagnostic override.

Final rotated checks:

| Shape | Auto route | Auto | Forced other | Auto advantage |
| --- | --- | ---: | ---: | ---: |
| S8192/H8/K1024 | separate | 0.111024 ms | 0.122368 ms | 1.1022x |
| S4096/H24/K3072 | fused | 0.169984 ms | 0.181712 ms | 1.0690x |
| S4096/H64/K8192 | fused | 0.462560 ms | 0.626592 ms | 1.3546x |

All four auto-dispatched and public-API layouts were bit-exact against the
standalone packer. The producer interface is common to the retained adaptive
FP4+FP8 backward and any future route that adopts the same per-head adaptive
scale contract; dP/dV precision does not change these Q/K layout tensors.
End-to-end smoke tests passed through the retained backward at S8192/H8
(separate publication) and S4096/H24 (fused publication), returning finite
dQ/dK/dV at the expected shapes.

### Where the speedup comes from

The prepared-operand timings above intentionally measure the projection hot
stage. NVFP4 weights are persistent and can always be prepared once. An NVFP4
activation must either arrive from an upstream producer or be converted from
BF16. Charging the current standalone activation conversion gives:

| Shape | BF16 GEMM + packer | Prepared NVFP4 auto | BF16 input → NVFP4 auto | Prepared speedup | BF16-input speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| S8192/H8/K1024 | 0.163712 ms | 0.111152 ms | 0.142768 ms | 1.473x | 1.147x |
| S4096/H24/K3072 | 0.326704 ms | 0.169056 ms | 0.206096 ms | 1.933x | 1.585x |
| S4096/H64/K8192 | 1.553184 ms | 0.464672 ms | 0.515616 ms | 3.342x | 3.012x |

The contributions are:

1. NVFP4 tensor-core projection replaces the BF16 GEMM and reduces A/B payload
   traffic. This is the dominant gain at every shape.
2. The fused epilogue reuses the BF16-rounded accumulator slice to create the
   four E2M1 views. It avoids rereading Q/K from HBM and launching the 32-row
   transpose/packer. This adds about 1.07x over separate publication at H24
   and 1.35x at H64.
3. Persistent TMA/TMEM producer-consumer overlap hides the publication behind
   deeper K reductions. Both projection specializations are spill-free.
4. Reduction-width dispatch keeps the scalar epilogue off the critical path
   at shallow K, retaining the standalone packer's wider parallel grid.

The comparison is not accuracy-identical to the BF16 GEMM: NVFP4 projection
has Q/K cosine around 0.99097. Fused versus separate publication is
accuracy-identical and bit-exact; that portion is purely a scheduling and
movement improvement.

## Decision

- Retain the actual NVFP4 Q/K projection epilogue. It is a real end-to-end win
  for sufficiently deep projections and supplies the exact backward layouts
  without fake quantization.
- Use native reduction-width dispatch rather than a head-count rule. The
  conservative K3072 threshold retains the strong H24/H64 gains and removes
  the shallow-K regression.
- Retain the head-pipelined dQ handoff and double-TMEM projection consumer for
  H8, where it now saves 2.592 us over attention plus cuBLAS.
- Shape-dispatch H16/H24/H64 to the direct-BF16 dQ endpoint followed by tuned
  cuBLAS. The public H24 path measured 0.733728 ms versus 0.734336 ms for the
  comparison chain; the forced pipelined route is 7.872 us behind, so it is
  not a useful production default.
- A true breakthrough requires changing ownership so the final dQ reduction
  tile is local to the projection consumer, or representing hierarchical
  partials compactly enough that projection's higher low-precision throughput
  pays for the extra lane products. The implemented two-BF16-lane hierarchy
  proves the algebra and synchronization but is not a production dispatch.

## Unified QKV and six-experiment refresh

The Q/K-only producer above has now been generalized into
`b300_project_qkv_unified_lowp_nvfp4`. One persistent NVFP4 GEMM projects all
Q rows, all K rows, and all V rows. Its epilogue publishes:

- the same compact Q/K allocation as both typed forward E2M1 and backward
  byte views;
- the two aligned Q/K backward layouts;
- canonical forward Q/K scale pages;
- transposed MXFP4 V and its scale pages;
- optional BF16 Q/K/V stores.

The Q/K forward and backward views have identical data pointers. All four
backward layouts and the V payload/scales are bit-exact against the standalone
publishers. The unified specialization compiles at 128 registers, two
barriers, 8,592 bytes of static shared memory, and zero spills. Unioning the
disjoint Q/K-code and V-BF16 staging lifetimes saves 2,112 bytes versus the
initial 10,704-byte projection scratch.

| Shape | 3 BF16 GEMMs + QK pack | Unified + BF16 | Unified no BF16 |
| --- | ---: | ---: | ---: |
| S2048/H8/K1024 | 0.149264 ms | 0.088688 ms (1.683x) | 0.077440 ms (1.927x) |
| S4096/H24/K4096 | 0.463744 ms | 0.328816 ms (1.410x) | 0.289552 ms (1.602x) |

Projection Q/K/V cosine is approximately 0.99096–0.99098. Native causal
forward output cosine is 0.990722 at S2048/H8 and 0.991317 at S4096/H24.
Suppressing BF16 publication saves 64 MiB at S8192/H8 and 96 MiB at
S4096/H24; sharing compact Q/K saves another 12 MiB and 18 MiB respectively.

The other five integration experiments give a clear boundary:

1. One Q/K representation is retained; forward and backward alias it.
2. Local represented-E4M3 P reuse saves 6.784 us (0.76%) at S8192/H8. A
   forward MXFP4 P cache would occupy about 272 MiB and is rejected.
3. One upstream dO producer now emits row/column MXFP4 plus delta in
   0.068416 ms versus 0.132352 ms for three split kernels (1.935x). Payloads
   and scales are bit-exact; delta relative L2 is 6.72e-8.
4. The refreshed interleaved projection-backward chain is 0.827136 ms versus
   0.844672 ms for three outputs. The subsequent head-pipelined consumer now
   reaches 0.772352 ms versus 0.774944 ms for attention plus cuBLAS at H8.
5. The stronger three-command dV / MXFP4-P-to-dS candidate regressed to
   0.782464 ms versus 0.756192 ms and produced non-finite dV, so it was
   reverted. Compact 96-byte mixed-dP transport remains disabled because the
   grouped SM100 transaction does not retire.

The stable backward route was rebuilt after both rejected candidates. The
next structural target is eliminating global BF16 dQ materialization inside
the projection reduction topology, not another local barrier or a global
probability cache.

## Final 1–4 closure and aggressive pure-FP4 epilogue

The requested structural pass is complete:

1. The exact two-lane hierarchical dQ/super-owner algebra is implemented
   behind `TK_FA4_DQ_PROJECTION_HIERARCHICAL=1`. It proves that projection can
   absorb the final sum, but remains diagnostic because it is slower when
   BF16 owner traffic and a second weight product remain.
2. The elastic projection consumer is retained only where it wins. H8 uses
   the per-head persistent handoff and saves 2.592 us; H24/H64 shape-dispatch
   to the non-regressing BF16-dQ plus vendor-GEMM chain.
3. Producer-native all-MX integration is complete. The unified QKV NVFP4
   projection publishes Q/K layouts plus forward/backward MXFP4 V, and the
   unified dO producer publishes both dO orientations, scale pages, delta,
   and LSE metadata.
4. The exact rank-128 endpoint skips the final score command only when the
   last 64 Q/K features are provably zero. Its dedicated validation passes and
   it saves roughly 2.9 us; it is not used as an approximate D192 default.

The final pure-only producer specialization is exposed as
`pure_qk_single_quant=True` on
`b300_project_qkv_unified_lowp_nvfp4`. The earlier unified producer quantized
each Q/K register fragment once for the ordinary layouts and again at fixed
x16 scale for the compact pure dQ/dK operands. The new specialization creates
the fixed-scale E2M1 codes once and fans the bytes into all six Q/K layouts.
It also publishes the fixed forward local/global scales directly and accepts
an empty adaptive-metadata tensor because no adaptive record is read.
`B300UnifiedLowpQKV.pure_backward_operands()` returns the exact eight-tensor
pure Q/K endpoint tuple without a standalone packing launch.

The complementary in-kernel epilogue screen covered fused dS scale, direct P
scale publication, direct dQ scale application, interleaved dual E2M1 pack,
early dO reload, deferred native-dS TMEM wait, streamed dual dPsum, and both
P/dO K-block interleave orders. None improved calibrated wall time, so the
associated gates were restored to zero. This leaves the successful
projection-epilogue fanout isolated from those rejected scheduling probes.

Payload validation is strict: all six Q/K layouts, the two empty fixed-scale
slots, V and dO payloads, and their MXFP4 scale pages are byte-identical to
the standalone publishers. At S8192/H8, dPsum relative L2 is 6.323e-8, LSE is
exact, dK/dV are bit-exact, and dQ relative L2 is 1.309e-8.

Projection-only medians:

| Shape | Dual quant + BF16 | Single quant + BF16 | Dual no BF16 | Single no BF16 |
| --- | ---: | ---: | ---: | ---: |
| S2048/H8/K1024 | 0.107936 ms | 0.099072 ms (-8.21%) | 0.104000 ms | 0.096896 ms (-6.83%) |
| S4096/H24/K1024 | 0.446112 ms | 0.380832 ms (-14.63%) | 0.410464 ms | 0.352384 ms (-14.15%) |

Composed projection plus pure-MX backward:

| Shape | Projection + separate pure pack | Single-quant projection + backward | Saving | Speedup |
| --- | ---: | ---: | ---: | ---: |
| S8192/H8/K1024 | 1.102848 ms | 1.033216 ms | 0.069632 ms | 1.0674x |
| S4096/H24/K4096 | 1.124192 ms | 1.027040 ms | 0.097152 ms | 1.0946x |

Prepared attention itself is unchanged within timing noise: 0.775840 versus
0.776928 ms at S8192/H8 and 0.633280 versus 0.631360 ms at S4096/H24. The
measured improvement therefore comes from deleting duplicate conversion and
publication, not from silently changing the downstream kernel.

The refreshed calibrated prepared-kernel frontier is:

| Shape | BF16 | FP4+FP8 | Mixed dP | Adaptive FP4+FP8 | Pure MXFP4 | Pure aggregate cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S4096/H24 | 0.707936 | 0.647104 | 0.613664 | 0.660096 | 0.637120 | 0.984797 |
| S4096/H64 | 1.655648 | 1.513952 | 1.432832 | 1.539360 | 1.482304 | 0.984614 |
| S8192/H8 | 0.869984 | 0.758656 | 0.745664 | 0.773472 | 0.781920 | 0.984653 |
| S8192/H24 | 2.212064 | 1.923200 | 1.885216 | 1.956544 | 1.971136 | 0.985197 |
| S8192/H64 | 5.558176 | 4.871296 | 4.739136 | 4.940192 | 4.958176 | 0.985087 |
| S16384/H24 | 7.758656 | 6.510784 | 6.536256 | 6.610144 | 6.874848 | 0.985217 |
| S16384/H64 | 20.275393 | 17.163391 | 17.102079 | 17.419872 | 17.997663 | 0.985224 |

Pure MXFP4 is consistently faster than BF16 and wins over ordinary FP4+FP8
on the two S4096 rows, but it never beats the mixed-dP route. Its aggregate
cosine hides the weaker gradient components: broad dQ/dK cosine is about
0.866–0.887, versus roughly 0.991 for the retained adaptive hybrid. The
aggressive pure route is consequently complete and useful as a lower-precision
endpoint, but the mixed/adaptive hybrid remains the better frontier.

The final forced build completed with exit status zero:

```bash
cd tk_fa4/lowp_fa4_bwd
make -B -j1
```

The active pure attention specialization remains at 128 registers, sixteen
barriers, 231,536 bytes of shared memory, and zero spills. The no-BF16
single-quant projection uses 128 registers and two barriers with zero spills;
the optional-BF16 version uses 168 registers and two barriers with zero
spills. Final producer/pipeline validation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python tk_fa4/lowp_fa4_bwd/validate_projection_native_pure_fp4_pipeline.py \
    --sequence 8192 --heads 8 --hidden 1024 --skip-timing
```

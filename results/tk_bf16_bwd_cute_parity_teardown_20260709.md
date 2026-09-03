# TK BF16 FA4 Backward vs CuTe DSL BF16 Backward Parity Teardown

Date: 2026-07-09

Scope: BF16 backward only, shape `S=2048, B=1, H=2, Dqk=192, Dv=128`, causal. MXFP4 and forward paths were not changed.

Baseline artifact: `results/tk_bf16_bwd_vs_cute_dsl_bf16_s2048h2_20260709T172738Z.json`

## Executive Summary

The routed TK hot candidate is slow for structural reasons, not because of preprocessing, scratch allocation, launch count, or a small ownership bug. CuTe DSL runs the BF16 backward as:

1. preprocess: compute `dPsum`, convert `LSE` to log2, zero `dq_accum`
2. one SM100 2-CTA main kernel: compute softmax replay, `dV`, `dK`, and FP32 `dQ_accum`
3. postprocess: convert/reduce `dQ_accum` to BF16 `dQ`

The comparable TK hot route uses a small preprocess plus an async memset, then launches separate dK/dV and dQ kernels on split streams. The dQ kernel overlaps with dK/dV, so the wall time is dominated by TK dK/dV. The TK dK/dV kernel is a Cluster1 warp-MMA path with 255 registers, stack usage, and register spills; CuTe's main kernel is a 2-CTA tcgen05/TMEM warp-specialized kernel with 128 registers and no reported spills.

There is no low-risk TK one-line change from this teardown. The next useful implementation is a new BF16 backward SM100 candidate that ports the CuTe schedule structure into TK: 2-CTA cluster, 512 threads, tcgen05/TMEM accumulators, fused dK/dV plus dQ accumulation, and the existing dQ postprocess shape.

## Source Path Confirmed

The CuTe DSL path used by the benchmark is the regular BF16 backward path, not the native FP4-QK backward path:

- TK benchmark wrapper: `tk_fa4/fp4_pv_experiments.py`
  - forward setup calls `flash_attn.cute.interface.flash_attn_func(..., return_lse=True)`
  - backward calls `flash_attn.cute.interface._flash_attn_bwd(..., deterministic=False)`
- CuTe interface: `flash-attention/flash_attn/cute/interface.py`
  - for SM100 and `head_dim=192`, sets `m_block_size=128`, `n_block_size=128`, `cluster_size=2`, `use_2cta_instrs=True`
  - allocates `dq_accum` as FP32 and launches preprocess, main backward, and dQ postprocess
  - for this MHA shape, `qhead_per_kvhead=1`, so `dKV_postprocess=False`; `dK` and `dV` are produced directly by the main kernel
- CuTe main kernel: `flash-attention/flash_attn/cute/flash_bwd_sm100.py`
- CuTe preprocess/postprocess: `flash-attention/flash_attn/cute/flash_bwd_preprocess.py`, `flash-attention/flash_attn/cute/flash_bwd_postprocess.py`

The FP4-specific file `flash-attention/flash_attn/cute/fp4_flash_bwd_sm100.py` has a similar SM100 structure, but it is not the path used for this BF16 baseline comparison.

## Timings

Graph-level timing from the restored baseline:

| Path | Graph time |
| --- | ---: |
| CuTe DSL BF16 backward | 0.120-0.125 ms |
| TK routed hot candidate | 0.425-0.429 ms |
| Ratio | TK is about 3.4-3.6x slower |

Bounded torch profiler run, same shape:

| Path | Device op | Time |
| --- | --- | ---: |
| CuTe | `FlashAttentionBackwardPreprocess` | 6.497 us |
| CuTe | `FlashAttentionBackwardSm100` main | 97.601 us |
| CuTe | `FlashAttentionBackwardPostprocess` | 9.440 us |
| TK | `preprocess_kernel` | 3.456 us |
| TK | device memset | 2.656 us |
| TK | `main_kernel_causal_fullseq_dkdv_only<config<128,128,192,128,1>, float>` | 409.731 us |
| TK | `main_kernel_causal_fullseq_dq_only_clustered<...>` | 234.978 us |

Interpretation: TK launches more work, but launch count and preprocessing are not the primary issue. TK dQ overlaps with TK dK/dV on the split stream route, so the 410 us dK/dV kernel is the critical path. Even if dQ were free, the routed TK candidate would still be roughly 3.3x slower than CuTe's whole backward graph.

## Resource Metadata

CuTe generated files were located in:

`tk_fa4/results/cute_dsl_bwd_dump_20260505T162123Z`

Persistent CuTe DSL cache is disabled unless `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`, and no newer matching cache dump was found. The existing dump is still useful for schedule/resource inspection.

Resource usage from `cuobjdump --dump-resource-usage`:

| Path | Kernel | Registers | Stack | Static shared | Local | Constant | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CuTe | preprocess | 80 | 0 | 0 | 0 | 1188 | small scalar/vector preprocess |
| CuTe | main BF16 bwd | 128 | 0 | 1024 | 0 | 2492 | dynamic shared is launch-time; source comment indicates about 231 KB for 2-CTA |
| CuTe | dQ postprocess | 168 | 0 | 1024 | 0 | 1028 | FP32 `dq_accum` to BF16 `dq` |
| TK | preprocess | 47 | 0 | 0 | 0 | 1920 | small |
| TK | dK/dV only | 255 | 552 | 85016 | 0 | 2816 | ptxas reports 1596 B spill stores, 3988 B spill loads |
| TK | dQ clustered | 168 | 16 | 190532 | 0 | 5120 | no reported spill in current build |
| TK | dQ reduction chunks | 72 | 0 | 0 | 0 | 5120 | present in binary, not the observed critical path |

CuTe SASS/PTX markers observed in the dump include `tcgen05.mma.cta_group::2`, `cp.async.bulk.tensor...cta_group::2`, `UTMALDG.4D.2CTA`, `UTMASTG.4D`, and `USETMAXREG`. Those match the source-level design: 2-CTA TMA loads, tcgen05 UMMA/TMEM accumulators, TMA stores, and warp-specific register budgeting.

## CuTe Schedule Details

For this shape CuTe configures:

| Item | CuTe BF16 SM100 backward |
| --- | --- |
| CTA tile | `tile_m=128`, `tile_n=128`, `tile_hdim=192` |
| Cluster | `cluster_shape_mn=(2,1)`, `cluster_size=2`, `use_2cta_instrs=True` |
| Threads | 512 threads per CTA |
| K/Q score tile | effective grouped K dimension is 256 across the 2-CTA group |
| MMA tiler, S | `(256,128,192)` for `S = K @ Q.T` |
| MMA tiler, V/dO | `(256,128,128)` for `dP.T = V @ dO.T` and `dV = P.T @ dO` |
| MMA tiler, dK | `(256,192,128)` for `dK = dS.T @ Q` |
| MMA tiler, dQ | `(128,192,256)` for `dQ = dS @ K` |
| Warp roles | reduce warps 0-3, compute warps 4-11, MMA warp 12, load warp 13, relay warp 14, empty warp 15 |
| Register budgets | for D=192 causal: reduce 136, compute 136, load 104, MMA 104, empty 24 |
| Pipelines | `Q_stage=1`, `dO_stage=1`, `sdKVaccum_stage=2`; dQ reduce stages are based on 32-column chunks |
| Accumulators | TMEM for score/dP/dS and dK/dV accumulators; FP32 global `dq_accum` for dQ |
| Outputs | `dK` and `dV` direct from main kernel; `dQ` through postprocess |

The important point is that CuTe's 97 us main kernel owns both dK/dV and dQ accumulation. It is not a split dK/dV-only pass plus a separate dQ replay pass.

## TK Hot Route Details

Current Python hot wrapper routes through:

`b300_mha_bwd_hot_cute16_candidate_internal`

The active TK route in `tk_fa4/b300_bwd_cute16_candidate.cuh` uses:

- `launch_preprocess`
- `launch_backward_dkdv_only<Cluster1Config, DkdvOutT>`
- `launch_backward_dq_only_clustered<HotClusteredDqConfig, DqOutT>`
- split streams/events so dQ can overlap dK/dV

The dK/dV kernel in `tk_fa4/b300_bwd_cute16_kernel_candidate.cuh` is:

`main_kernel_causal_fullseq_dkdv_only<config<128,128,192,128,1>, float>`

This route is finite and close enough on dK/dV correctness to be usable as a baseline, but it is resource bound:

- Cluster1 dK/dV ownership, not the CuTe 2-CTA grouped schedule.
- dK/dV accumulators are held in high-pressure warp-level register state.
- ptxas reaches 255 registers and spills.
- It does not use a single SM100 tcgen05/TMEM fused main kernel for dK/dV plus dQ accumulation.

## Structural Differences To Port

| Area | CuTe DSL BF16 bwd | TK routed hot candidate | Porting implication |
| --- | --- | --- | --- |
| Main work decomposition | one fused main kernel computes dK, dV, and dQ accumulation | separate dK/dV and dQ kernels | split streams are not enough; port fused main ownership or at least first fix dK/dV schedule |
| CTA/cluster layout | 2-CTA cluster for D=192, grouped K tile 256 | dK/dV route uses Cluster1 | implement 2-CTA dK/dV ownership for BF16 |
| SM100 instructions | tcgen05 UMMA with TMEM accumulators | warp-level MMA/register accumulators | dK/dV accumulator storage must move out of 255-register per-thread pressure |
| Register control | `setmaxregister` budgets by warp role | dK/dV hits 255 regs and spills | use warp-specialized roles and explicit register budgeting |
| dK/dV output | direct from main kernel, no postprocess for MHA | direct from dK/dV-only kernel | output format is compatible; schedule is the issue |
| dQ strategy | FP32 `dq_accum` in main, BF16 postprocess | separate dQ clustered replay path | dQ is secondary until dK/dV is below about 235 us, then fused dQ accumulation matters |
| Preprocess/scratch | preprocess 6.5 us and dQ accumulator init | preprocess plus memset about 6.1 us total | not the gap |
| Launch count | 3 launches | preprocess, memset, dK/dV, dQ | not the dominant gap at this shape |

## Correctness Notes

Current routed TK candidate is finite at this shape. Prior restored-baseline comparison showed:

| Tensor | Relative L2 error vs CuTe |
| --- | ---: |
| `dq` | about 0.90 eager, higher under the exact graph comparison artifact |
| `dk` | about 0.0029 |
| `dv` | about 0.0017 |

Candidate2 is faster but produces non-finite `dk`, and the last ownership patch made it slower rather than closer to CuTe. It should stay out of the immediate critical path.

## Next Implementation Plan

No source edit was made in this teardown. The obvious next change is not low risk; it is a new BF16 backward SM100 candidate route.

Recommended plan:

1. Add a new TK BF16 backward candidate route instead of mutating the current routed hot candidate. Keep `b300_mha_bwd_hot_cute16_candidate_internal` as the finite fallback.
2. Implement a fixed-shape first milestone for `Dqk=192, Dv=128, causal, BSHD`:
   - `tile_m=128`, `tile_n=128`
   - `cluster_shape=(2,1)`, 2-CTA grouped K tile 256
   - 512 threads per CTA
   - warp roles matching CuTe: reduce 0-3, compute 4-11, MMA 12, load 13, relay 14, empty 15
   - explicit register budgets near CuTe's 136/136/104/104/24 split
3. First performance milestone: replace only the current TK dK/dV kernel with a 2-CTA tcgen05/TMEM dK/dV route while keeping the existing dQ clustered route. This targets the current 410 us critical path directly and gives a clean A/B before fusing dQ.
4. Second milestone: add dQ accumulation into the new main kernel and reuse a small dQ postprocess. This matches CuTe's decomposition and removes the separate 235 us dQ replay once dK/dV is no longer the bottleneck.
5. Validate in this order:
   - finite `dq/dk/dv` at `S=2048,H=2`
   - relative/max errors vs CuTe from the same forward/LSE
   - graph timing vs current TK route and CuTe DSL
   - ptxas/cuobjdump resources: target no dK/dV spills, registers far below 255, and dynamic shared consistent with one resident 2-CTA cluster

Expected bottleneck after milestone 1: if dK/dV falls near or below 235 us, the existing separate dQ route becomes the wall-time limiter. At that point the CuTe-style fused main plus dQ postprocess is required to approach the 0.12 ms CuTe graph time.

## Implementation Follow-Up: 2CTA and Candidate2 Gates

The first 2CTA dK/dV scaffold was implemented as a new opt-in route rather than
promoting over the finite fallback. It is finite but not faster:

| Path | Event / component | Result |
| --- | ---: | --- |
| TK fallback event median | 0.432736 ms | finite |
| TK fallback dK/dV split | 415-423 us | current critical path |
| TK 2CTA experimental event median | 0.470592 ms | finite, slower |
| TK 2CTA experimental dK/dV split | 463-464 us | slower than fallback |
| CuTe BF16 event median in same harness | 0.190272 ms | finite |

Candidate2 remains the fastest local route, but every local correctness patch
failed:

| Probe | Event median | Resource change | Result |
| --- | ---: | --- | --- |
| restored candidate2 | 0.226240 ms | 120/127 regs, 0 spills | invalid: `dk` non-finite |
| no-patch candidate2 | 0.240736 ms | same low-reg shape | invalid: main kernel already produces bad `dk` |
| no-patch masked diagonal | 0.238656 ms | causal 128 regs, 4/8 B spills | invalid: `dk` non-finite |
| no-patch direct dK store | 0.236480 ms | causal 128 regs, 4/8 B spills | invalid: relay is not root cause |
| masked diagonal with patches | 0.238336 ms | causal 128 regs, 4/8 B spills | invalid: `dk` non-finite |

The retained code state is therefore:

- finite fallback remains the default route.
- 2CTA dK/dV route remains experimental/opt-in only.
- candidate2 source is restored with no experimental diff.

Next implementation should stop making local candidate2 micro-patches and build
a true CuTe-like SM100 dK/dV kernel: TMEM/tcgen05 accumulator ownership, explicit
warp-role register budgeting, and a real masked diagonal path in the main
schedule. The acceptance gate remains finite `dq/dk/dv` and dK/dV below the
fallback's `415-423 us` before considering any route promotion.

## Implementation Follow-Up: Existing TMEM dK/dV-Only Exposure

Attempted next step: expose the existing `bwd_cute16_kernel::launch_backward_dkdv_only`
as a new internal opt-in route and pair it with the same BF16 preprocess,
causal patch sidecars, and clustered dQ route used by the 2CTA experiment.
This was intended as the smallest CuTe-like TMEM/tcgen05 baseline before
writing a new bespoke dK/dV kernel.

The attempt was rejected and reverted.

| Item | Result |
| --- | --- |
| Source state | temporary `candidate_tmem_dkdv` route added, then reverted |
| Resource result while patched | `dkdv_only_kernel<causal>` used 128 registers but still spilled 4412 B stores / 5608 B loads, stack 1544 B, smem 140404 B |
| Register budgeting | ptxas emitted `setmaxnreg` ignored warning, so the role budget was not a usable fix |
| Runtime gate | bounded 360 s harness timed out before the TMEM route emitted a split timing line |
| Fallback during same harness | dK/dV 413.6-420.8 us after warmup, total 434-444 us |
| 2CTA experimental during same harness | dK/dV 460.6-466.9 us after warmup, total 479-489 us |
| Final action | removed TMEM binding/wrapper and register-budget patch; rebuilt safe extension |

Conclusion: a thin wrapper around the existing `bwd_cute16_kernel`
dK/dV-only kernel is not a viable milestone. Even before runtime correctness,
it had large spills and then hung under the opt-in split wrapper. The next
implementation should be a new dK/dV kernel with a deliberately smaller live
state and a clear standalone semaphore/TMEM protocol, not a host-side exposure
of the old CuTe16 helper.

## Restored-State Validation: CuTe vs Fallback vs 2CTA

Run after reverting the failed TMEM-helper exposure and rebuilding the safe
extension. One input set was used for all routes:
`B=1, S=2048, H=2, Dqk=192, Dv=128, causal=True`, seed `20260709`.
The TK routes used the CuTe BF16 forward output/LSE as saved-forward inputs.

| Route | Event median | Event min/max | dK/dV split after warmup | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL FA4 BF16 bwd | 0.239056 ms | 0.226048 / 0.722048 ms | n/a | finite |
| TK fallback | 0.471632 ms | 0.469280 / 0.488704 ms | 412.45-417.47 us, median about 415.06 us | finite |
| TK 2CTA experimental | 0.512560 ms | 0.508192 / 0.535712 ms | 455.07-462.30 us, median about 459.52 us | finite, slower |

Correctness vs CuTe DSL BF16 backward on the same saved-forward tensors:

| Route | Tensor | rel L2 | max abs | bad values |
| --- | --- | ---: | ---: | ---: |
| TK fallback | `dq` | 0.878111 | 0.0281992 | 0 |
| TK fallback | `dk` | 0.0027386 | 0.000177894 | 0 |
| TK fallback | `dv` | 0.00166496 | 0.00195163 | 0 |
| TK 2CTA experimental | `dq` | 0.878111 | 0.0281992 | 0 |
| TK 2CTA experimental | `dk` | 0.0027386 | 0.000177894 | 0 |
| TK 2CTA experimental | `dv` | 0.00166496 | 0.00195163 | 0 |

Interpretation: restored state is safe. The 2CTA route remains opt-in only and
is slower than fallback. The next viable patch still has to beat the fallback
dK/dV critical path around 415 us before any promotion discussion.

## Implementation Follow-Up: Standalone Hot-TMEM dK/dV Scaffold

Attempted a new internal opt-in dK/dV route paired with the existing clustered
dQ route. The route was not promoted and was fully removed after failing the
acceptance gate.

| Variant | Resource result | Runtime result | Correctness result | Action |
| --- | --- | --- | --- | --- |
| New standalone scaffold using `hot_compute_dkdv_loop` | 168 regs, 56 B spill stores / 64 B spill loads, about 140 KB smem | event median 0.318 ms; dK/dV about 237-244 us after warmup | finite but wrong: `dk` rel L2 1.349, `dv` rel L2 0.936 | rejected |
| Scaffold switched to exact TMEM loop | 168 regs, 756 B spill stores / 672 B spill loads, 124004 B smem | event median 0.4526 ms; dK/dV about 374-382 us | invalid: `dk`/`dv` non-finite | rejected |
| Exact loop plus explicit TMEM zero-init | 168 regs, 768 B spill stores / 700 B spill loads, 124004 B smem | event median 0.4484 ms; dK/dV about 375-384 us | invalid: `dk`/`dv` non-finite | rejected |
| Existing exact BSHD TMEM dK/dV launcher under the new binding | 168 regs, 904 B spill stores / 816 B spill loads, 163940 B smem | event median 0.4627 ms; dK/dV about 379-384 us | invalid: `dk`/`dv` non-finite | rejected |

Final action: removed the hot-TMEM scaffold, wrapper, and pybind exports, then
rebuilt the safe extension. The old `bwd_cute16_kernel` dK/dV helper was not
re-exposed. The 2CTA route remains opt-in only.

## Restored-State Validation After Standalone Revert

Run after reverting the failed hot-TMEM route and rebuilding. One input set was
used for all routes: `B=1, S=2048, H=2, Dqk=192, Dv=128, causal=True`, seed
`20260709`. `HAS_HOT_TMEM_BINDING` was `False`.

| Route | Event median | Event min/max | dK/dV split after warmup | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL FA4 BF16 bwd | 0.204160 ms | 0.200512 / 0.226624 ms | n/a | finite |
| TK fallback | 0.513792 ms | 0.502176 / 0.525280 ms | about 415-427 us | finite |
| TK 2CTA experimental | 0.551360 ms | 0.545856 / 0.558464 ms | about 462-469 us | finite, slower |

Correctness vs CuTe DSL BF16 backward:

| Route | Tensor | rel L2 | max abs | bad values |
| --- | --- | ---: | ---: | ---: |
| TK fallback | `dq` | 1.025024 | 2.95648 | 0 |
| TK fallback | `dk` | 0.00286054 | 0.007846 | 0 |
| TK fallback | `dv` | 0.00167621 | 0.007564 | 0 |
| TK 2CTA experimental | `dq` | 1.026905 | 2.95648 | 0 |
| TK 2CTA experimental | `dk` | 0.00286054 | 0.007846 | 0 |
| TK 2CTA experimental | `dv` | 0.00167621 | 0.007564 | 0 |

Next bottleneck: a true CuTe-like dK/dV kernel still needs an exact causal
frontier/diagonal protocol that does not poison TMEM accumulators. The useful
performance target remains dK/dV below the fallback's roughly 415-423 us while
matching fallback-level `dk/dv` errors.

## Standalone Dense-TMEM Route: Accepted Opt-In Milestone

Run on 2026-07-10 with one shared input set, seed `20260710`, shape
`B=1, S=2048, H=2, Dqk=192, Dv=128`, `causal=True`. CuTe forward output and
LSE were supplied to all backward routes. The finite fallback remains the
default; this route is internal and opt-in only.

The standalone route uses a 2-CTA dense dK/dV main, exact causal frontier
sidecars, and the existing clustered dQ route. The key correctness fix was to
transpose the full Q-by-K probability/dS tile across the eight compute warps,
then pass true 64-column shared-tile references to WGMMA. TK's two-dimensional
`st_subtile` view is not a valid WGMMA operand; it caused dK chunk 0 to be
correct while chunks 1 and 2 silently reused the wrong descriptor base.

| Step | dK/dV time | `dk` rel L2 | `dv` rel L2 | Resource/result |
| --- | ---: | ---: | ---: | --- |
| register-hot dense math | 284-291 us | 0.903 | 0.490 | finite but wrong |
| exact dense math, no cross-warp transpose | 335-339 us | 0.961 | 0.672 | finite but wrong |
| explicit cross-warp transpose, invalid WGMMA subtiles | 368-372 us | 0.786 | 0.00167 | dV fixed; dK chunks 1/2 wrong |
| true WGMMA column subtiles, combined frontier | 358-364 us | 0.00288 | 0.00165 | correct; frontier still 255 regs, 940/1608 B spills |
| separate dK/dV frontier kernels | 380-384 us | 0.00292 | 0.00164 | correct; dK still 255 regs, 244/444 B spills |
| one dK chunk per concurrently scheduled block | 365-375 us | 0.00290 | 0.00167 | accepted opt-in milestone |

Final same-input event results for the accepted version:

| Route | Event median | Event min/max | dK/dV split | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL BF16 backward | 0.201856 ms | 0.191744 / 0.222816 ms | n/a | finite |
| TK fallback/default | 0.503168 ms | 0.492416 / 0.704736 ms | 416-424 us | finite |
| TK dense-TMEM opt-in | 0.449792 ms | 0.439680 / 0.460768 ms | 365-375 us | finite, faster than fallback |

Correctness against CuTe on the same inputs:

| Route | Tensor | rel L2 | max abs | bad values |
| --- | --- | ---: | ---: | ---: |
| TK fallback | `dq` | 1.02255 | 2.47185 | 0 |
| TK fallback | `dk` | 0.00289890 | 0.00847316 | 0 |
| TK fallback | `dv` | 0.00167192 | 0.00762296 | 0 |
| TK dense-TMEM opt-in | `dq` | 1.02259 | 2.47185 | 0 |
| TK dense-TMEM opt-in | `dk` | 0.00289892 | 0.00846720 | 0 |
| TK dense-TMEM opt-in | `dv` | 0.00167179 | 0.00761223 | 0 |

Accepted resource and profiler breakdown:

| Kernel | Time | Registers | Stack | Spill stores/loads | Shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense TMEM dK/dV main | 282.403 us | 168 | 112 B | 56 / 60 B | 156740 B |
| dK frontier, one 64-column chunk per block | 39.872 us | 228 | 0 | 0 / 0 | 0 |
| dV frontier | 36.224 us | 231 | 0 | 0 / 0 | 0 |
| clustered dQ, overlapped | 235.331 us | 168 | 16 B | 0 / 0 | 189508 B |

The next measured bottleneck is the 282 us dense main. The frontier is no
longer a 255-register/spill path. Closing the remaining gap requires replacing
the main's per-warp score/dP reconstruction with a more CuTe-like
warp-specialized tcgen05 score/dP pipeline, or otherwise reducing its replay
and synchronization cost. No route promotion was made.

### Two-Stage Q/dO Pipeline

The dense main was changed to double-buffer Q, dO, LSE, and dPsum. The load
warp now issues stage `n+1` while the compute warps consume stage `n`; the
existing block barrier guards shared-stage reuse. This keeps the exact math and
TMEM ownership unchanged.

| Metric | Before pipeline | Two-stage pipeline |
| --- | ---: | ---: |
| dense main profiler time | 282.403 us | 211.328 us |
| total dK/dV split | 365-375 us | 297.7-301.4 us |
| TK opt-in graph median | 0.449792 ms | 0.388448 ms |
| TK fallback graph median | 0.503168 ms | 0.510464 ms |
| CuTe graph median | 0.201856 ms | 0.201504 ms |
| dense main shared memory | 156740 B | 198724 B |
| dense main registers/spills | 168, 56/60 B | 168, 56/60 B |

The final two-stage correctness remained at fallback level: `dk` rel L2
`0.00288242`, `dv` rel L2 `0.00167915`, with no non-finite values. Profiler
times after the change were dense main `211.328 us`, dK frontier `40.288 us`,
dV frontier `36.096 us`, and overlapped dQ `235.776 us`. The next local
opportunity is to overlap the independent dK and dV frontier sidecars after
the dense main; after that, dQ becomes the critical path.

### Concurrent dK/dV Frontier Sidecars

The dK and dV frontier kernels depend on the dense main but not on each other.
The opt-in route now records a main-complete event, launches dK on the current
stream and dV on the cached auxiliary stream, then joins dV before reporting
dK/dV completion. The finite fallback and default routing are unchanged.

| Metric | Sequential sidecars | Concurrent sidecars |
| --- | ---: | ---: |
| dK/dV split | 297.7-301.4 us | 261.1-264.7 us |
| dQ split under contention | 260.7-264.2 us | 262.5-267.8 us |
| total split after warmup | 319.1-322.0 us | 279.7-285.7 us |
| opt-in graph median | 0.388448 ms | 0.346112 ms |
| fallback graph median | 0.510464 ms | 0.508992 ms |
| CuTe graph median | 0.201504 ms | 0.201216 ms |

The concurrent version remains finite and at fallback accuracy: `dk` rel L2
`0.00283625`, `dv` rel L2 `0.00162675`, with max absolute errors
`0.00819743` and `0.00772905`. dK/dV and dQ are now balanced around 263-265 us;
dQ replay is the next bottleneck.

### Frontier Scratch Overlap: Retained

The frontier kernels were moved ahead of the dense-main dependency. They now
write complete dK/dV frontier contributions to cached FP32 scratch on one
auxiliary stream while the dense main runs. A single 18-register vector kernel
adds both scratch tensors after the join. This remains internal and opt-in.

| Kernel/branch | Profiler time | Registers | Spills |
| --- | ---: | ---: | ---: |
| dense TMEM main | 211.040 us | 168 | 56 / 60 B |
| dK frontier to scratch | 38.848 us | 228 | 0 / 0 |
| dV frontier to scratch | 35.840 us | 231 | 0 / 0 |
| combined scratch add | 4.320 us | 18 | 0 / 0 |
| clustered dQ under overlap | 240.448 us | 168 | 0 / 0 |

The measured dK/dV branch dropped from about `261-268 us` to `225-230 us`.
The branch is now below both the fallback and dQ, and no 255-register kernel is
on this opt-in route.

### Rejected Follow-Ups

Each experiment below was built, measured on the same shape, rejected, and
removed before the final rebuild.

| Experiment | Timing/result | Correctness/resource reason |
| --- | --- | --- |
| two-stage clustered dQ | graph 0.367 ms; split 294-305 us | finite, but slower; dQ shared grew to 231492 B |
| 2-CTA launch attributes for dQ | graph 0.413 ms; dQ 261-272 us | finite but slower than Cluster1 |
| fused dense dQ, serialized chunk stores | graph 0.507 ms; dense branch 401-405 us | finite; dQ rel L2 0.807, but register MMA erased overlap gain |
| fused dense dQ, batched chunk stores | graph 0.506 ms; dense branch 407-412 us | still too slow despite 168 regs and 52/56 B spills |
| warpgroup score/dP, Q/K order plus transpose | dK/dV 168-170 us | fast but wrong: dK 0.669, dV 0.466 rel L2 |
| warpgroup score/dP, K/Q order plus transpose | dK/dV 165-168 us | wrong: dK 0.973, dV 0.685 rel L2 |
| warpgroup score/dP, native direct TMEM orientation | dK/dV 146-151 us | wrong: dK 0.704, dV 0.490 rel L2 |
| native orientation plus K/V warpgroup restaging | about 155 us | dV became non-finite |

The fast score-path probes establish that tcgen05 throughput is available, but
the TK shared/TMEM tile interpretation does not yet match the direct CuTe dK/dV
orientation. It must be validated with a small tile-level layout probe before
being reintroduced. The failed fused-dQ probes also show that register MMA is
not a viable substitute for CuTe's TMEM dQ accumulation and postprocess.

## Final Same-Input Matrix (2026-07-10)

Shape `B=1, S=2048, H=2, Dqk=192, Dv=128`, causal, seed `20260710`.
All routes used the same Q/K/V/dO and CuTe forward output/LSE.

| Route | Event median | Event min | dK/dV split | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL BF16 backward | 0.216896 ms | 0.202272 ms | n/a | finite |
| TK fallback/default | 0.504704 ms | 0.502080 ms | 416-421 us | finite |
| TK retained 2CTA | 0.554976 ms | 0.544352 ms | 458-463 us | finite, slower |
| TK standalone scratch-overlap | 0.351328 ms | 0.342624 ms | 225-230 us | finite, opt-in |
| TK candidate2 | 0.229888 ms | 0.228832 ms | n/a | invalid: non-finite dK |

| Route | Tensor | rel L2 | max abs | Finite |
| --- | --- | ---: | ---: | --- |
| fallback | dQ | 1.049102 | 2.937726 | yes |
| fallback | dK | 0.00289359 | 0.00741386 | yes |
| fallback | dV | 0.00166952 | 0.01352787 | yes |
| retained 2CTA | dQ | 1.049102 | 2.937726 | yes |
| retained 2CTA | dK | 0.00289359 | 0.00741386 | yes |
| retained 2CTA | dV | 0.00166952 | 0.01352787 | yes |
| standalone scratch-overlap | dQ | 1.049102 | 2.937726 | yes |
| standalone scratch-overlap | dK | 0.00289356 | 0.00742102 | yes |
| standalone scratch-overlap | dV | 0.00166946 | 0.01355028 | yes |
| candidate2 | dK | NaN | NaN | no |

The finite fallback remains the routed default. The retained standalone route
meets the dK/dV milestone and reduces wrapper event time by about 30% versus
fallback, but it remains about 1.62x CuTe in this run. The next critical branch
is clustered dQ replay at roughly `251-257 us` under contention. Matching CuTe
requires TMEM dQ accumulation in the main schedule plus a small postprocess;
the rejected register-MMA fusion should not be revisited.

## TMEM dQ Follow-Up (2026-07-11)

This pass profiled the retained standalone route against CuTe, replaced the
clustered dQ kernel's register-MMA output stage with an opt-in TMEM output
stage, and moved that opt-in dQ launch ahead of the dense/frontier dK/dV host
launches. The finite fallback and the prior standalone route remain unchanged;
the new path is exposed only as:

`b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_chunked_tmem_dq_internal`

### Matched dQ Profile

Bounded torch-profiler and NCU runs used seed `20260710` and
`B=1, S=2048, H=2, Dqk=192, Dv=128`, causal.

| Path/kernel | Time | Grid/block | Cluster | Registers | Shared memory | ptxas spills | Achieved occupancy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CuTe preprocess | 6.848 us | 32 blocks | 1 | 80 | small | none reported | not sampled |
| CuTe `FlashAttentionBackwardSm100` | 97.058 us profiler, 99.07 us NCU | `(16,2,1)` / 512 | 2 | 128 | 223.23 KiB dynamic plus 1.02 KiB driver | no static spill metadata; 1088 NCU local requests | 23.03% |
| CuTe dQ postprocess | 9.760 us profiler, 13.38 us NCU | `(16,2,1)` / 128 | 1 | 168 | 98.30 KiB dynamic plus 1.02 KiB driver | none reported | 6.25% |
| TK retained clustered dQ | 237.9-246.8 us | `(16,2,1)` / 288 | 1 | 168 | 189.51 KiB static | 0 / 0 B | 13.74% |
| TK chunked-TMEM dQ | 238.17 us profiler, 240.93 us NCU | `(16,2,1)` / 288 | 1 | 168 | 189.52 KiB static | 24 / 44 B | 13.86% |

The retained TK dQ kernel is not bandwidth limited. NCU reported about 3.6%
SM throughput, 34.4 GB/s memory throughput, and one block per SM due to both
registers and shared memory. CuTe reaches about 23% achieved occupancy with 16
warps per CTA, versus TK's 9 warps and 13.7-13.9%. CuTe concurrently drains
32-column dQ fragments from TMEM with four reducer warps; TK had one consumer
warpgroup reconstruct full 192-column dQ in registers after the score/dP
replay.

The retained optimization keeps the validated q-major score/dP orientation.
Each of the two consumer warpgroups now issues `dQ = dS @ K` into a 64x192
TMEM tile. Consumer 0 waits on explicit completion barriers, loads both TMEM
outputs in 64-column slices, sums them, and sends three FP32 TMA reduce-adds.
This avoids the old full-width register MMA and improves dQ agreement with
CuTe, while preserving causal ownership.

### Launch and Overlap

The original standalone host order enqueued the dense main and both frontier
kernels before dQ. A matched profiler trace placed dQ at about 102-336 us,
after dense dK/dV had already started at about 59 us. Since dQ is now the
critical branch, the new opt-in order enqueues dQ immediately after its
preprocess dependency, then enqueues dK/dV.

Final profiler timeline, relative device time:

| Operation | Start | End | Duration |
| --- | ---: | ---: | ---: |
| preprocess | 0.000 us | 3.776 us | 3.776 us |
| dQ memset | 29.663 us | 32.383 us | 2.720 us |
| chunked-TMEM dQ | 65.951 us | 304.122 us | 238.171 us |
| dense TMEM dK/dV | 83.870 us | 294.298 us | 210.428 us |
| dK frontier | 96.606 us | 144.349 us | 47.743 us |
| dV frontier | 146.397 us | 181.884 us | 35.487 us |
| frontier scratch add | 298.009 us | 302.266 us | 4.257 us |

The two branches now finish within about 2 us in this profiler run. With only
split timing enabled, warmed opt-in calls reported dK/dV `226-230 us`, dQ
kernel `242-247 us`, and total split `271-282 us`. Enabling the nested
clustered-dQ timer synchronizes dQ before dK/dV is enqueued, so those serialized
totals are not valid overlap measurements.

### Final Same-Input Matrix

This final bounded run used 8 warmups and 31 event-timed iterations per route.
Clock/load variation moved every absolute result relative to the 2026-07-10
matrix, so comparisons below are only within this row set.

| Route | Event median | Event min | Event max | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL BF16 backward | 0.203392 ms | 0.193856 ms | 0.254816 ms | finite |
| TK fallback/default | 0.436928 ms | 0.432672 ms | 0.456960 ms | finite, unchanged default |
| TK standalone scratch-overlap | 0.278272 ms | 0.275104 ms | 0.291680 ms | finite, prior opt-in |
| TK chunked-TMEM dQ plus early enqueue | 0.264800 ms | 0.262144 ms | 0.276544 ms | finite, new opt-in |

The new path is 4.8% faster than the retained standalone route in this final
run and about 1.30x CuTe, down from the supervisor's retained 0.3513 ms versus
0.2169 ms comparison. It is not promoted to the default route.

| Route | Tensor | rel L2 vs CuTe | max abs vs CuTe | Finite |
| --- | --- | ---: | ---: | --- |
| fallback | dQ | 1.049102 | 2.937726 | yes |
| fallback | dK | 0.00289359 | 0.00741386 | yes |
| fallback | dV | 0.00166952 | 0.01352787 | yes |
| standalone scratch-overlap | dQ | 1.049102 | 2.937726 | yes |
| standalone scratch-overlap | dK | 0.00289356 | 0.00742102 | yes |
| standalone scratch-overlap | dV | 0.00166946 | 0.01355028 | yes |
| chunked-TMEM dQ plus early enqueue | dQ | 0.865916 | 1.105581 | yes |
| chunked-TMEM dQ plus early enqueue | dK | 0.00289356 | 0.00742102 | yes |
| chunked-TMEM dQ plus early enqueue | dV | 0.00166946 | 0.01355028 | yes |

### Attempts in This Pass

Every failed variant was removed and followed by a safe rebuild before the
next attempt.

| Attempt | Result | Decision |
| --- | --- | --- |
| full-width TMEM load and register reduction | finite; graph 0.313792 ms vs retained 0.269952 ms; dQ main 286-288 us; 764/792 B spills; dQ rel L2 0.843723 | rejected and removed |
| 64-column TMEM load/reduction | finite; graph 0.266784 ms vs retained 0.271648 ms before launch reorder; 24/44 B spills; dQ rel L2 0.868398 | retained |
| separate TMA add for each consumer | finite; graph 0.299872 ms vs retained 0.271808 ms; 72/140 B spills; dQ rel L2 0.868654 | rejected and removed |
| consumer-1 TMEM-side MMA accumulation | finite; graph 0.261824 ms, slower than passing 0.259168 ms run; dQ rel L2 regressed to 1.617755 | rejected and removed |
| enqueue dQ before dense/frontier dK/dV | finite; graph 0.259168 ms vs retained 0.270656 ms in the direct A/B, final matrix 0.264800 vs 0.278272 ms | retained for new opt-in only |

### Next Bottleneck

The host overlap is now balanced, and serial reducer variations have exhausted
the small local gains. The remaining structural gap is the dQ replay itself:
TK still uses a separate 9-warp, one-block-per-SM kernel with 168 registers and
about 190.5 KiB static shared memory. CuTe fuses dQ production into its
512-thread 2-CTA main and overlaps TMEM draining through dedicated reducer
warps before a 10 us postprocess.

The next implementation should add dedicated reducer warps and a two-stage dQ
TMEM/full-empty semaphore pipeline, with double-buffered Q/dO or integration
into the dense dK/dV main. It must let compute advance while the prior dQ tile
is drained. More serial TMA splitting, register-MMA fusion, 2-CTA launch-attribute
changes, and score-orientation probes should not be repeated.

## Dedicated dQ Reducer Pipeline (2026-07-11)

The next opt-in route adds four aligned reducer warps to the standalone dQ
replay. Compute warps 0-7 produce alternating 64x192 TMEM outputs, reducer
warps 8-11 drain 64-column fragments, and warp 12 owns Q/dO and statistics
loads. Explicit full/empty mbarriers replace both per-q-block block-wide
barriers. The route is exposed only as:

`b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_internal`

The published chunked-TMEM route and fallback/default route are unchanged.

### Measured Steps

All rows used seed `20260710`, `B=1, S=2048, H=2, Dqk=192, Dv=128`, causal,
and the same CuTe forward output/LSE within each comparison.

| Attempt | Graph/event result | dQ/resource result | Decision |
| --- | ---: | --- | --- |
| dedicated reducers, two TMEM stages | 0.250400 ms matched median vs chunked 0.259136 ms and CuTe 0.187104 ms | dQ 232-236 us; REG 128, SHARED 157844 B, STACK 16 B | retained as basis |
| overlap score and dP tcgen05 issue | 0.251552 ms vs prior 0.250400 ms | finite with unchanged errors | rejected and removed |
| two shared dS stages | 0.246624 ms short matched median | moves producer wait from every iteration to two-iteration stage reuse | retained |

The two-dS stage lets the next score/dP replay proceed while the prior dQ MMA
consumes its other shared stage. Its final static resource record is REG 128,
SHARED 174228 B, STACK 24 B, with no 255-register kernel on the route. Warmed
split timing reported dK/dV `225-228 us` and dQ kernel `237-240 us`.

### Final Same-Input Matrix

This bounded validation used 8 warmups and 31 event-timed calls per route.

| Route | Event median | Event min | Event max | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL BF16 backward | 0.190112 ms | 0.184128 ms | 0.218592 ms | finite |
| TK fallback/default | 0.430816 ms | 0.426144 ms | 0.444800 ms | finite, unchanged default |
| TK chunked-TMEM dQ | 0.260096 ms | 0.257536 ms | 0.270784 ms | finite, published opt-in |
| TK pipelined reducer plus two dS stages | 0.254720 ms | 0.251360 ms | 0.260608 ms | finite, new opt-in |

| Tensor | rel L2 vs CuTe | max abs vs CuTe | Finite |
| --- | ---: | ---: | --- |
| dQ | 0.868661 | 1.105581 | yes |
| dK | 0.00289356 | 0.00742102 | yes |
| dV | 0.00166946 | 0.01355028 | yes |

The dedicated reducers remove about `5.4 us` from the final matched graph and
cut the dQ kernel's register count from 168 to 128, but the graph remains about
1.34x CuTe. The remaining structural cost is the separate score/dP replay.
The next route should issue TMEM dQ chunks from the already-produced dense-main
dS, with a small frontier-only dQ kernel for the two causal boundary blocks.

## Rejected Dense-Main dQ Integration (2026-07-11)

A standalone opt-in probe reused the dense dK/dV main's validated q-major dS.
It issued three 64-column dQ tcgen05 outputs into TMEM columns 0, 64, and 448,
while a frontier-only instance of the pipelined dQ kernel covered the two causal
boundary q-blocks. This established a correct ownership decomposition:

| Tensor | rel L2 vs CuTe | max abs vs CuTe | Finite |
| --- | ---: | ---: | --- |
| dQ | 0.552531 | 0.988281 | yes |
| dK | 0.00289356 | 0.00742102 | yes |
| dV | 0.00166946 | 0.01355028 | yes |

dQ agreement improved substantially relative to the retained standalone replay,
and dK/dV were unchanged. Performance did not pass the retained-route gate:

| Dense role schedule | Resource/result | Graph median |
| --- | --- | ---: |
| 13 warps, dedicated load | REG 128, SHARED 215204 B, STACK 440 B, 368/400 B spills | 0.291872 ms |
| 16 warps with 168/96/24 register requests | ptxas kept REG 128 and ignored part of setmaxnreg; 348/364 B spills | 0.299648 ms |
| 12 warps, reducer warp also loads | REG 168, STACK 136 B, 60/64 B spills | 0.411712 ms |
| 12 warps, compute warp also loads | REG 168, SHARED 216228 B, STACK 144 B, 72/104 B spills | 0.358848 ms |

For the first schedule, split timing isolated dense dK/dV+dQ at `268-271 us`
and the frontier-only dQ kernel at `33-34 us`; the retained standalone graph was
about `0.2560 ms` in the same run and CuTe was `0.1889 ms`. The role-merging
variants recovered the original dense kernel's register count but serialized
either the reducer warpgroup or a compute warpgroup around input loading.

All fused-main source and binding changes were removed after these failures.
The finite `2b460b7` pipelined standalone route remains the retained opt-in
checkpoint and the fallback remains the default.

## Double-Buffered dQ Inputs (2026-07-11)

The standalone dQ replay still serialized each q-block's Q, dO, and statistics
load behind release of the prior block. This pass added two input stages with
independent Q/dO TMA barriers, statistics-ready barriers, and consumer-release
barriers. The existing two-stage dS/TMEM reducer pipeline is unchanged. The
winning route is opt-in only through:

`b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_dkdv_first_internal`

The dK/dV branch is enqueued before the double-buffered dQ branch. A matched
15-iteration A/B measured `0.240928 ms` for this order, `0.246240 ms` when dQ
was enqueued first, and `0.255616 ms` for the published single-input-stage
route. The slower dQ-first double-buffer binding was removed before the final
build.

### Resources and Split Timing

| dQ route | Registers | Shared memory | Stack | ptxas spill stores/loads | Warm dQ kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| single input stage | 128 | 173204 B | 24 B | 4 / 4 B | 237-240 us before this pass |
| double input stage | 128 | 215188 B | 32 B | 20 / 28 B | 160-168 us under dK/dV-first contention |

With split instrumentation enabled, warmed calls reported dK/dV `224.5-229.1
us`, dQ total `173.5-184.0 us`, and total `246.98-257.54 us`. The overlap is
now bounded by the dense dK/dV branch rather than standalone dQ replay.

### Final Same-Input Matrix

The final bounded run used seed `20260710`, identical CuTe forward output/LSE,
8 warmups, and 31 event-timed calls per route at
`B=1, S=2048, H=2, Dqk=192, Dv=128`, causal.

| Route | Event median | Event min | Event max | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL BF16 backward | 0.192576 ms | 0.186016 ms | 0.259424 ms | finite |
| TK fallback/default | 0.431712 ms | 0.427936 ms | 0.455296 ms | finite, unchanged default |
| TK single-input-stage pipelined dQ | 0.254688 ms | 0.253120 ms | 0.260544 ms | finite, published opt-in |
| TK double-input-stage dQ, dK/dV first | 0.240256 ms | 0.237184 ms | 0.249664 ms | finite, new opt-in |

| Route | Tensor | rel L2 vs CuTe | max abs vs CuTe | Finite |
| --- | --- | ---: | ---: | --- |
| fallback | dQ | 1.049102 | 2.937726 | yes |
| fallback | dK | 0.00289359 | 0.00741386 | yes |
| fallback | dV | 0.00166952 | 0.01352787 | yes |
| double-input-stage | dQ | 0.857361 | 1.099335 | yes |
| double-input-stage | dK | 0.00289356 | 0.00742102 | yes |
| double-input-stage | dV | 0.00166946 | 0.01355028 | yes |

This checkpoint is 5.7% faster than the prior pipelined route in the final
matrix and remains about 1.25x CuTe. The next measured bottleneck is the dense
dK/dV main kernel (`REG 168`, `198724 B` shared, `56/60 B` spills), whose
roughly `225 us` branch time determines the graph. The fallback remains the
default.

### Published Checkpoints

| Commit | Push result | Contents |
| --- | --- | --- |
| `351a6d4` | pushed to `origin/tk-fa4-sm100-rewrite` | finite standalone dK/dV overlap checkpoint |
| `2b460b7` | pushed to `origin/tk-fa4-sm100-rewrite` | dedicated reducer/two-stage TMEM dQ pipeline |
| `a6bad14` | pushed to `origin/tk-fa4-sm100-rewrite` | rejected fused-main schedule measurements |
| `9ef88c8` | pushed to `origin/tk-fa4-sm100-rewrite` | double-buffered standalone dQ input pipeline |

## Split-2 Dense dK/dV Checkpoint (2026-07-11)

After the double-buffered dQ checkpoint, the dense dK/dV branch was the graph
limit at roughly `225-229 us`. Several local schedule probes did not change
that conclusion. Every rejected source change below was removed and followed
by a safe extension rebuild.

| Attempt | Result | Decision |
| --- | --- | --- |
| remove dead `dp_block_t` state | compiler/resource no-op; graph about `0.244 ms` | removed |
| explicit dense input semaphore pipeline | finite; `0.245952 ms` vs checkpoint `0.241664 ms`; dK/dV `229-233 us` | removed |
| remove named barrier 10 | finite; `0.245984 ms` | removed |
| per-warpgroup transpose barrier | non-finite dV | removed |
| q-major tcgen05 score/dP replacement | `0.192256 ms`, but dK rel L2 `0.6685` and dV rel L2 `0.4659` | incorrect orientation mixing; removed |
| direct fragment store | finite; `0.242592 ms`, unchanged resources | removed |
| dedicated MMA warp with full/empty pipeline | `0.2536 ms`; REG 168, STACK 88 B, 52/56 B spills; non-finite dK/dV | removed |
| direct K-owned transpose | `0.2540 ms`; dK exact, dV rel L2 `0.01054`; 164/160 B spills | outside error gate; removed |
| disable loop unrolling for register budget | finite but `0.32432 ms` | removed |

The retained change splits each dense q-block sequence by parity across two
otherwise identical clustered CTA groups. Split 0 writes the primary FP32
dK/dV outputs, split 1 writes cached FP32 scratch, and the existing frontier
scratch is preserved. A small vector kernel sums primary, dense-split, and
frontier values after both branches complete. The default and published
single-split routes are unchanged.

The first dK/dV-first version was finite and reduced the warmed dK/dV branch
from `224-229 us` to `126-134 us`, but its separate dQ replay became critical
at `179.5-190.2 us`. Its matched 15-iteration median was `0.199232 ms` versus
CuTe `0.192000 ms` and the published checkpoint `0.240224 ms`. Enqueuing dQ
before split dK/dV exposed enough independent work to hide the shorter dense
branch. That order is the sole retained split-2 binding:

`b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_internal`

### Resource Record

| Kernel | Registers | Shared memory | Stack | ptxas spill stores/loads |
| --- | ---: | ---: | ---: | ---: |
| original dense dK/dV | 168 | 198724 B static | 112 B | 56 / 60 B |
| split-2 dense dK/dV | 168 | 198724 B static | 64 B | 20 / 20 B |
| split/frontier add | 24 | 0 B | 0 B | 0 / 0 B |
| double-buffered dQ replay | 128 | 215188 B static | 32 B | 20 / 28 B |

`cuobjdump` reports 199748 B shared for each dense kernel after the 1024 B
driver allocation. No kernel on the retained route uses 255 registers.

### Final Same-Input Matrix

This validation used seed `20260710`, one CuTe forward output/LSE and one input
set, 8 warmups, and 31 CUDA-event iterations per route at
`B=1, S=2048, H=2, Dqk=192, Dv=128`, causal.

| Route | Event median | Event min | Event max | Status |
| --- | ---: | ---: | ---: | --- |
| CuTe DSL BF16 backward | 0.190816 ms | 0.185888 ms | 0.758848 ms | finite; one timing outlier |
| TK fallback/default | 0.434400 ms | 0.429920 ms | 0.448288 ms | finite, unchanged default |
| TK published double-buffer checkpoint | 0.243648 ms | 0.240480 ms | 0.257088 ms | finite |
| TK split-2, dK/dV first | 0.205408 ms | 0.201344 ms | 0.445088 ms | finite; rejected ordering |
| TK split-2, dQ first | 0.169024 ms | 0.166304 ms | 0.183904 ms | finite, retained opt-in |

| Tensor | rel L2 vs CuTe | max abs vs CuTe | Finite |
| --- | ---: | ---: | --- |
| dQ | 0.859581 | 1.105581 | yes |
| dK | 0.00289356 | 0.00742078 | yes |
| dV | 0.00166945 | 0.01355028 | yes |

With split instrumentation enabled, warmed dQ-first calls measured dK/dV at
`126.2-129.9 us`, dQ total at `157.8-163.9 us`, and the dQ kernel itself at
`148.2-150.3 us`. Total instrumented calls were `167.9-173.4 us`. The dQ
branch is again the limit, but the retained opt-in graph is 11.4% faster than
CuTe's median in the final same-input matrix. The requested target is reached
without changing the fallback default.

After removing the losing split-2 ordering binding and rebuilding the final
extension, a second 8-warmup/31-iteration check measured CuTe at `0.191360 ms`
median and the retained route at `0.171392 ms`. dQ/dK/dV remained finite; dK
and dV relative L2 were `0.00289356` and `0.00166945`.

## Broad-Sweep Follow-Up

The retained split-2 route was subsequently validated for sequence lengths
divisible by 512 from 512 through 8192. See
[the 2026-07-11 broad sweep](tk_bf16_bwd_broad_sweep_20260711.md) for the full
19-shape timing/error matrix, aggregate win rates, loss profiles, and rejected
schedule-order probes.

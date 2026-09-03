# TK BF16 FA4 Backward S=2048 H=2 Probe

Date: 2026-07-09

Scope: BF16 FA4 backward only. No forward code or MXFP4 code was changed.

## Baseline Reproduction

Reference artifact: `results/tk_bf16_bwd_vs_cute_dsl_bf16_s2048h2_20260709T172738Z.json`

Shape: `B=1, S=2048, H=2, Dqk=192, Dv=128`, causal.

Short bounded reproduction with graph replay:

| Path | Graph event time |
| --- | ---: |
| TK hot routed candidate | 0.426752 ms, 0.423360 ms |
| CuTe DSL BF16 backward | 0.122368 ms, 0.119936 ms |

This keeps TK at about 3.5x slower than CuTe DSL for the hot graph path.

After reverting the exact-ownership experiment and rebuilding, a final short restore check produced:

| Path | Graph event time |
| --- | ---: |
| TK hot routed candidate | 0.429280 ms, 0.424832 ms |
| CuTe DSL BF16 backward | 0.124672 ms, 0.120352 ms |

Final restored graph ratio: about 3.44x slower than CuTe DSL.

## Variant Status

Same-shape variant smoke:

| Variant | Event / graph time | Correctness status |
| --- | ---: | --- |
| `candidate` | about 0.420 ms graph | finite |
| `candidate_bf16_dkdv` | about 0.425 ms graph | finite, not faster |
| `trusted` | about 0.423 ms graph | finite, same speed class |
| `candidate2` | about 0.208 ms graph | not viable: DK is non-finite; DV also has large error |

Same-forward comparison against CuTe DSL backward showed the meaningful finite gradients for `candidate` are close on `dk` and `dv` but not on `dq`:

| Variant | dq rel_l2 | dk rel_l2 | dv rel_l2 | Notes |
| --- | ---: | ---: | ---: | --- |
| `candidate` | about 1.02 | about 0.002888 | about 0.001654 | finite |
| `candidate_bf16_dkdv` | high dq | about 0.003209 | near 0 in that run | finite |
| `trusted` | similar to candidate | similar to candidate | similar to candidate | finite |
| `candidate2` | non-finite risk | non-finite | about 0.811 | not viable |

After restoring the source and rebuilding, a bounded direct `candidate2` smoke completed quickly and reproduced the original failure:

```text
iter 0 finite True False True
iter 1 finite False False True
```

## Timing Attribution

`torch.profiler` and the local split timing instrumentation agree that wrapper/preprocess/scratch overhead is not the main 0.420 ms vs 0.120 ms gap.

Steady split timings for routed TK `candidate`:

| Component | Time |
| --- | ---: |
| preprocess | about 15 us |
| dK/dV kernel | about 421 us |
| dQ zero | about 9-11 us |
| dQ kernel | about 245-266 us |
| split-timing total | about 446-451 us |

The graph wall time is lower, about 0.420 ms, because dQ overlaps with dK/dV on the split stream. The critical path is still the dK/dV kernel.

Profiler kernel-level comparison:

| Path | Main work |
| --- | ---: |
| TK `candidate` dK/dV kernel | about 408.6 us |
| TK `candidate` dQ kernel | about 233.9 us, overlapped |
| TK `candidate2` main kernel | about 151.9 us, plus patch kernels, but incorrect |
| CuTe DSL main kernel | about 96.8 us, plus about 6.7 us preprocess and 9.2 us postprocess |

The routed TK path uses cached split resources. In graph replay there is no evidence that scratch allocation or preprocessing dominates. Launch count and split stream/event overhead are secondary; the stable dK/dV kernel is the bottleneck.

## Code Change Attempt

I attempted the smallest plausible `candidate2` correctness fix: copy the base kernel's causal diagonal skip for consumer 1 on the first causal q iteration into `b300_bwd_cute16_kernel_candidate2.cuh`.

Result:

- With the copied `group<4>::sync(...)`, the short candidate2 comparison hung until the outer timeout.
- Removing that sync still hung in a candidate2-only smoke.
- The patch was reverted and the extension was rebuilt from the restored source.

No unsafe source change was retained.

I also tested the existing S=2048 exact-ownership flag in `b300_bwd_cute16_candidate.cuh`:

- Changed only `kUseSeq2048ExactOwnership` from `false` to `true`.
- Rebuilt `_C`.
- Normal event comparison was finite and improved `dq` relative error to about `0.002923`, but graph replay slowed to `0.517408 ms` versus CuTe DSL `0.122304 ms`.
- Split timing showed the reason: steady dK/dV was about `459-461 us`, and the replacement dQ path was about `501-503 us`.
- This was reverted and `_C` was rebuilt back to the faster baseline route.

## Next Bottleneck

The competitive path is not removing `candidate`/`trusted` Python wrapper or scratch overhead. The stable TK path needs a faster dK/dV implementation.

The immediate options are:

1. Fix `candidate2` by understanding its internal causal diagonal semaphore / score-ready / dP-ready protocol. Its single-main-kernel shape is directionally closer to CuTe DSL speed, but the naive base-kernel skip deadlocks and its current DK/DV outputs are wrong.
2. Tune or replace the current `candidate` dK/dV kernel. Build output indicates very high register pressure and spills on the dK/dV kernel, so reducing register pressure and memory traffic should be the first profiling target.
3. Only after dK/dV is below the overlapped dQ time should dQ or split synchronization become the next critical-path target.

## Continuation: Candidate2 After 2CTA dK/dV Scaffold

Scope remained BF16 backward only. No forward or MXFP4 code was touched.

The new opt-in 2CTA dK/dV route is documented separately in
`results/tk_bf16_bwd_2cta_dkdv_candidate_s2048h2_20260709.md`. Supervisor smoke
confirmed it is finite but not a performance win:

| Path | Result |
| --- | ---: |
| CuTe DSL BF16 backward | about 0.174 ms |
| TK new 2CTA total | about 0.47-0.49 ms |
| TK new 2CTA dK/dV | about 455-465 us |
| TK new 2CTA dQ | about 249-256 us |
| TK fallback dK/dV | about 410 us |

The 2CTA route remains experimental only and is not the default.

### Candidate2 Probe A: Compute-Warp dK Store

I tested whether candidate2's non-finite `dk` came from the relay/empty-warp dK
TMEM readback path:

- changed candidate2 locally so compute warpgroups loaded `dk0/dk1/dk2` from
  their own TMEM accumulators and stored `dk` directly.
- left the existing dV relay path unchanged.
- rebuilt successfully.

Resource result for the modified candidate2 main:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate2 direct dK-store main, noncausal/causal | 128 | 56 B | 165100 B | 4 B | 8 B |

Validation result at `B=1,S=2048,H=2,Dqk=192,Dv=128,causal=True`:

| Path | Tensor | finite | rel L2 vs CuTe | max abs | Notes |
| --- | --- | --- | ---: | ---: | --- |
| candidate2 direct dK-store | dq | yes | 1.12615 | 1.83019 | still not usable |
| candidate2 direct dK-store | dk | no | n/a | n/a | 9436 non-finite values; max finite abs near `3.402e38` |
| candidate2 direct dK-store | dv | yes | 0.822619 | 1.34059 | large error |

Timing with preallocated float32 outputs:

| Path | Event median | Graph median | Correctness |
| --- | ---: | ---: | --- |
| fallback | 0.429120 ms | 0.419696 ms | finite |
| 2CTA experimental | 0.473952 ms | 0.463360 ms | finite |
| candidate2 direct dK-store | 0.230368 ms | 0.207792 ms | `dk` non-finite |

Conclusion: the dK relay store is not the primary source of non-finite values.
The modified candidate2 code was reverted.

### Candidate2 Probe B: Causal Diagonal Skip

The likely source of non-finite `dk` is candidate2's first causal diagonal block:
for consumer 1 it runs dense `compute_dkdv_loop<false,...>` at
`q_block_idx == kv_tile_base + 1`. That computes masked-future columns against a
causal LSE, which can overflow. The finite base path skips this block and lets
the causal patch kernels repair it.

I tested the direct skip:

- restored candidate2's original relay dK store.
- added a consumer-1 diagonal skip with zeroed `ds_warp_smem`, mirroring the
  base kernel structure.
- rebuilt successfully; candidate2 returned to the original low-register shape:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| restored/skip candidate2 main, noncausal | 120 | 48 B | 165124 B | 0 B | 0 B |
| restored/skip candidate2 main, causal | 127 | 48 B | 165124 B | 0 B | 0 B |

Runtime validation timed out under a 240 s outer bound before the first
correctness line, matching earlier skip experiments. This skip likely leaves
one of candidate2's score/dP semaphore phases or the pipelined q/do handoff in
an inconsistent state. The skip patch was reverted.

### Candidate2 Probe C: Masked Diagonal With dK/dV Disabled

I then tested a variant intended to keep candidate2's semaphore pipeline moving:
for the consumer-1 diagonal block only, call `compute_dkdv_loop<true,false,...>`.
That applies the causal mask and writes the normal `ds_warp_smem` side effect,
but disables dK/dV accumulation so the existing patch kernels remain responsible
for the diagonal contribution.

Build/resource result:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate2 masked-diagonal main, noncausal | 120 | 48 B | 165124 B | 0 B | 0 B |
| candidate2 masked-diagonal main, causal | 128 | 56 B | 165124 B | 4 B | 8 B |

Validation result:

| Path | Tensor | finite | rel L2 vs CuTe | max abs | Notes |
| --- | --- | --- | ---: | ---: | --- |
| candidate2 masked diagonal | dq | yes | 1.09203 | 1.34427 | still high |
| candidate2 masked diagonal | dk | no | n/a | n/a | 41034 non-finite values |
| candidate2 masked diagonal | dv | no | n/a | n/a | 16078 non-finite values |

Timing stayed fast but invalid:

| Path | Event median | Graph median | Correctness |
| --- | ---: | ---: | --- |
| candidate2 masked diagonal | 0.236448 ms | 0.218784 ms | `dk` and `dv` non-finite |

This patch was reverted and `_C` was rebuilt.

Final restored smoke after rebuilding all failed candidate2 probes:

| Path | dq finite | dk finite | dv finite | Non-finite counts |
| --- | --- | --- | --- | --- |
| fallback | yes | yes | yes | `[0, 0, 0]` |
| 2CTA experimental | yes | yes | yes | `[0, 0, 0]` |
| candidate2 restored | yes | no | yes | `[0, 9606, 0]` |

There is no remaining source diff in `tk_fa4/b300_bwd_cute16_kernel_candidate2.cuh`.

## Continuation: Restored Candidate2 and Failed Local Fixes

Scope remained BF16 backward only. No forward or MXFP4 code was edited.

After the 2CTA dK/dV scaffold proved finite but slower, I restored candidate2
to its original low-register source and rechecked the same `B=1,S=2048,H=2,
Dqk=192,Dv=128,causal=True` shape against CuTe BF16.

One-process restored timing/error run:

| Path | Event median | dq rel L2 / max abs | dk rel L2 / max abs | dv rel L2 / max abs | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| CuTe BF16 | 0.190272 ms | 0 / 0 | 0 / 0 | 0 / 0 | finite |
| TK fallback | 0.432736 ms | 1.02687 / 2.95648 | 0.0028605 / 0.007846 | 0.0016762 / 0.007564 | finite |
| TK 2CTA experimental | 0.470592 ms | 1.02691 / 2.95648 | 0.0028605 / 0.007846 | 0.0016762 / 0.007564 | finite, slower |
| candidate2 restored | 0.226240 ms | 1.12615 / 1.83019 | n/a | 0.822619 / 1.34059 | `dk` non-finite, 11262 bad values |

Restored ptxas shape:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate2 restored main, noncausal | 120 | 48 B | 165124 B | 0 B | 0 B |
| candidate2 restored main, causal | 127 | 48 B | 165124 B | 0 B | 0 B |

### Probe D: Disable Candidate2 Causal Patch Launches

I disabled candidate2's causal patch launches to isolate whether the main kernel
or the patch kernels first introduce the non-finite `dk`.

Result:

| Tensor | finite | rel L2 vs CuTe | max abs | Notes |
| --- | --- | ---: | ---: | --- |
| dq | yes | 58.6328 | 321.746 | invalid |
| dk | no | n/a | n/a | 432 non-finite values, max finite abs about `3.40189e38` |
| dv | yes | 5.23491 | n/a | invalid |

Event median: `0.240736 ms`.

Conclusion: the candidate2 main kernel already produces bad `dk`; the patch
kernels are not the first source of corruption.

### Probe E: No Patches Plus Masked Diagonal Accumulation

I kept patch launches disabled and changed only consumer 1's first diagonal block
to call `compute_dkdv_loop<true,true,...>` so it applies the causal mask while
still preserving the semaphore side effects.

Resource result:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate2 no-patch masked diagonal, causal | 128 | 56 B | 165124 B | 4 B | 8 B |

Validation:

| Tensor | finite | rel L2 vs CuTe | Notes |
| --- | --- | ---: | --- |
| dq | yes | 29.2331 | invalid |
| dk | no | n/a | 772 non-finite values |
| dv | yes | 5.10182 | invalid |

Event median: `0.238656 ms`.

Conclusion: diagonal masking reduces one failure mode but does not fix the main
dK accumulation path.

### Probe F: No Patches Plus Direct Compute-Warp dK Store

I bypassed candidate2's relay dK readback and had compute warpgroups load/store
their own dK TMEM accumulators directly.

Resource result:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate2 no-patch direct dK store, noncausal/causal | 128 | 56 B | 165124 B | 4 B | 8 B |

Validation:

| Tensor | finite | rel L2 vs CuTe | Notes |
| --- | --- | ---: | --- |
| dq | yes | 74.4986 | invalid |
| dk | no | n/a | 419 non-finite values |
| dv | yes | 4.88795 | invalid |

Event median: `0.236480 ms`.

Conclusion: relay readback is not the root cause. Candidate2 corrupts dK before
the relay store path.

### Probe G: Masked Diagonal With Patch Launches Kept

I then tested the only remaining low-risk local candidate2 variant: keep the
existing patch launches enabled, but use `compute_dkdv_loop<true,true,...>` for
consumer 1's first diagonal block.

Resource result:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate2 masked diagonal with patches, noncausal | 120 | 48 B | 165124 B | 0 B | 0 B |
| candidate2 masked diagonal with patches, causal | 128 | 56 B | 165124 B | 4 B | 8 B |

Validation/timing:

| Tensor | finite | rel L2 vs CuTe | max abs / notes |
| --- | --- | ---: | --- |
| dq | yes | 0.999953 | max abs 1.83019 |
| dk | no | n/a | 13282 non-finite values |
| dv | yes | 0.816707 | max abs 1.20059 |

Event median: `0.238336 ms`.

This failed the acceptance gate and was reverted. The extension was rebuilt after
reverting.

### Fallback Chunk-Store Attempt

I also tested a narrow fallback dK store change in the finite dK/dV kernel:
store the three 64-wide dK chunks directly instead of stitching into a 192-wide
register tile before the final store.

Result:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| fallback dK/dV chunk-store attempt | 255 | 552 B | 83992 B | 1596 B | 3988 B |

The ptxas resource shape did not change, and split timing stayed in the same
`416-424 us` dK/dV range. The change was reverted.

Final restored smoke after all rejected probes and rebuild:

| Path | dq finite | dk finite | dv finite | Non-finite counts |
| --- | --- | --- | --- | --- |
| fallback | yes | yes | yes | `[0, 0, 0]` |
| 2CTA experimental | yes | yes | yes | `[0, 0, 0]` |
| candidate2 restored | yes | no | yes | `[0, 12384, 0]` |

No experimental source diff remains in `tk_fa4/b300_bwd_cute16_kernel_candidate2.cuh`.

## Next Implementation Plan

Candidate2 remains a useful diagnostic because it is fast and low-register, but
the simple fixes did not make it viable. The next non-speculative route is a
proper CuTe-like dK/dV kernel rather than more local candidate2 store patches:

1. Build a new opt-in dK/dV-only kernel behind the existing experimental route,
   preserving default fallback behavior.
2. Keep CuTe's structural shape: `tile_m=128`, `tile_n=128`,
   `cluster_shape=(2,1)`, 512 threads/CTA, reduce warps 0-3, compute warps
   4-11, MMA 12, load 13, relay 14, empty 15.
3. Implement the first causal diagonal block as a real masked/repair path inside
   the new kernel, not by skipping candidate2's semaphore pipeline.
4. Keep dK/dV accumulators in TMEM/tcgen05-style storage and avoid the current
   fallback's full live register tiles.
5. Acceptance gate for the next patch: finite `dq/dk/dv`, `dk`/`dv` rel L2 in
   the fallback range, dK/dV resource use below 255 registers with no spills,
   and graph replay below the current fallback's about 0.42 ms before any
   default-route promotion.

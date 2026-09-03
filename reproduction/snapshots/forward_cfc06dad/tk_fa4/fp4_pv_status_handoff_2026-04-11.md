# FP4 PV Status Handoff

## Scope

This note summarizes the current state of the `QK fp4 + PV fp4` causal FA4 work on branch `tk-fa4-sm100-rewrite` as of `2026-04-11`.

The active production path is `forward_fp4pv(...)` in:

- [/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu)

The main experiments and harness files are:

- [/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu)
- [/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py)
- [/workspace/codebases/fp4_matmul/tk_fa4/Makefile](/workspace/codebases/fp4_matmul/tk_fa4/Makefile)

## Current Production State

- `QK` is in NVFP4 and is effectively done for the current benchmark matrix.
- `PV` is running in the production fp4 path and is materially better than earlier broken checkpoints.
- Current keepable production defaults include:
  - direct row update path
  - cluster fence on the `P` stage handoff
  - non-first fp4pv `PV` issue using `mm2`
  - larger grid / effectively fullgrid behavior at `S >= 4096`
- Production `kernel_fp4pv` is currently on the healthy compile line:
  - `16B` stack
  - `20B` spill stores
  - `24B` spill loads

## Main Findings So Far

### 1. QK is not the main problem anymore

Earlier matrix runs showed:

- `qk_only_live_nvfp4` is very tight against `softmax_store_p`
- `qk_nvfp4_plus_ref_pv` is also effectively exact through `S=2048`

So the remaining gap is overwhelmingly on the `PV` side.

### 2. The basic stored-P tcgen mainpath is exact

The stored-`P` late-tile oracle is exact for late windows such as:

- `tile_begin=15, tiles_to_run=1`
- `tile_begin=14, tiles_to_run=2`

That means the raw tcgen mainpath itself is not the long-sequence bug.

Relevant harness entrypoints:

- `measure_stored_p_mainpath_tile_window(...)`
- `measure_stored_p_mainpath_tile_window_subprocess(...)`

### 3. The remaining bug is in the live streaming path

The strongest current diagnosis is:

- not `QK`
- not `LSE`
- not the stored-`P` tcgen mainpath
- likely in the live streaming handoff / remote-half path

The recurring bad rows were localized with row-block profiling to the:

- `224:255 mod 256` slice

That is the last 32 rows of the `cta_rank==1` half-cluster region.

### 4. Production PV is much better than it used to be

Keepable reruns on isolated GPUs showed production improved into a usable regime on the fixed-input comparisons:

- `S=1024`, random: mean abs diff around `0.00955`
- `S=1024`, zero-QK: mean abs diff around `0.02855`
- `S=2048`, random: mean abs diff in roughly the `0.019` to `0.029` range across reruns
- `S=2048`, zero-QK: mean abs diff in roughly the `0.068` to `0.092` range across reruns
- `LSE` stayed tight in those runs

### 5. Direct/shared consumer is the numerical ceiling, tcgen path is the speed path

Experiments showed:

- `streaming_live_localcta_direct` is the numerically strongest live PV path
- `streaming_live_localcta_direct_tcgenaccum` and production tcgen-style paths are faster but looser
- the cheap direct/shared consumer port tried so far was not keepable in production because it hurt the resource line and/or speed too much

## Relevant Commands

### Build

Production extension:

```bash
make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal -B -j1
```

Experiments extension:

```bash
make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments -B -j1
```

Python syntax check:

```bash
python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py
```

### Main matrix / benchmark helpers

Full QK/PV matrix:

```python
benchmark_nvfp4_qk_pv_matrix(...)
```

Focused production persistent vs fullgrid:

```python
benchmark_production_fp4pv_launch_modes(...)
benchmark_production_fp4pv_launch_modes_subprocess(...)
```

Row-block drift profiler:

```python
measure_pv_variant_row_block_profiles(...)
```

Stored-P late-tile oracle:

```python
measure_stored_p_mainpath_tile_window(...)
measure_stored_p_mainpath_tile_window_subprocess(...)
```

### New late-tile live helpers

Windowed live quant compare:

```python
measure_live_quant_tile_window_against_ref(...)
measure_live_quant_tile_window_against_ref_subprocess(...)
```

Windowed live BF16-only compare:

```python
measure_live_bf16_tile_window_against_ref_subprocess(...)
```

## Files Worth Reading First

Production kernel:

- [/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu)

Experiments kernel:

- [/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu)

Harness / orchestration:

- [/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py)

## What Was Added Most Recently

### In the experiments kernel

Added a BF16-only selected-tile dump that reuses the live QK+softmax path but skips the FP4 quantizer consumers:

- `dispatch_quantize_p_from_scores_tile_window_live_bf16_only_debug(...)`

This is intended to answer:

- is the bad late live tile already wrong in BF16 before live FP4 quantization?

### In the Python harness

Added:

- fixed windowed reference compare logic for the selected-tile path
- subprocess-safe live selected-tile compare
- subprocess-safe live BF16-only selected-tile compare

## Current Blockers

### 1. Large-S live dump instability

The raw live dump path remains unstable in the old unsafe regime:

- selected-window live dump returns
- immediate `torch.cuda.synchronize()` can still hang afterward

This means the new selected-window oracle is useful structurally, but the underlying live dump kernel still needs a more stable execution path for large `S`.

### 2. BF16-only selected-window oracle is not signed off yet

The BF16-only selected-tile path was added and builds, but it has not been fully runtime-validated yet because the host CUDA runtime kept flapping with intermittent:

- `cudaGetDeviceCount() ... Error 304`

### 3. Residual long-sequence PV issue is still live-path-specific

The strongest remaining open question is:

- does the remote-half mismatch first appear in live BF16 tile values,
- or only after live FP4 quantization / publication / handoff?

The new BF16-only tile-window oracle was added specifically to answer that.

## Recommended Next Steps

### Highest signal

Run the new BF16-only selected-tile subprocess helper first:

```python
measure_live_bf16_tile_window_against_ref_subprocess(
    seqlen=512,
    input_mode="zero_qk_random_v",
    tile_begin=3,
    tiles_to_run=1,
)
```

If that is healthy, then run:

```python
measure_live_bf16_tile_window_against_ref_subprocess(
    seqlen=2048,
    input_mode="zero_qk_random_v",
    tile_begin=15,
    tiles_to_run=1,
)
```

Interpretation:

- if BF16-only already drifts, the bug is upstream in live QK+softmax / live tile publication
- if BF16-only is clean, the remaining bug is lower in live FP4 quantization / publication / handoff

### If BF16-only is still unstable

Build an even smaller live selected-tile probe that:

- limits execution to one chosen tile window
- avoids the older full live-dump machinery
- minimizes persistent/task reuse and heavy debug bookkeeping

## Commit Hygiene

There are unrelated local modifications in the repo outside the fp4pv work. When committing or cherry-picking from this checkpoint, keep the scope to:

- [/workspace/codebases/fp4_matmul/tk_fa4/Makefile](/workspace/codebases/fp4_matmul/tk_fa4/Makefile)
- [/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu)
- [/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu)
- this handoff file



## Continued Work After This Handoff

### New code added

- Added an exact-grid BF16-only selected-window entrypoint in the experiments extension:
  - `quantize_p_from_scores_tile_window_live_bf16_only_minimal_debug(...)`
- `_dump_live_bf16_tile_window_from_scores(...)` now prefers that minimal exact-grid path when the rebuilt extension is present.

### Runtime findings on the current host

- `cuda:0` was not usable during this pass:
  - `CUDA-capable device(s) is/are busy or unavailable`
- `cuda:1`, `cuda:2`, and `cuda:3` were allocatable.
- On `cuda:1`, the direct exact-grid BF16-only dump returns successfully for:
  - `S=512, tile_begin=3, tiles_to_run=1`
- The first unsafe point is still the first post-launch CUDA sync / consumer op, not the Python call return.

### Exact-grid BF16-only sync boundary

Using the direct dump plus an immediate `torch.cuda.synchronize()` on `cuda:1`:

- `S=128` is sync-safe
- `S=256` is sync-safe
- `S=384` hangs after printing `s384-live` and times out at `45s`
- `S=512` hangs after printing `s512-live` and times out at `45s`

This is a stronger boundary than we had before: the minimal BF16-only probe is healthy through `S=256` and first reproduces the live-path hang by `S=384`.

### Additional interpretation

- The exact-grid / no-debug change removed some launcher noise but did not remove the core live-path instability.
- The current selected-window BF16 helper is still **not** a true compute-limited tile-window probe.
- It only narrows the output / debug stores; the kernel still runs the full causal `idx` loop needed for the active row tiles.
- That means the remaining hang can still be caused by earlier live score-tile work even when only one late window is being dumped.

### Row-tile probe implemented

- Added a dedicated exact-grid row-tile BF16-only entrypoint in the experiments extension:
  - `quantize_p_from_scores_row_tile_window_live_bf16_only_minimal_debug(...)`
- The new path keeps the full prior-`K` scan for the selected row cluster, but launches only one cluster per `(batch, head)` and only materializes the chosen `row_tile`.
- Added Python wrappers:
  - `_dump_live_bf16_row_tile_window_from_scores(...)`
  - `measure_live_bf16_row_tile_window_sync_subprocess(...)`

### Row-tile sync localization on `cuda:1`

Using the new subprocess-safe row-tile probe with an immediate `torch.cuda.synchronize()`:

- `S=256, row_tile=1, tile_begin=1`:
  - `sync_ok=True`
- `S=384, row_tile=0, tile_begin=0`:
  - `sync_ok=True`
- `S=384, row_tile=2, tile_begin=2`:
  - `sync_ok=False`
  - timeout after `25s`
- `S=512, row_tile=2, tile_begin=2`:
  - `sync_ok=False`
  - timeout after `25s`

### Updated interpretation

- The first unsafe case now localizes to the late row cluster that needs `3+` causal score-tile iterations.
- An early `2`-iteration cluster at `S=384` is clean.
- This is materially stronger than the earlier full-row exact-grid result: the hang is no longer just "somewhere after launch at `S>=384`"; it reproduces with a single selected late row tile.

### Executed-iteration split on `S=384, row_tile=2`

Added one more control to the same row-tile probe:

- optional `executed_k_iters`
- optional `store_bf16`
- optional `store_lse`

Results on `cuda:1`:

- `executed_k_iters=2, tile_begin=0, store_bf16=True, store_lse=True`:
  - `sync_ok=True`
- `executed_k_iters=3, tile_begin=0, store_bf16=True, store_lse=True`:
  - `sync_ok=False`
  - timeout after `25s`
- `executed_k_iters=3, tile_begin=1, store_bf16=True, store_lse=True`:
  - `sync_ok=False`
  - timeout after `25s`
- `executed_k_iters=3, tile_begin=2, store_bf16=True, store_lse=True`:
  - `sync_ok=False`
  - timeout after `25s`
- `executed_k_iters=3, tile_begin=0, store_bf16=False, store_lse=True`:
  - `sync_ok=False`
  - timeout after `25s`
- `executed_k_iters=3, tile_begin=0, store_bf16=False, store_lse=False`:
  - `sync_ok=False`
  - timeout after `25s`

Interpretation:

- The transition is exactly between `2` executed causal iterations and `3` executed causal iterations for the first bad late cluster.
- Which BF16 tile window is selected for output does not matter once `3` iterations execute.
- BF16 tile stores are not required for the hang.
- LSE stores are also not required for the hang.
- So the current evidence points upstream of all host-visible BF16/LSE stores, inside the live late-cluster QK / online-softmax / inter-warpgroup handoff path for the third causal iteration.

### Third-iteration stage-cut sweep

Added a new `stage_cut` control on the same row-tile BF16-only probe. This is still the `QK_ONLY` experiments path.

Supported cuts:

- `full`
- `issue3_only`
- `consume3_copy`
- `consume3_rowmax`
- `consume3_exp`
- `consume3_scan`

Canonical sweep case:

- `row_tile=2`
- `executed_k_iters=3`
- `tile_begin=0`
- `tiles_to_run=1`
- `store_bf16=False`
- `store_lse=False`

Results on `cuda:1`:

- `S=384`:
  - `full`: timeout after `25s`
  - `issue3_only`: timeout after `25s`
  - `consume3_copy`: timeout after `25s`
  - `consume3_rowmax`: timeout after `25s`
  - `consume3_exp`: timeout after `25s`
  - `consume3_scan`: timeout after `25s`
- `S=512`:
  - `full`: timeout after `25s`
  - `issue3_only`: timeout after `25s`
  - `consume3_copy`: timeout after `25s`
  - `consume3_rowmax`: timeout after `25s`
  - `consume3_exp`: timeout after `25s`
  - `consume3_scan`: timeout after `25s`

Regression check:

- `S=384, row_tile=2, executed_k_iters=2, stage_cut=full, store_bf16=False, store_lse=False`:
  - `sync_ok=True`

Updated interpretation:

- The stage-cut plumbing did not move the boundary: `2` executed iterations remain safe and `3` executed iterations remain unsafe.
- The earliest failing cut is already `issue3_only`.
- That means the deadlock appears as soon as the producer side performs the third score-tile issue/commit sequence; the consumer does not need to enter its third-iteration math for the failure to reproduce.
- The remaining suspect region is therefore even narrower than before: third-iteration producer-side reuse / `p_copy_done` gating / `scores_arrived[0]` publication, not the later consumer-side row-max / exp / scan stages.

### Producer-side third-issue split

Added producer-side milestone cuts inside the third `issue_next_qk(...)` path:

- `producer3_post_wait`
- `producer3_post_issue`
- `producer3_post_commit`
- `producer3_post_ksc_finish`
- `consume3_wait_only`

Canonical case stayed:

- `row_tile=2`
- `executed_k_iters=3`
- `tile_begin=0`
- `tiles_to_run=1`
- `store_bf16=False`
- `store_lse=False`

Results on `cuda:1`:

- `S=384`:
  - `producer3_post_wait`: `sync_ok=True`
  - `producer3_post_issue`: `sync_ok=True`
  - `producer3_post_commit`: timeout after `25s`
  - `producer3_post_ksc_finish`: timeout after `25s`
  - `issue3_only`: timeout after `25s`
  - `consume3_wait_only`: timeout after `25s`
- `S=512`:
  - `producer3_post_wait`: `sync_ok=True`
  - `producer3_post_issue`: `sync_ok=True`
  - `producer3_post_commit`: timeout after `25s`
  - `producer3_post_ksc_finish`: timeout after `25s`
  - `issue3_only`: timeout after `25s`
  - `consume3_wait_only`: timeout after `25s`

Updated interpretation:

- The third producer-side `wait(p_copy_done[reuse_buf], ...)` is not the failing step.
- The third `issue_qk_chunked_qsc_tmem(...)` is also not the failing step.
- The first unsafe transition is the third `detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0])`.
- Allowing the consumer to execute the third `wait(scores_arrived[0], scores_phase)` is still not sufficient to make the kernel complete.
- So the remaining suspect region is now the third `scores_arrived[0]` commit itself, or the immediate wait/commit handshake semantics around that commit.

### Third-commit skip-tail probe

Added one more producer-side cut:

- `producer3_skip_commit_tail`

This cut skips only the third `detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0])`, but still executes the rest of the producer tail for that final iteration:

- `arrive(k_sc_finished[k_sc_slot])`
- `fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1)`
- final `detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0])`

Canonical case stayed:

- `row_tile=2`
- `executed_k_iters=3`
- `tile_begin=0`
- `tiles_to_run=1`
- `store_bf16=False`
- `store_lse=False`

Results on `cuda:1`:

- `S=384`:
  - `producer3_post_issue`: `sync_ok=True`
  - `producer3_post_commit`: timeout after `25s`
  - `producer3_skip_commit_tail`: `sync_ok=True`
  - `producer3_post_ksc_finish`: timeout after `25s`
- `S=512`:
  - `producer3_post_issue`: `sync_ok=True`
  - `producer3_post_commit`: timeout after `25s`
  - `producer3_skip_commit_tail`: `sync_ok=True`
  - `producer3_post_ksc_finish`: timeout after `25s`

Updated interpretation:

- The producer tail after the third score issue is not sufficient to reproduce the hang by itself.
- The failure requires executing the third `scores_arrived[0]` commit; skipping just that one commit while keeping `k_sc_finished` and final `q_finished` publication makes the kernel complete.
- That narrows the remaining suspect region to the third `detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0])` instruction itself, or state that this instruction mutates immediately on publication.

### Fresh-semaphore third-commit probe

Added one more producer-side cut:

- `producer3_fresh_commit_tail`

This cut keeps the third producer-side commit, but diverts it to a fresh debug semaphore instead of reusing `scores_arrived[0]`. The rest of the producer tail stays the same as `producer3_skip_commit_tail`:

- `arrive(k_sc_finished[k_sc_slot])`
- `fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1)`
- final `detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0])`

Canonical case stayed:

- `row_tile=2`
- `executed_k_iters=3`
- `tile_begin=0`
- `tiles_to_run=1`
- `store_bf16=False`
- `store_lse=False`

Results on `cuda:1`:

- `S=384`:
  - `producer3_post_commit`: timeout after `25s`
  - `producer3_skip_commit_tail`: `sync_ok=True`
  - `producer3_fresh_commit_tail`: `sync_ok=True`
- `S=512`:
  - `producer3_post_commit`: timeout after `25s`
  - `producer3_skip_commit_tail`: `sync_ok=True`
  - `producer3_fresh_commit_tail`: `sync_ok=True`

Updated interpretation:

- The third producer-side commit instruction is not inherently toxic on this debug path; the same commit primitive works when it targets a fresh semaphore.
- The failure depends on reusing `scores_arrived[0]` for the third publication.
- That makes the highest-probability root cause a reuse/phase/count mismatch on `scores_arrived[0]`, not `k_sc_finished`, not final `q_finished`, and not a generic “third commit” failure.

### Fresh-semaphore consumer probes

Added three more stage cuts:

- `fresh3_wait_only`
- `fresh3_copy`
- `fresh3_full`

All three route only the third producer-side score publication onto the fresh debug semaphore, and then let the consumer progress by different amounts on that third iteration:

- `fresh3_wait_only`: third consumer wait only
- `fresh3_copy`: third consumer wait plus `tcgen05.ld`
- `fresh3_full`: full third consumer iteration with the fresh semaphore

Canonical case stayed:

- `row_tile=2`
- `executed_k_iters=3`
- `tile_begin=0`
- `tiles_to_run=1`
- `store_bf16=False`
- `store_lse=False`

Results on `cuda:1`:

- `S=384`:
  - `fresh3_wait_only`: `sync_ok=True`
  - `fresh3_copy`: `sync_ok=True`
  - `fresh3_full`: `sync_ok=True`
- `S=512`:
  - `fresh3_wait_only`: `sync_ok=True`
  - `fresh3_copy`: `sync_ok=True`
  - `fresh3_full`: `sync_ok=True`

Updated interpretation:

- The third score tile itself is consumable: the third wait, TMEM score load, and full third softmax iteration all complete when only the publication slot changes.
- That rules out third-iteration score contents, `tcgen05.ld`, and downstream QK-only softmax math as the source of the timeout.
- The remaining suspect is now specifically reuse of `scores_arrived[0]` in this debug/minimal path.

### Dual-slot full reuse probe

Added one more debug mode:

- `dual_slot_full`

This mode stops treating the row-tile minimal probe as a one-slot `scores_arrived[0]` pipeline and instead alternates the score publication / wait pair across two slots with per-slot parity:

- iteration 0: `scores_arrived[0]`, phase `0`
- iteration 1: fresh slot, phase `0`
- iteration 2: `scores_arrived[0]`, phase `1`

Results on `cuda:1` for the canonical late row-tile case:

- `S=384`:
  - `full`: timeout after `25s`
  - `dual_slot_full`: `sync_ok=True`
- `S=512`:
  - `full`: timeout after `25s`
  - `dual_slot_full`: `sync_ok=True`

Store-enabled validation:

- `dual_slot_full` also syncs with `store_bf16=True` and `store_lse=True` at both `S=384` and `S=512`.
- On the zero-`QK` canonical case, stored `LSE` matches the pure BF16 causal reference to numerical noise:
  - `lse_max_abs_diff = 9.5367431640625e-07`
- The stored BF16 tile window is not a full final-softmax equivalence check in this path because the live row-tile dump stores tile-local normalized values before later online rescale contributions are folded back into earlier tiles.

Updated interpretation:

- The hang is not an unavoidable consequence of three causal iterations.
- The minimal row-tile probe becomes healthy once score publication/wait uses a two-slot parity discipline instead of repeatedly reusing one slot through `0 -> 1 -> 0`.
- The practical fix target is now the collapsed one-slot `scores_arrived[0]` handoff in the minimal debug path.

### Full-path promotion and focused validation

Promoted the two-slot handoff to the default row-tile `stage_cut="full"` behavior and added an explicit legacy repro mode:

- `full`: healthy two-slot parity path
- `dual_slot_full`: backward-compatible alias of the same healthy path
- `legacy_single_slot_full`: old one-slot reuse path for reproducing the timeout

Focused validation results:

- `cuda:1`
  - `S=384, row_tile=0, full, store_bf16/store_lse=True`: `sync_ok=True`
  - `S=384, row_tile=2, full, store_bf16/store_lse=True`: `sync_ok=True`
- `cuda:2`
  - `S=384, row_tile=2, full, store_bf16/store_lse=False`: `sync_ok=True`
  - `S=384, row_tile=2, full, store_bf16/store_lse=True`: `sync_ok=True`
  - `S=384, row_tile=2, legacy_single_slot_full, store on/off`: both timeout after `25s`
- `cuda:3`
  - `S=384, row_tile=2, full, store_bf16/store_lse=True`: `sync_ok=True`
  - `S=512, row_tile=2, full, store_bf16/store_lse=False`: `sync_ok=True`
  - `S=512, row_tile=2, full, store_bf16/store_lse=True`: `sync_ok=True`
  - `S=512, row_tile=2, legacy_single_slot_full, store on/off`: both timeout after `25s`

Alias/numerical sanity on `cuda:1`:

- `full` and `dual_slot_full` are bit-identical on the canonical `S=384, row_tile=2` case.
- Zero-`QK` `LSE` still matches the pure BF16 causal reference:
  - `lse_max_abs_diff = 9.5367431640625e-07`

Environment note:

- A direct `torch.empty(..., device='cuda:0')` still fails with `cudaErrorDevicesUnavailable` in this runtime, so the focused validation was completed on `cuda:1`, `cuda:2`, and `cuda:3`.

### Larger-sequence comparisons

Scaled the fixed row-tile `full` path to later row tiles and larger sequence lengths by always selecting the final row tile and final stored tile window.

Zero-`QK` sync health with stores enabled:

- `cuda:1`
  - `S=768, row_tile=5, full`: `sync_ok=True`
  - `S=768, row_tile=5, legacy_single_slot_full`: timeout after `25s`
  - `S=1024, row_tile=7, full`: `sync_ok=True`
  - `S=1024, row_tile=7, legacy_single_slot_full`: timeout after `25s`
  - `S=1536, row_tile=11, full`: `sync_ok=True`
  - `S=1536, row_tile=11, legacy_single_slot_full`: timeout after `25s`
  - `S=6144, row_tile=47, full`: `sync_ok=True`
  - `S=8192, row_tile=63, full`: `sync_ok=True`
- `cuda:2`
  - `S=2048, row_tile=15, full`: `sync_ok=True`
  - `S=2048, row_tile=15, legacy_single_slot_full`: timeout after `25s`
  - `S=3072, row_tile=23, full`: `sync_ok=True`
  - `S=3072, row_tile=23, legacy_single_slot_full`: timeout after `25s`
- `cuda:3`
  - `S=4096, row_tile=31, full`: `sync_ok=True`
  - `S=4096, row_tile=31, legacy_single_slot_full`: timeout after `25s`

Random-live-FP4 sync sanity with stores enabled:

- `cuda:2`
  - `S=2048, row_tile=15, full`: `sync_ok=True`
  - `S=2048, row_tile=15, legacy_single_slot_full`: timeout after `25s`
- `cuda:3`
  - `S=4096, row_tile=31, full`: `sync_ok=True`
  - `S=4096, row_tile=31, legacy_single_slot_full`: timeout after `25s`

Final-tile numerical comparison against reference on zero-`QK`:

- For the final stored tile window, `full` matches the pure BF16 reference exactly at `S=768, 1024, 1536, 2048, 3072, 4096`.
  - `bf16_max_abs_diff = 0.0`
  - `lse_max_abs_diff = 9.5367431640625e-07`
- The same final-tile exactness also holds with an analytic zero-`QK` reference at `S=8192` and `S=16384`.
  - `bf16_max_abs_diff = 0.0`
  - `lse_max_abs_diff = 9.5367431640625e-07`
- `full` and `dual_slot_full` remain bit-identical on the final-tile comparison sweep.

Updated interpretation:

- The promoted two-slot `full` path is stable well beyond the original `S=384/512` failure boundary.
- The legacy one-slot reuse path still fails immediately once the selected row tile needs `3+` score publications, even when the selected row tile is the final tile of much larger sequences.
- The earlier BF16 mismatch was a tile-window interpretation issue for non-final windows; on the final tile, the stored BF16 window matches the BF16 causal reference exactly.

### Exact-grid helper comparison

Followed up on the earlier question of whether the wider exact-grid minimal BF16 helper still carried the old one-slot behavior.

Current result: it does not. The exact-grid helper now shares the promoted two-slot `full` path through the common debug kernel.

Kernel-side reason:

- `DEBUG_STAGE_CUT_FULL` now selects the two-slot score handoff in the shared debug kernel via `debug_dual_scores_slot_mode(...)`.
- The exact-grid minimal helper launches the same kernel with the default `debug_stage_cut=0`, so it inherits the same two-slot score-publication rule.

Sync health for the exact-grid minimal BF16 helper (`_dump_live_bf16_tile_window_from_scores(...)`) on the final tile window:

- zero-`QK`
  - `S=256`: `sync_ok=True`
  - `S=384`: `sync_ok=True`
  - `S=768`: `sync_ok=True`
  - `S=2048`: `sync_ok=True`
  - `S=4096`: `sync_ok=True`
  - `S=8192`: `sync_ok=True`
- random-live-FP4
  - `S=2048`: `sync_ok=True`
  - `S=4096`: `sync_ok=True`

Exact-grid versus row-tile comparison:

- zero-`QK`
  - `S=256, final tile`: exact-grid selected-row slice and row-tile `full` are bit-identical
  - `S=2048, final tile`: exact-grid selected-row slice and row-tile `full` are bit-identical
- random-live-FP4
  - `S=2048, final tile`: exact-grid selected-row slice and row-tile `full` are numerically very close
    - `bf16_max_abs_diff = 3.814697265625e-06`
    - `lse_max_abs_diff = 0.00011682510375976562`

Exact-grid versus reference on the final tile window:

- zero-`QK`, `S=2048`
  - `bf16_max_abs_diff = 0.0`
  - `lse_max_abs_diff = 0.0`
- random-live-FP4, `S=2048`
  - `bf16_mean_abs_diff = 1.305401383433491e-06`
  - `bf16_max_abs_diff = 0.000209808349609375`
  - `lse_mean_abs_diff = 0.003015863010659814`
  - `lse_max_abs_diff = 0.0805019736289978`

Updated interpretation:

- The earlier exact-grid hang result is superseded by the current workspace state.
- Promoting the two-slot `full` behavior in the shared debug kernel fixed both the row-tile minimal helper and the wider exact-grid minimal BF16 helper.
- In the current tree, the only preserved repro of the old deadlock is the explicit row-tile `legacy_single_slot_full` mode.

### Random-seed final-tile characterization

Expanded the random-live-FP4 comparisons to separate three different questions:

- exact-grid versus reference over the full selected tile window,
- exact-grid versus row-tile on the final row tile,
- and exact-grid / row-tile versus reference on the final row tile only.

Exact-grid versus reference, full selected tile window (`random_live_fp4`):

- `S=2048`, seeds `0..4`
  - `bf16_mean_abs_diff` stayed in `[1.3052e-06, 1.3335e-06]`
  - `bf16_max_abs_diff` stayed in `[2.0599e-04, 2.3842e-04]`
  - `lse_mean_abs_diff` stayed in `[2.9511e-03, 3.1048e-03]`
  - `lse_max_abs_diff` stayed in `[6.2256e-02, 2.1972e-01]`
- `S=4096`, seeds `0..2`
  - `bf16_mean_abs_diff` stayed in `[3.1910e-07, 3.2894e-07]`
  - `bf16_max_abs_diff` stayed in `[1.0109e-04, 1.1826e-04]`
  - `lse_mean_abs_diff` stayed in `[2.1837e-03, 2.3499e-03]`
  - `lse_max_abs_diff` stayed in `[4.0723e-02, 1.6366e-01]`

Sync-only scale sanity:

- exact-grid `random_live_fp4`, final tile:
  - `S=8192`: `sync_ok=True`

Exact-grid versus row-tile on the final row tile (`random_live_fp4`):

- `S=2048`, seeds `0..4`
  - `bf16_max_abs_diff <= 3.814697265625e-06`
  - `lse_max_abs_diff <= 0.00061798095703125`
- `S=4096`, seeds `0..2`
  - `bf16_max_abs_diff <= 1.9073486328125e-06`
  - `lse_max_abs_diff <= 0.00029659271240234375`

That is much smaller than either helper’s gap to the reference, so the remaining random-input discrepancy is shared by both debug paths rather than caused by exact-grid versus row-tile launch topology.

Final row tile versus reference (`random_live_fp4`):

- `S=2048`, seeds `0..4`
  - exact-grid and row-tile are effectively the same within measurement noise
  - `bf16_mean_abs_diff` is about `2.09e-05 .. 2.13e-05`
  - `bf16_max_abs_diff` is about `2.0599e-04 .. 2.3842e-04`
  - `lse_mean_abs_diff` is about `3.72e-03 .. 4.79e-03`
  - `lse_max_abs_diff` is about `1.1726e-02 .. 1.9225e-02`
- `S=4096`, seed `0`
  - exact-grid:
    - `bf16_mean_abs_diff = 1.0210264008492231e-05`
    - `bf16_max_abs_diff = 0.0001010894775390625`
    - `lse_mean_abs_diff = 0.0042823925614356995`
    - `lse_max_abs_diff = 0.013444900512695312`
  - row-tile:
    - `bf16_mean_abs_diff = 1.0211486369371414e-05`
    - `bf16_max_abs_diff = 0.0001010894775390625`
    - `lse_mean_abs_diff = 0.004277773201465607`
    - `lse_max_abs_diff = 0.013338088989257812`

Updated interpretation:

- The previously reported large random-input `lse_max_abs_diff` numbers were full-sequence metrics and overstated the late-tile discrepancy that matters for the selected final row tile.
- On the final row tile, the remaining exact-grid/reference and row-tile/reference gap is noticeably smaller and quite stable across seeds.
- Because exact-grid and row-tile match each other much more closely than either matches the reference, the remaining gap is common kernel behavior rather than a consequence of the row-tile debug specialization.

### Decoded-FP4 reference attribution

Added a direct live-Q/K dequantization helper in Python:

- `_dequantize_live_fp4_qk_for_debug(...)`

This helper reverses the live Q/K FP4 payload plus prepared-scale format back to BF16 for debug/reference comparisons, including collapsing the depth-expanded live-`K_sc` layout back to the original per-128-row scale blocks.

Sanity on paired random-live-FP4 inputs (`v5` Q/K packing):

- `S=2048`, seed `0`
  - `Q` mean abs diff versus original BF16 source after dequantization: about `0.02196`
  - `K` mean abs diff versus original BF16 source after dequantization: about `0.02214`

Using that decoded-FP4 BF16 Q/K as the reference source of truth:

- `softmax_store_p(...)` matches the decoded-FP4 BF16 causal reference almost exactly on the final row tile.
- The shared debug kernel remains the outlier.

Final row tile versus decoded-FP4 BF16 causal reference:

- `S=2048`, seeds `0..4`
  - `softmax_store_p`
    - `bf16_max_abs_diff <= 3.814697265625e-06`
    - `lse_max_abs_diff <= 2.2411346435546875e-05`
  - exact-grid debug BF16 tile window
    - `bf16_max_abs_diff` stayed in `[2.0599e-04, 2.3842e-04]`
    - `lse_max_abs_diff` stayed in `[1.1638e-02, 1.9495e-02]`
- `S=4096`, seed `0`
  - `softmax_store_p`
    - `bf16_max_abs_diff = 1.9073486328125e-06`
    - `lse_max_abs_diff = 1.1444091796875e-05`
  - exact-grid debug BF16 tile window
    - `bf16_max_abs_diff = 0.0001010894775390625`
    - `lse_max_abs_diff = 0.013553619384765625`

Implication:

- The remaining discrepancy is not a disagreement between `forward_persistent` and the intended decoded-FP4 softmax semantics.
- `forward_persistent` is already very close to the decoded-FP4 BF16 reference.
- The numerical drift is in the shared debug score-to-`P` kernel family (`kernel_quantize_p_from_scores_debug`), not in the production `softmax_store_p` path.

Minimal BF16-only versus mixed debug kernel:

- `S=2048`, seeds `0..4`
  - `min_vs_mix_bf16_max_abs_diff <= 3.814697265625e-06`
  - `min_vs_mix_lse_max_abs_diff <= 0.0010967254638671875`
- `S=4096`, seed `0`
  - `min_vs_mix_bf16_max_abs_diff = 1.9073486328125e-06`
  - `min_vs_mix_lse_max_abs_diff = 0.0006513595581054688`

That rules out the BF16-only `QK_ONLY` specialization as the cause. The mixed debug path that still runs `P` quantization consumers has essentially the same drift as the BF16-only minimal path.

Tilewise online-state comparison on the final row tile (`S=2048`, seed `0`):

- Reconstructed per-tile `LSE` from the stable tile-window debug `row_max`/`row_sum` outputs and compared it to the decoded-FP4 BF16 reference history.
- The drift is already present by the second causal tile, not only at the final tile.
  - tile `0`: `lse_max_abs_diff = 0.0005645751953125`
  - tile `1`: `lse_max_abs_diff = 0.010117530822753906`
  - tile `15`: `lse_max_abs_diff = 0.015621662139892578`

Hypothesis check:

- Tried replacing the debug kernel’s thresholded `acc_scale` update with the production-style strict rescale rule.
- Rebuilt and reran the decoded-FP4 comparison.
- Result: no material change in the late-tile BF16/LSE error.
- That change was reverted, so the current workspace does not keep an unvalidated kernel edit from this experiment.

Updated interpretation:

- The live semaphore/handoff debugging work is effectively done for the debug probes.
- The remaining issue is now a pure numerical mismatch in the shared debug score-to-`P` kernel family.
- The mismatch shows up even when comparing against a decoded-FP4 BF16 causal reference, so it is not explained by FP4 Q/K quantization alone.
- Because the mixed debug path and BF16-only minimal path agree closely with each other, the remaining drift is upstream of `P` packing/store consumers and downstream of the decoded-FP4 Q/K semantics.

### Best next step

At this point the debug-only BF16 probes are healthy in both the row-tile and exact-grid minimal helpers, and the remaining random-input discrepancy is common to both. The highest-value follow-on is therefore to explain that shared kernel/reference gap rather than to keep comparing launch shapes:

- compare the shared debug-kernel math against the decoded-FP4 BF16 causal reference at a more granular stage boundary to determine whether the residual difference comes from score-tile accumulation, causal masking, exponentiation, or final BF16 normalization/store,
- inspect the fused/live kernel’s score-publication slots around the third publication to confirm whether production already matches the healthy parity discipline,
- or clean up the row-tile experiments API by eventually dropping the `dual_slot_full` alias once local scripts stop depending on it.

The current evidence is now strong enough that future work should focus on the shared debug-kernel numerical gap, not on re-opening the already-fixed debug-probe deadlock or on second-guessing `forward_persistent`.

## 2026-04-12 follow-up: tile-local score probe and QK-backend split

I added two new tile-window debug observables to the shared score-to-`P` kernel:

- `tile_score_max`: the per-row max of the current score tile before the online `row_max = max(old, tile)` update
- `tile_exp_sum`: the per-row sum of `exp(score - new_row_max)` for the current tile before the online `row_sum` accumulation

These are now returned by the Python tile-window wrapper `_dump_live_quant_p_tile_window_and_bf16_ref_from_scores(...)` in the `debug` dict.

### What the new probe showed

On the same decoded-FP4 reference case as above (`random_live_fp4`, `S=2048`, seed `0`, final row tile on `cuda:1`):

- tile `0`
  - `tile_score_max_mean_abs = 1.2679e-04`
  - `tile_score_max_max_abs = 5.5699e-04`
  - `tile_exp_sum_mean_abs = 1.4070e-02`
  - `tile_exp_sum_max_abs = 5.2826e-02`
- tile `1`
  - `tile_score_max_mean_abs = 3.2951e-02`
  - `tile_score_max_max_abs = 1.1760e-01`
  - `tile_exp_sum_mean_abs = 2.3170`
  - `tile_exp_sum_max_abs = 12.1488`
- tile `15`
  - `tile_score_max_mean_abs = 4.3143e-02`
  - `tile_score_max_max_abs = 2.1987e-01`
  - `tile_exp_sum_mean_abs = 3.0731`
  - `tile_exp_sum_max_abs = 14.0365`

This is the important correction to the earlier interpretation:

- the first large error is already in `tile_score_max` at tile `1`
- `tile_exp_sum`, `row_max`, and `row_sum` then drift downstream of that
- so the dominant issue is upstream of the online normalization logic, in the per-tile score generation path from the second causal tile onward

### Full-vs-legacy check at 2 causal iterations

To make sure the promoted two-slot `scores_arrived` fix was not contaminating the numerical comparison, I checked the 2-tile case where both the old and new publication modes are safe:

- case: `S=256`, `row_tile=1`, `executed_k_iters=2`, `tile_begin=1`, `random_live_fp4`
- `full`, `dual_slot_full`, and `legacy_single_slot_full` were numerically identical
  - `bf16_mean_abs_diff = 1.9758e-04`
  - `bf16_max_abs_diff = 0.00238037`
  - `lse_mean_abs_diff = 0.00467281`
  - `lse_max_abs_diff = 0.0180354`

So the remaining numerical issue is not caused by the debug-probe semaphore fix. It already exists in the common 2-iteration path.

### Backend split: `localcta` is exact, `v5` and `toy` are not

This was the highest-value result from this round.

Using the same decoded-live-FP4 BF16 reference and the same exact-grid tile-window debug path:

- `S=256`, final row tile, final tile
  - `v5`
    - `bf16_mean_abs_diff = 1.9737e-04`
    - `bf16_max_abs_diff = 0.00241089`
    - `lse_mean_abs_diff = 0.00360378`
    - `lse_max_abs_diff = 0.0153165`
  - `localcta`
    - `bf16_mean_abs_diff = 0.0`
    - `bf16_max_abs_diff = 0.0`
    - `lse_mean_abs_diff = 1.2293e-07`
    - `lse_max_abs_diff = 4.7684e-07`
  - `toy`
    - `bf16_mean_abs_diff = 1.9477e-04`
    - `bf16_max_abs_diff = 0.00247192`
    - `lse_mean_abs_diff = 0.00345178`
    - `lse_max_abs_diff = 0.0143342`

- `S=2048`, final row tile, final tile
  - `v5`
    - `bf16_mean_abs_diff = 2.0879e-05`
    - `bf16_max_abs_diff = 2.0981e-04`
    - `lse_mean_abs_diff = 0.00435890`
    - `lse_max_abs_diff = 0.0155678`
  - `localcta`
    - `bf16_mean_abs_diff = 0.0`
    - `bf16_max_abs_diff = 0.0`
    - `lse_mean_abs_diff = 2.7567e-07`
    - `lse_max_abs_diff = 9.5367e-07`

Tile-1 `tile_score_max` on the 2-tile case (`S=256`) shows the same split:

- `v5`: `mean/max = 0.04211 / 0.19818`
- `toy`: `mean/max = 0.04174 / 0.21009`
- `localcta`: `mean/max = 0.0 / 0.0`

That means the remaining problem is not a universal softmax/math drift in `kernel_quantize_p_from_scores_debug`. The debug kernel is numerically clean on `localcta` Q/K inputs and wrong on `v5` / `toy` Q/K inputs.

### Updated interpretation

The previous “shared debug-kernel math drift” conclusion was too broad. The tighter statement is:

- the debug kernel is healthy on `localcta` live Q/K inputs
- the remaining mismatch is specific to the `v5` / `toy` live Q/K representation path
- the first large error appears at the second causal tile in `tile_score_max`, so the issue is upstream of online normalization and downstream of input packing / score-generation setup

`toy` has all-ones prepared scales, so this is not purely a `q_sc` / `k_sc` staging problem. The common factor between `v5` and `toy` is the non-`localcta` live Q/K payload path.

Follow-up check on scalar `sg`:

- started from an exact `localcta` Q/K payload
- manually replaced `q_sg` / `k_sg` with:
  - `0.2`
  - `4.5e-4` / `4.4e-4` (roughly `v5`-like magnitude)
- the debug kernel stayed exact in both cases:
  - `bf16_mean_abs_diff = 0.0`
  - `bf16_max_abs_diff = 0.0`
  - `lse_max_abs_diff = 4.76837158203125e-07`
  - tile-1 `tile_score_max mean/max = 0.0 / 0.0`

So the scalar `q_sg` / `k_sg` path is not the culprit either. The remaining common factor is now tighter still:

- the debug kernel is correct on `localcta` FP4 payloads under arbitrary scalar `sg`
- it is wrong on `v5` and `toy` FP4 payloads
- therefore the remaining issue is in the non-`localcta` FP4 Q/K payload interpretation/consumption path, not in scalar scale application

## 2026-04-12 correction: `localcta` live-QK control was degenerate

The earlier `localcta` “exact” control result was not a valid nonzero live-QK comparison.

I checked the actual live tensors produced by `_pack_live_fp4_qk_localcta_from_bf16(...)` on nonzero random inputs and found:

- the folded live `Q_sc` tensor was entirely zero
- the expanded live `K_sc` tensor was entirely zero
- `_dequantize_live_fp4_qk_for_debug(...)` therefore returned all-zero `Q/K`

Concrete check on `S=256`, `seed=0`, `batch=heads=1`:

- `q_sc.shape = (1, 2, 3, 512)`, all values `0.0`
- `k_sc.shape = (1, 4, 3, 512)`, all values `0.0`
- `q_deq abs mean/max = 0.0 / 0.0`
- `k_deq abs mean/max = 0.0 / 0.0`

So the earlier `localcta` “exact” comparison was effectively a zero-QK control, not evidence that the debug kernel was correct on a nontrivial localcta live-QK path.

I added a guard in `_pack_live_fp4_qk_localcta_from_bf16(...)` so this path now raises on nonzero inputs when the folded live prepared scales underflow completely to zero, instead of silently returning a degenerate control input.

## 2026-04-12 follow-up: the real `v5` split is in combined `Q_sc * K_sc`

After removing the invalid `localcta` control, the strongest live-QK result is now on the real `v5` path:

Baseline `v5`, `S=256`, final row tile / final tile, compared against `softmax_store_p(...)` on the exact same inputs:

- `bf16_mean_abs_diff = 1.9736e-04`
- `bf16_max_abs_diff = 0.00241089`
- `lse_mean_abs_diff = 0.00360297`
- `lse_max_abs_diff = 0.0153050`

Key scale-split experiments on the same `v5` payload:

- replace only `Q_sc` with ones:
  - `bf16_max_abs_diff = 3.0517578125e-05`
  - `lse_max_abs_diff = 7.772445678710938e-05`
- replace only `K_sc` with ones:
  - `bf16_max_abs_diff = 3.0517578125e-05`
  - `lse_max_abs_diff = 7.200241088867188e-05`
- replace both `Q_sc` and `K_sc` with ones:
  - `bf16_mean_abs_diff = 0.0`
  - `bf16_max_abs_diff = 0.0`
  - `lse_max_abs_diff = 4.76837158203125e-07`

So the real `v5` mismatch is not “payload only” and not scalar `q_sg/k_sg`.

It depends specifically on having both nontrivial prepared-scale factors active at once.

Another useful cut was a constant-scale sweep on the same `v5` payload with `Q_sc = K_sc = c`:

- `c=1`: exact
- `c=16`: `bf16_max_abs_diff = 3.0517578125e-05`, `lse_max_abs_diff = 9.1552734375e-05`
- `c=64`: `bf16_max_abs_diff = 2.13623046875e-04`, `lse_max_abs_diff = 0.00147152`
- `c=128`: `bf16_max_abs_diff = 8.85009765625e-04`, `lse_max_abs_diff = 0.00591803`
- `c=256`: `bf16_max_abs_diff = 0.00405884`, `lse_max_abs_diff = 0.0242076`
- `c=448`: `bf16_max_abs_diff = 0.0160217`, `lse_max_abs_diff = 0.0794311`

That points to a combined prepared-scale handling issue in the debug score path, not an online-state bug and not a one-sided `Q_sc` or `K_sc` bug.

### Updated best next step

The next useful debugging branch should focus on how the debug path combines nontrivial `Q_sc` and `K_sc` when issuing QK:

- compare the prepared-scale staging/consumption in `kernel_quantize_p_from_scores_debug` against the `forward_persistent` path specifically for the case where both scale operands are active
- do not use `qk_quant_backend='localcta'` as a nonzero live-QK control unless the live prepared scales are verified nonzero
- treat `toy` as a separate payload-format experiment; the production-relevant issue remains the `v5` two-sided prepared-scale interaction

### Best next step

The stale suggestion to use `qk_quant_backend='localcta'` as the main live-QK control should be treated as obsolete in this tree, because that path now correctly raises when the folded live prepared scales underflow to all-zero on nonzero inputs.

The next debugging branch for the remaining mismatch should focus specifically on why `kernel_quantize_p_from_scores_debug` disagrees with `forward_persistent` on `v5` / `toy` live Q/K payloads:

- inspect whether the debug kernel’s Q/K score path assumes `localcta`/centric FP4 payload encoding
- compare the `v5` and `toy` payload interpretation against the production `forward_persistent` path
- avoid spending more time on the already-closed semaphore/reuse issue unless a new hang appears

## 2026-04-12 follow-up: raw score dump confirms the same two-sided `Q_sc * K_sc` split

I added a raw-score sidecar to the tile-window debug path:

- `dispatch_quantize_p_from_scores_tile_window_with_live_bf16_ref_debug(...)` now also returns `Scores_bf16_live`
- `_dump_live_quant_p_tile_window_and_bf16_ref_from_scores(...)` now exposes it as `debug["scores_bf16_live"]`

Important interpretation detail:

- `Scores_bf16_live` is stored in the same unscaled accumulator domain as `tile_score_max`
- to compare it against decoded BF16 reference scores, it must be multiplied by `q_sg * k_sg / sqrt(192)`

Concrete check on `S=256`, `seed=0`, `batch=heads=1`, `qk_quant_backend='v5'`, final row tile / final K tile (`tile_begin=1`):

- baseline `v5`, after applying the kernel scale factor to `scores_bf16_live` and comparing against decoded-FP4 BF16 scores:
  - `score_mean_abs_diff = 0.0820467`
  - `score_max_abs_diff = 0.376805`
  - `tile_score_max_mean_abs_diff = 0.0457449`
  - `tile_score_max_max_abs_diff = 0.275553`
- replace only `Q_sc` with ones:
  - `score_mean_abs_diff = 4.09016e-04`
  - `score_max_abs_diff = 0.00197795`
  - `tile_score_max_max_abs_diff = 0.00131644`
- replace only `K_sc` with ones:
  - `score_mean_abs_diff = 3.85033e-04`
  - `score_max_abs_diff = 0.00172784`
  - `tile_score_max_max_abs_diff = 0.00114023`
- replace both `Q_sc` and `K_sc` with ones:
  - `score_mean_abs_diff = 1.92227e-06`
  - `score_max_abs_diff = 8.97592e-06`
  - `tile_score_max_max_abs_diff = 6.10548e-06`

This is the main new result:

- the same split seen in final `P/LSE` output is already present in the raw score tile itself
- one-sided `Q_sc=1` or `K_sc=1` nearly fixes the score mismatch
- `Q_sc=K_sc=1` makes the raw score tile essentially exact

So the remaining `v5` bug is upstream of exponentiation, online normalization, and BF16/FP4 `P` storage. It is already present in the score-generation path when both nontrivial prepared-scale operands participate together.

One more useful cut on the same case:

- tile `0` baseline `v5`, scaled raw-score comparison:
  - `score_mean_abs_diff = 1.63684e-04`
  - `score_max_abs_diff = 0.00123236`
- tile `1` baseline `v5`, scaled raw-score comparison:
  - `score_mean_abs_diff = 0.0820467`
  - `score_max_abs_diff = 0.376805`

So on the short `S=256` case, the raw score path is already nearly exact on the first causal tile and then degrades sharply when the second causal tile is entered. That matches the earlier tile-history drift story, but now at the direct score level.

### Updated best next step

The next debugging cut should stay inside the QK score path and specifically compare how the debug kernel stages and consumes prepared `Q_sc/K_sc` for later causal tiles:

- inspect the scale-fragment layout and per-chunk pairing in `issue_qk_chunked_qsc_tmem(...)`
- compare the `v5` prepared-scale load/interpretation against the production `forward_persistent` path on the same input
- stop treating the residual issue as a softmax/update problem; the evidence now points to score generation before exponentiation

### Ruled-out follow-up

I also tested one concrete `K_sc`-layout hypothesis and reverted it:

- hypothesis: because live `K_sc` is depth-expanded with pairwise duplicates, the debug kernel should index `K_sc` by `2 * k_tile_idx` instead of `k_tile_idx`
- implementation test: patched both debug-kernel `k_sc_coord` load sites in `bf16_b300_mha_causal_fp4_pv_experiments.cu`, rebuilt, and reran the canonical `S=256`, seed `0`, final-tile `v5` comparison
- result: no improvement
  - scaled raw-score diff stayed essentially unchanged: `score_mean_abs_diff = 0.08349`, `score_max_abs_diff = 0.37649`
  - final-tile output diff also stayed in the same band: `p_max_abs_diff = 0.00283813`, `lse_max_abs_diff = 0.0160403`
- conclusion: the pairwise-expanded `K_sc` depth index is not the root cause, and that edit has been reverted

## 2026-04-12 follow-up: source truncation rules out a cross-chunk-only failure

I restored the CUDA build after a bad revert left `globals_quant_p_from_scores_debug` without its closing `;`. The current tree builds again:

- `make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments`
- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`

Then I ran the safer source-side truncation experiment instead of adding another kernel debug knob:

- generate BF16 source `Q/K/V` with `_make_live_bf16_source_inputs(...)`
- zero selected `Q/K` dimensions in Python
- repack through `_fp4_inputs_from_bf16_source(..., qk_quant_backend='v5')`
- dump tile-1 raw scores with `_dump_live_quant_p_tile_window_and_bf16_ref_from_scores(...)`
- compare `debug["scores_bf16_live"] * (q_sg * k_sg / sqrt(192))` against decoded live-FP4 BF16 scores from `_dequantize_live_fp4_qk_for_debug(...)`

Canonical case for all numbers below:

- `S=256`
- `seed=0`
- `batch=heads=1`
- final row tile / final K tile: `tile_begin=1`, `tiles_to_run=1`

### 64-d and 128-d truncation

Symmetric truncation on both `Q` and `K` still reproduces a large raw-score error even when only one 64-d chunk is active:

- full `0:192`:
  - `score_mean_abs_diff = 0.0820467`
  - `score_max_abs_diff = 0.376805`
- only `0:64`:
  - `score_mean_abs_diff = 0.0475035`
  - `score_max_abs_diff = 0.248037`
- only `0:128`:
  - `score_mean_abs_diff = 0.0675150`
  - `score_max_abs_diff = 0.328890`
- only `64:192`:
  - `score_mean_abs_diff = 0.0669601`
  - `score_max_abs_diff = 0.300979`
- only `64:128`:
  - `score_mean_abs_diff = 0.0478541`
  - `score_max_abs_diff = 0.264904`
- only `128:192`:
  - `score_mean_abs_diff = 0.0464442`
  - `score_max_abs_diff = 0.218600`

This rules out the narrower theory that only the later 64-d accumulation chunks are broken. A single active 64-d chunk is already sufficient.

### One-sided scale flattening still fixes the single-64 case

On the `0:64`-only case:

- baseline:
  - `score_mean_abs_diff = 0.0475035`
  - `score_max_abs_diff = 0.248037`
- `Q_sc = 1`, original `K_sc`:
  - `score_mean_abs_diff = 2.33678e-04`
  - `score_max_abs_diff = 0.00116497`
- original `Q_sc`, `K_sc = 1`:
  - `score_mean_abs_diff = 2.14474e-04`
  - `score_max_abs_diff = 0.00106148`
- `Q_sc = K_sc = 1`:
  - `score_mean_abs_diff = 1.05621e-06`
  - `score_max_abs_diff = 5.11242e-06`

So the earlier one-sided-fix result is not a multi-chunk artifact. The bad interaction survives with only one 64-d chunk active and still disappears when either prepared-scale operand is flattened.

### 16-d group isolation

I then isolated single 16-d groups inside the first 64-d chunk:

- only `0:16`:
  - `score_mean_abs_diff = 0.0223927`
  - `score_max_abs_diff = 0.145968`
- only `16:32`:
  - `score_mean_abs_diff = 0.0221580`
  - `score_max_abs_diff = 0.143675`
- only `32:48`:
  - `score_mean_abs_diff = 0.0239934`
  - `score_max_abs_diff = 0.155335`
- only `48:64`:
  - `score_mean_abs_diff = 0.0224763`
  - `score_max_abs_diff = 0.159200`

This is the tightest boundary so far: the mismatch already exists with a single active 16-wide prepared-scale group.

Representative one-sided check on the `0:16` group:

- baseline:
  - `score_mean_abs_diff = 0.0223927`
  - `score_max_abs_diff = 0.145968`
- `Q_sc = 1`:
  - `score_mean_abs_diff = 9.72182e-05`
  - `score_max_abs_diff = 7.93325e-04`
- `K_sc = 1`:
  - `score_mean_abs_diff = 9.51142e-05`
  - `score_max_abs_diff = 5.48150e-04`
- `Q_sc = K_sc = 1`:
  - `score_mean_abs_diff = 4.12839e-07`
  - `score_max_abs_diff = 2.17298e-06`

### Updated conclusion

The remaining debug-kernel score bug is not specific to:

- later causal tiles alone
- later 64-d accumulation chunks alone
- cross-chunk accumulation across multiple 64-d chunks

The smallest reproducer I have now is:

- one final-tile score dump
- one active 16-wide `Q/K` subgroup
- nontrivial prepared `Q_sc` and `K_sc` on both sides

Flattening either prepared-scale operand still nearly eliminates the error. So the next debugging step should compare the per-16-group prepared-scale fragment interpretation in `kernel_quantize_p_from_scores_debug` against `forward_persistent`, rather than adding more coarse chunk-level probes.

## 2026-04-12 follow-up: bypassing `q_sc` TMEM staging does not change the bug

I added a debug-only `q_scale_mode` switch to `_dump_live_quant_p_tile_window_and_bf16_ref_from_scores(...)` / `dispatch_quantize_p_from_scores_tile_window_with_live_bf16_ref_debug(...)`:

- `q_scale_mode='tmem'`: existing path, stage `q_sc` once with `stage_q_scale_tmem(...)` and use `issue_qk_chunked_qsc_tmem(...)`
- `q_scale_mode='smem_chunked'`: bypass the staged full-`q_sc` TMEM path and call `issue_qk_chunked(...)` directly from `q_sc_smem` each chunk

This changes only the experiments/debug score-dump path.

Canonical comparison results:

- full `0:192`, `tile_begin=1`
  - `tmem`: `score_mean_abs_diff = 0.0820467`, `score_max_abs_diff = 0.376805`
  - `smem_chunked`: `score_mean_abs_diff = 0.0820467`, `score_max_abs_diff = 0.376805`
- single-group `0:16`, `tile_begin=1`
  - `tmem`: `score_mean_abs_diff = 0.0223927`, `score_max_abs_diff = 0.145968`
  - `smem_chunked`: `score_mean_abs_diff = 0.0223927`, `score_max_abs_diff = 0.145968`

So the residual mismatch is not caused by the debug kernel’s one-time `q_sc` TMEM staging path. The bug survives unchanged when `q_sc` is consumed directly from `q_sc_smem` chunk-by-chunk.

This narrows the remaining common path further:

- `issue_qk_chunked(...)` / `issue_qk_chunked_qsc_tmem(...)`
- `load_k_scale_chunk(...)`
- the common `tcgen05::st_st(...)` score-generation path when both prepared-scale operands are nontrivial

It is no longer reasonable to suspect `stage_q_scale_tmem(...)` or the full-`QScTTFull` wrapper itself.

## 2026-04-12 follow-up: the debug dump picks K-scale rows by Q tile, and fixing that alone does not fix accuracy

I added two more debug-only knobs to the tile-window score dump:

- `stage_cut` now also accepts `legacy_single_slot_full` for the full tile-window dump
- `k_scale_depth_mode` switches the debug kernel’s `K_sc` depth index between:
  - `tile128`: current behavior, use the 128-tile-based `k_tile_idx`
  - `slab64`: use the raw 64-slab depth index `idx * C::CLUSTER_SIZE + cta_rank`

I also added a `tmem_layout_mode` A/B:

- `compact`: the existing debug TMEM layout
- `production_like`: move `q_sc_tm/k_sc_tm` to the same TMEM offsets used by the production persistent kernel

### Negative cuts

On the canonical `S=256`, `seed=0`, final-row-tile score dump (`tile_begin=1`):

- `stage_cut=full`, `legacy_single_slot_full`, and `dual_slot_full` are identical
  - `score_mean_abs_diff = 0.0820467`
  - `score_max_abs_diff = 0.376805`
- `tmem_layout_mode=compact` and `production_like` are identical
  - `score_mean_abs_diff = 0.0820467`
  - `score_max_abs_diff = 0.376805`
- `q_scale_mode=tmem` and `smem_chunked` are identical
  - same result as above

So the remaining score mismatch is not explained by:

- the dual-slot versus legacy single-slot score-publication discipline on this short non-hanging case
- the compact debug TMEM placement of `q_sc_tm/k_sc_tm`
- the one-time `q_sc` TMEM staging path

### Positive localization: which expanded `K_sc` row each window actually uses

Using the canonical `v5` `S=256` input and zeroing individual expanded `K_sc` depth rows:

- debug score dump, early Q tile (`rows 0:127`), `tile_begin=0`
  - only `K_sc` row `0` matters
- debug score dump, final Q tile (`rows 128:255`), `tile_begin=0`
  - only `K_sc` row `1` matters
- debug score dump, final Q tile (`rows 128:255`), `tile_begin=1`
  - current `tile128` mode: only `K_sc` row `1` matters

This is the important new structural result:

- in the current debug dump, the final Q tile uses `K_sc` row `1` for both 128-col windows
- so the later K window is not selecting a distinct expanded `K_sc` depth row at all in the current path

Production-side control with `softmax_store_p(...)` on the same input is different:

- `rows 0:127, cols 0:127`: only `K_sc` row `0` matters
- `rows 128:255, cols 0:127`: mainly `K_sc` row `1` matters
- `rows 128:255, cols 128:255`: mainly `K_sc` row `3` matters

So the production final output path does distinguish the later window in a way the current debug score dump does not.

### Slab-64 remap result

Switching the debug dump to `k_scale_depth_mode='slab64'` changes the final-window sensitivity exactly as expected:

- final Q tile, `tile_begin=1`
  - sensitivity moves from `K_sc` row `1` to `K_sc` row `3`

But it does **not** improve accuracy:

- `tile_begin=1`, `tile128`
  - `score_mean_abs_diff = 0.0820467`
  - `score_max_abs_diff = 0.376805`
- `tile_begin=1`, `slab64`
  - `score_mean_abs_diff = 0.0834911`
  - `score_max_abs_diff = 0.376492`

So correcting the obvious later-window `K_sc` row selection in isolation is not sufficient. It changes which expanded scale row is consulted, but the large score mismatch remains.

### Updated conclusion

The remaining bug is narrower now:

- the debug score dump’s current `K_sc` depth selection for later windows is structurally wrong relative to production behavior
- but that is not the only issue, because forcing the later window onto the production-like `K_sc` row still leaves the same large score error

The next useful cut should compare the full per-window `K_sc` fragment consumed by the debug score dump against what the production persistent kernel effectively uses for the same final Q tile, rather than continuing with coarse address/layout toggles.

## 2026-04-12 follow-up: the tile-window producer consumes the exact local `K_sc` tile selected by its own depth rule

I added a debug-only `K_sc` sidecar to the tile-window dump so the producer can copy the exact `slot_tile_at(k_sc_smem, k_sc_slot)` bytes it consumes into a host-visible tensor. The sidecar is observational only: the canonical `S=256`, `tile_begin=1` raw-score mismatch is unchanged after the patch.

Canonical recheck, `random_live_fp4`, `seed=0`, final row tile:

- `k_scale_depth_mode='tile128'`
  - `score_mean_abs_diff = 0.08204665780067444`
  - `score_max_abs_diff = 0.37680473923683167`
- `k_scale_depth_mode='slab64'`
  - `score_mean_abs_diff = 0.08349110186100006`
  - `score_max_abs_diff = 0.37649160623550415`

The new sidecar shows the local producer is consuming the exact `K_sc` tensor slice predicted by the current depth rule:

- `tile_begin=1`, `k_scale_depth_mode='tile128'`
  - captured `debug["k_sc_loaded"]` is byte-exact to `K_sc[:, depth=0/1, head_rows, :]`
  - on the canonical `v5` input, depths `0` and `1` are identical because live `K_sc` is repeat-interleaved in pairs
- `tile_begin=1`, `k_scale_depth_mode='slab64'`
  - captured `debug["k_sc_loaded"]` is byte-exact to `K_sc[:, depth=2/3, head_rows, :]`
  - depths `2` and `3` are likewise identical paired repeats

So the new boundary is tighter:

- the tile-window debug producer is not fabricating or corrupting the local `K_sc` tile after selection
- the `tile128 -> first repeated pair` and `slab64 -> second repeated pair` selection behavior is real and host-visible
- changing which repeated `K_sc` pair is selected still does not fix the raw-score mismatch

That eliminates another branch: the remaining bug is not in the local `K_sc` TMA load or the local `K_sc` shared-memory tile contents. The next useful cut is upstream/downstream of that exact tile, most likely:

- compare the consumed `Q_sc` fragment the same way, or
- probe how the correct `Q_sc`/`K_sc` tiles are combined inside the `st_st` / score-read path, since the loaded `K_sc` bytes are now confirmed exact.

## 2026-04-12 follow-up: the tile-window producer stages only the cluster-base `Q_sc` tile

I added the symmetric `Q_sc` sidecar to the tile-window dump so the producer can copy the exact `slot_tile_at(q_sc_smem, 0)` bytes it stages before issuing `st_st(...)`.

Canonical case, `S=256`, `seed=0`, `random_live_fp4`, `tile_begin=1`, `heads=1`:

- input `Q_sc` tile `0` and tile `1` are materially different
  - `max_abs_diff = 344.0`
  - `mean_abs_diff = 52.484375`
- captured `debug["q_sc_loaded"][tile=0]` is byte-exact to input `Q_sc` tile `0`
- captured `debug["q_sc_loaded"][tile=1]` remains zero, because the issue lane only stages one `Q_sc` tile per cluster/task

So the shared debug score kernel really is staging the cluster-base `Q_sc` tile only. That is a concrete structural bug in the debug path for the second row tile of each cluster.

I also tried the obvious confirmation experiment:

- force input `Q_sc[tile1] = Q_sc[tile0]`
- rerun the canonical final-window raw-score compare

That does **not** fix the mismatch by itself:

- baseline, `tile128`
  - `score_mean_abs_diff = 0.08204665780067444`
  - `score_max_abs_diff = 0.37680473923683167`
- with `Q_sc[tile1] = Q_sc[tile0]`, `tile128`
  - `score_mean_abs_diff = 0.08663640171289444`
  - `score_max_abs_diff = 0.4587453305721283`
- with `Q_sc[tile1] = Q_sc[tile0]`, `slab64`
  - `score_mean_abs_diff = 0.08814236521720886`
  - `score_max_abs_diff = 0.4603171944618225`

So the new state is:

- the debug kernel definitely stages the wrong `Q_sc` tile for the second row tile in a cluster
- the debug kernel definitely selects the wrong repeated `K_sc` pair under the default `tile128` rule for later windows
- fixing either one in isolation is not sufficient to restore score accuracy

The remaining high-value branch is now the combination step itself: either the debug kernel needs both the correct per-row `Q_sc` and the correct per-window `K_sc` staged together, or there is still a deeper issue in how those otherwise-correct fragments are consumed inside the shared `st_st` / score-read path.

## 2026-04-12 follow-up: the `Q_sc` sidecar was a preload observation, not proof that the final row tile uses the wrong `Q_sc`

The `Q_sc` sidecar I added in the previous step only captures the cluster-base preload observation from CTA0. It does **not** prove that the final row tile actually consumes `Q_sc[tile0]`.

Two behavioural checks correct the interpretation:

- On the canonical `S=256`, `seed=0`, `tile_begin=1`, `random_live_fp4` case, zeroing `Q_sc[tile0]` causes **no** change in the debug final-window raw scores.
- Zeroing `Q_sc[tile1]` causes a huge change in the debug final-window raw scores.
- Production `softmax_store_p(...)` shows the same row-tile sensitivity pattern on the final tile:
  - zeroing `Q_sc[tile0]`: no effect on final-window `P`/`LSE`
  - zeroing `Q_sc[tile1]`: clear effect on final-window `P`/`LSE`

Concrete numbers:

- debug final-window raw-score sensitivity
  - zero `Q_sc[tile0]`: `mean_abs_delta = 0.0`, `max_abs_delta = 0.0`
  - zero `Q_sc[tile1]`: `mean_abs_delta = 4090566.0`, `max_abs_delta = 19267584.0`
- production final-window sensitivity
  - zero `Q_sc[tile0]`: `P_mean_abs_delta = 0.0`, `P_max_abs_delta = 0.0`, `LSE_max_abs_delta = 0.0`
  - zero `Q_sc[tile1]`: `P_mean_abs_delta = 0.00013971049338579178`, `P_max_abs_delta = 0.002044677734375`, `LSE_max_abs_delta = 0.014063835144042969`

So the earlier “debug path stages only the cluster-base `Q_sc` tile, therefore the final row tile uses the wrong `Q_sc`” conclusion was too strong. The sidecar is still useful as a preload observation, but it is not a faithful record of the final row tile’s effective `Q_sc` consumption.

## 2026-04-12 follow-up: the remaining score mismatch has a strong permutation component

I ran a row-wise sorted-score check on the canonical `S=256`, `tile_begin=1`, `random_live_fp4`, `seed=0` case:

- direct row-wise score error
  - average row mean abs diff = `0.083628810942173`
  - worst row max abs diff = `0.37680473923683167`
- after sorting each row’s finite causal scores before comparison
  - average row mean abs diff = `0.022244413194130175`
  - worst row max abs diff = `0.19806595146656036`

So sorting reduces the error substantially but not to zero. That means the remaining problem is not just a scalar-scale mismatch:

- there is a large score-layout / permutation component in the debug raw-score mismatch
- but there is also a residual value mismatch after permutation is factored out

That shifts the next useful cut again. The most promising target is now the score-read / score-layout path itself around the `tcgen05.ld` extraction and quarter/block mapping, rather than continuing to focus only on which `Q_sc` or `K_sc` tile was loaded.

## 2026-04-12 follow-up: the readout path is not the source of the remaining raw-score mismatch

I added two more raw-score sidecars to the tile-window debug path:

- `debug["scores_bf16_scan_repacked"]`
  - same `scores_reg` values, round-tripped through `store_scores_quarter_to_localcta_scan(...)` and read back row-major from `fp4pv_p_scan3d`
- `debug["scores_bf16_tmem_x16"]`
  - same TT score tile, but read directly from TMEM with `tcgen05.ld.sync.aligned.32x32b.x16.b32` instead of the existing `x32` path

Build and syntax checks still pass:

- `make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments`
- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`

Canonical check, `S=256`, `seed=0`, `random_live_fp4`, `tile_begin=1`, final row tile, compared against the decoded-live-FP4 BF16 QK reference:

- existing raw sidecar (`scores_bf16_live`)
  - `mean_abs_diff = 0.08204665780067444`
  - `max_abs_diff = 0.37680473923683167`
- scan-repacked sidecar (`scores_bf16_scan_repacked`)
  - `mean_abs_diff = 0.08204665780067444`
  - `max_abs_diff = 0.37680473923683167`
- direct `x16` TMEM sidecar (`scores_bf16_tmem_x16`)
  - `mean_abs_diff = 0.08204665780067444`
  - `max_abs_diff = 0.37680473923683167`
- sidecar-to-sidecar equality
  - `scores_bf16_live == scores_bf16_scan_repacked` exactly
  - `scores_bf16_live == scores_bf16_tmem_x16` exactly

So the remaining mismatch is **not** introduced by:

- the quarter-scan repack / readback path
- the `tcgen05.ld ... x32` score load itself versus an `x16` readout

The wrong values or wrong ordering are already present in the TT score contents before those readout choices diverge.

I also tightened the permutation result with block-local sorting on the same canonical case. Sorting only within contiguous finite-score blocks already reduces the error monotonically:

- block size `1`
  - `mean_abs_diff = 0.08362880989443511`
  - `max_abs_diff = 0.37680473923683167`
- block size `2`
  - `mean_abs_diff = 0.0693410421081353`
- block size `4`
  - `mean_abs_diff = 0.057585334609029815`
- block size `8`
  - `mean_abs_diff = 0.04376456841418985`
- block size `16`
  - `mean_abs_diff = 0.034879587750765495`
  - `max_abs_diff = 0.2006760537624359`
- block size `32`
  - `mean_abs_diff = 0.02785093274724204`
  - `max_abs_diff = 0.19806595146656036`
- block size `64`
  - `mean_abs_diff = 0.02401211859250907`
- block size `128`
  - `mean_abs_diff = 0.022244412706641015`
  - `max_abs_diff = 0.19806595146656036`

Interpretation:

- a large part of the mismatch is still a permutation / ordering problem
- that permutation is already local at small contiguous fragment scales, not only a whole-row or quarter swap
- because `x32`, scan-repacked, and direct `x16` are identical, the likely source moved earlier again: the TT score production / fragment-placement path, not the consumer-side readout

The next useful branch is to inspect the TT score producer itself in the debug path, especially how the later-window `st_st(...)` / fragment-placement path combines nontrivial `Q_sc` and `K_sc` when both are active.

## 2026-04-12 follow-up: the mismatch is already present in a single 64-wide QK chunk

I added a debug-only `qk_chunk_mask` to the tile-window BF16-ref dump so the producer can issue only selected `st_st(...)` chunks into the TT score tile. This stays in the experiments/debug path only.

Files:

- [bf16_b300_mha_causal_fp4_pv_experiments.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu)
- [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py)

Build and syntax checks passed after the change:

- `make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments`
- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`

Canonical case: `S=256`, `seed=0`, `random_live_fp4`, `tile_begin=1`, final row tile, compared against a decoded-live-FP4 BF16 reference restricted to the same chunk mask.

Chunk-mask sweep:

- mask `001`
  - `score_mean_abs_diff = 0.047295909374952316`
  - `score_max_abs_diff = 0.22985634207725525`
- mask `010`
  - `score_mean_abs_diff = 0.048967376351356506`
  - `score_max_abs_diff = 0.273945152759552`
- mask `100`
  - `score_mean_abs_diff = 0.0471796877682209`
  - `score_max_abs_diff = 0.21505647897720337`
- mask `011`
  - `score_mean_abs_diff = 0.06751501560211182`
  - `score_max_abs_diff = 0.32888978719711304`
- mask `101`
  - `score_mean_abs_diff = 0.06633483618497849`
  - `score_max_abs_diff = 0.3503929376602173`
- mask `110`
  - `score_mean_abs_diff = 0.06816761940717697`
  - `score_max_abs_diff = 0.3188328742980957`
- mask `111`
  - `score_mean_abs_diff = 0.08204665780067444`
  - `score_max_abs_diff = 0.37680473923683167`

So the remaining raw-score mismatch does **not** require multi-chunk accumulation. A single 64-wide chunk is already wrong.

I then repeated the check for representative masks `001` and `111` while neutralizing one prepared-scale operand at a time:

- mask `001`
  - baseline: `mean_abs_diff = 0.047295909374952316`
  - `Q_sc = 1`: `mean_abs_diff = 0.00023312294797506183`
  - `K_sc = 1`: `mean_abs_diff = 0.00022224726853892207`
  - `Q_sc = K_sc = 1`: `mean_abs_diff = 1.0959726068904274e-06`
- mask `111`
  - baseline: `mean_abs_diff = 0.08204665780067444`
  - `Q_sc = 1`: `mean_abs_diff = 0.0004090155125595629`
  - `K_sc = 1`: `mean_abs_diff = 0.00038503282121382654`
  - `Q_sc = K_sc = 1`: `mean_abs_diff = 1.922274577736971e-06`

Interpretation:

- the bug is already present at single-chunk granularity
- the earlier “two-sided prepared-scale interaction” diagnosis still holds at that finer granularity
- multi-chunk accumulation makes the total error larger, but it is not the root cause

That pushes the next debugging step one level deeper into the producer-side scale/fragment path for a single `st_st(...)` chunk, rather than into later chunk accumulation or consumer readout.

## 2026-04-12 follow-up: single-chunk error is independent of Q-scale staging mode and TMEM layout

I used the new `qk_chunk_mask` on the canonical bad case to rerun the single-chunk `001` issue under the previously tested Q-scale staging/layout variants:

- `q_scale_mode='tmem', tmem_layout_mode='compact'`
- `q_scale_mode='smem_chunked', tmem_layout_mode='compact'`
- `q_scale_mode='tmem', tmem_layout_mode='production_like'`
- `q_scale_mode='smem_chunked', tmem_layout_mode='production_like'`

Case:

- `S=256`
- `seed=0`
- `random_live_fp4`
- `tile_begin=1`
- `qk_chunk_mask=0b001`

All four variants are bit-identical on the raw score dump and have the same error vs decoded-live-FP4 BF16 reference:

- `score_mean_abs_diff = 0.047295909374952316`
- `score_max_abs_diff = 0.22985634207725525`

So the remaining single-chunk bug is not in:

- one-time `Q_sc` TMEM staging
- the `smem_chunked` versus `QScTTFull` path
- compact versus production-like TMEM placement for the `Q_sc` / `K_sc` TT tiles

That shrinks the single-chunk suspect set further. The remaining likely fault is now in the shared K-scale / `st_st(...)` chunk issue path itself, not in consumer readout and not in the Q-scale staging/layout variants.

## 2026-04-12 follow-up: TT scale readback sidecar is intrusive, and the short-case `K_sc` depth rule matches production

I tried to add a debug sidecar that dumps the TT-loaded `Q_sc` / `K_sc` chunks back to host from the tile-window score dump. The intent was to compare the SMEM source chunks against the actual TT-loaded chunks on the canonical bad case:

- `S=256`
- `seed=0`
- `random_live_fp4`
- `tile_begin=1`
- `qk_chunk_mask=0b001`

Two useful results came out of this, even though the sidecar itself is not yet usable:

1. The first implementation was invalid because it read back a `128x32` TT debug tile after only writing its left `128x16` subtile.
2. After fixing that readback bug and adding the missing `tensor_before_thread_sync()` before the warp consumed the loaded registers, the shared tile-window helper still hung on the same short case as soon as the TT scale readback sidecar was enabled.

Because that sidecar was intrusive enough to break an otherwise healthy short debug case, I disabled its runtime capture path again in the shared tile-window helper:

- `debug["q_sc_tmem_loaded"]` is now `None`
- `debug["k_sc_tmem_loaded"]` is now `None`
- the standard tile-window score dump returns again on `cuda:3`

Build / syntax after that rollback-to-null-capture step:

- `make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments`
- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`

Health check after disabling TT capture:

- the canonical tile-window dump returns on `cuda:3`
- `scores_bf16_live.shape == (1, 1, 256, 128)`
- `q_sc_tmem_loaded is None`
- `k_sc_tmem_loaded is None`

Separately, I re-read the production `forward_persistent(...)` path in `b300_causal/bf16_b300_mha_causal_fp4.cu` and corrected one earlier hypothesis:

- on the short `S=256` case, production also uses `k_sc_coord.depth = k_tile_idx`, with `k_tile_idx = min(idx * CLUSTER_SIZE + cta_rank, tiles_m - 1)`
- so the debug kernel’s default `tile128` `K_sc` depth rule matches production on this short case
- the earlier suspicion that production should be using the later repeated live-`K_sc` pair (`depth 2/3`) for `tile_begin=1` was wrong for `softmax_store_p(...)`

So the remaining likely culprit set is narrower again:

- not Q-scale staging mode
- not compact versus production-like TMEM base placement
- not multi-chunk accumulation
- not the short-case default `K_sc` depth rule

The next useful debugging step is now a **separate isolated TT scale-dump kernel or dispatch**, rather than trying to piggyback TT readback onto the shared tile-window score helper. The current shared helper is safe to use again for score comparisons, but not for TT scale introspection.

## 2026-04-12 update: isolated scale-chunk probe stage cuts

I continued on the separate isolated `probe_loaded_qk_scale_chunks_debug(...)` path and added stage cuts so the scale-dump kernel can stop at specific milestones instead of only “full”.

Current stage names in the Python wrapper:

- `source_only`
- `q_load_sync`
- `q_rt_load_only`
- `q_copy_shared_only`
- `q_load_copy`
- `k_load_sync`
- `k_load_copy`
- `full`

Canonical repro case for all runs below:

- `device='cuda:3'`
- `seqlen=256`
- `seed=0`
- `q_tile_idx=1`
- `k_sc_depth_idx=1`
- `chunk_idx=0`
- input mode from `make_random_live_fp4_attention_inputs(...)`

Verified-good stages:

- `source_only`: clean
- `q_load_sync`: clean
- `q_rt_load_only`: clean

Verified-bad stages:

- `q_copy_shared_only`: illegal memory access
- `q_load_copy`: illegal memory access
- `k_load_sync`: illegal memory access, but only after the failing Q-copy stage
- `k_load_copy`: illegal memory access
- `full`: illegal memory access

What this means:

- the isolated raw source chunk copy from SMEM is fine
- `load_q_scale_chunk(...)` into TT is fine
- TT subtile readback into a `rt_fp8e4m3<32,32>` register tile is also fine
- the first bad step is trying to store that `32x32` register subtile back into shared memory

I tested two versions of that copy-back step:

1. FP8 path:
   - `rt_fp8e4m3<32,32>` -> `st_fp8e4m3<32,32,false>`
2. BF16-converted path:
   - `rt_fp8e4m3<32,32>` -> `rt_bf<32,32>` -> `st_bf<32,32>`

Both fail at the same `q_copy_shared_only` stage boundary. So the scale load itself is not the problem; the bad operation is broader: in this isolated probe, `kittens::group<1>::store(...)` of the `32x32` register subtile back into shared is faulting regardless of whether the register tile is FP8 or BF16-converted.

Build / syntax after adding the stage cuts and the BF16-copy variant:

- `make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments`
- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`

Implication for the main debug question:

- this does **not** implicate `load_q_scale_chunk(...)` / `load_k_scale_chunk(...)`
- it also does **not** implicate TT readback into registers
- it only blocks this isolated introspection helper from dumping the loaded TT scale contents through a shared-memory staging buffer

So the next useful step is no longer “is the scale load corrupting data?”. The next useful step is to build a **direct register-to-global dump** for the loaded `32x32` TT subtile, or find an existing safe non-shared store path for these register tiles. That is the remaining blocker to comparing the actually loaded TT scale fragment against the expected repeated `(32,16)` source chunk.

### Follow-up: direct register-to-global dump works

I then added a debug-only direct-dump path that bypasses shared-memory staging entirely:

- stage modes added:
  - `q_direct_dump`
  - `full_direct_dump`
- implementation:
  - load the `32x32` TT subtile into `rt_fp8e4m3<32,32>`
  - convert to `rt_bf<32,32>`
  - write the left `32x16` half directly to global with a manual row/col mapping

This bypass is important because the earlier `q_copy_shared_only` / `q_load_copy` failure is specifically in the debug extraction path’s register-to-shared store, not in the TT scale load itself.

Canonical short-case result, `seqlen=256, seed=0, q_tile_idx=1, k_sc_depth_idx=1, chunk_idx=0`:

- `q_direct_dump`: clean
- `full_direct_dump`: clean after fixing the control flow so it skips the old shared-copy path

Comparison of the dumped TT scale contents against the naive repeated source chunk:

- `q_sc_tmem_vs_repeat32_mean_abs_diff = 47.65625`
- `q_sc_tmem_vs_repeat32_max_abs_diff = 272.0`
- `k_sc_tmem_vs_repeat32_mean_abs_diff = 44.09375`
- `k_sc_tmem_vs_repeat32_max_abs_diff = 272.0`

Chunk sweep on the same case:

- chunk 0:
  - `Q mean/max = 47.65625 / 272.0`
  - `K mean/max = 44.09375 / 272.0`
- chunk 1:
  - `Q mean/max = 41.109375 / 320.0`
  - `K mean/max = 44.8125 / 272.0`
- chunk 2:
  - `Q mean/max = 48.34375 / 320.0`
  - `K mean/max = 43.65625 / 272.0`

Early-tile check:

- `q_tile_idx=0, k_sc_depth_idx=0, chunk_idx=0` shows the same kind of mismatch
- so this is not a late-window-only selection problem; it appears to be present in the raw scale load / interpretation path in general

Permutation sanity on the canonical chunk-0 dump:

- row-wise sorting reduces the error, but does not remove it
- `Q row-sort mean/max = 14.28125 / 192.0`
- `K row-sort mean/max = 14.25 / 192.0`

Interpretation:

- the direct dump proves that the isolated TT scale-load path can now be inspected without touching the broken shared-store introspection path
- the dumped loaded-scale fragment is not equal to the naive repeated source chunk for either `Q_sc` or `K_sc`
- however, this still leaves one ambiguity:
  - either `load_q_scale_chunk(...)` / `load_k_scale_chunk(...)` is materially wrong
  - or the generic TT-to-register readback used by the direct dump is not interpreting the TT scale tile in the same logical order that `tcgen05::st_st(...)` expects

So the next useful step is now a control experiment on the **TT interpretation**, not more work on the shared-store sidecar:

- compare this direct TT dump against a known-good minimal `load_mxnv_scale_async2(...)` use case, or
- derive the exact logical layout expected by `tcgen05::st_st(...)` for these scale TT tiles and test the dump against that layout instead of against naive `repeat(4, 1)`

### Correction: first-pass direct dump was using an invalid FP8->BF16 tile conversion

I found a bug in that first direct-dump attempt:

- converting `rt_fp8e4m3<32,32>` to `rt_bf<32,32>` with the generic `rt::operator=` is not a valid elementwise reinterpretation here
- the packed-tile widths differ (`fp8` tile width 1 base tile vs `bf16` tile width 2 base tiles for the same `32x32` logical size), so that path was not trustworthy

I replaced that with explicit FP8x4 unpacking for the direct dump. After that correction:

- `q_copy_shared_only` still fails with illegal memory access using the restored original FP8 shared-copy path
- `q_direct_dump` is clean
- `full_direct_dump` is clean

However, the direct-dump **mapping** is still not correct yet. A synthetic control with a simple `4x4` block-ID pattern in the source `(32,16)` chunk shows that the current direct unpack logic is reading the TT register tile in the wrong logical row/col order.

Synthetic control:

- source rows `0:4`:
  - `[1 1 1 1 | 2 2 2 2 | 3 3 3 3 | 4 4 4 4]`
- dumped rows `0:4`:
  - `[1 1 1 1 | 9 9 9 9 | 1 1 1 1 | 9 9 9 9]`
  - `[2 2 2 2 | 10 10 10 10 | 2 2 2 2 | 10 10 10 10]`
  - `[3 3 3 3 | 11 11 11 11 | 3 3 3 3 | 11 11 11 11]`
  - `[4 4 4 4 | 12 12 12 12 | 4 4 4 4 | 12 12 12 12]`

So the current direct dump is still mixing data from different source row blocks. That means the earlier `repeat(4, 1)` mismatch numbers from the first direct-dump attempt are **not** reliable evidence about whether `load_q_scale_chunk(...)` / `load_k_scale_chunk(...)` itself is wrong.

Current reliable state:

- reliable:
  - `source_only` clean
  - `q_load_sync` clean
  - `q_rt_load_only` clean
  - `q_copy_shared_only` illegal memory access on the original FP8 shared-copy path
  - `full_direct_dump` runs without fault
- not yet reliable:
  - interpreting the direct TT dump as logical `(row, col)` scale data
  - comparing that dump directly against naive `repeat(4, 1)`

So the next step is now very specific:

- derive the correct logical mapping of `rt_fp8e4m3<32,32>` for this TT scale tile, instead of assuming a contiguous row-major unpack, then rerun the direct-dump comparison under that mapping

## 2026-04-12 update: raw packed-word TT decoder is now validated

I added a raw-register sidecar to the isolated scale-chunk probe and stopped using the BF16 direct dump as the correctness oracle.

New probe outputs:

- `q_sc_tmem_raw_words`
- `k_sc_tmem_raw_words`

Shape per side:

- `(4, 32, 2, 4)` = `row_block x lane x rt_height x packed_word`

The Python decoder now reconstructs the logical `128x32` TT matrix from those raw packed words using the actual ThunderKittens FP8 row-tile structure:

- `TILE_ROW_DIM(fp8) = 16`
- `TILE_COL_DIM(fp8) = 32`
- `rt_fp8e4m3<32,32>` therefore has `height = 2`, `width = 1`, `packed_per_tile = 4`
- decoded indices:
  - `row = subtile_row * 16 + (packed_word % 2) * 8 + lane // 4`
  - `col = (packed_word // 2) * 16 + (lane % 4) * 4 + elem`

Synthetic validation on `cuda:1` is exact for all tested source patterns:

- `unique_rows`: `Q/K mean,max = 0.0 / 0.0`
- `unique_col_groups`: `Q/K mean,max = 0.0 / 0.0`
- `unique_blocks8x8`: `Q/K mean,max = 0.0 / 0.0`

That replaces the earlier incorrect “mixed row-block” interpretation from the first BF16 direct dump attempt.

Canonical real-data recheck, `S=256`, `seed=0`, `qk_quant_backend='v5'`, `q_tile_idx=1`, `k_sc_depth_idx=1`, `chunk_idx=0`, `stage_mode='full_direct_dump'`, on `cuda:1`:

- decoded `Q_sc` vs expected `repeat(4, 1)` left half: mean/max `0.0 / 0.0`
- decoded `K_sc` vs expected `repeat(4, 1)` left half: mean/max `0.0 / 0.0`
- decoded `Q_sc` vs expected full `cat([source, source], dim=1).repeat(4, 1)`: mean/max `0.0 / 0.0`
- decoded `K_sc` vs expected full `cat([source, source], dim=1).repeat(4, 1)`: mean/max `0.0 / 0.0`

So the isolated `load_q_scale_chunk(...)` / `load_k_scale_chunk(...)` path is now behaving exactly as expected under a validated TT decode. The earlier “raw scale load mismatch” conclusion was a decode bug, not a scale-load bug.

Stage-mode sanity after the raw sidecar change, on `cuda:1`:

- clean:
  - `source_only`
  - `q_load_sync`
  - `q_rt_load_only`
  - `q_direct_dump`
  - `full_direct_dump`
- still faulting:
  - `q_copy_shared_only` with illegal memory access on the old FP8 shared-store sidecar path

Current conclusion:

- the scale-load path itself is not the remaining issue
- the failing shared-memory introspection path is still broken, but it is now out of the critical correctness path
- the remaining prepared-scale bug is downstream of the validated TT scale materialization

## 2026-04-13 update: default tile-window quant dump now uses the stable production path

Practical fix applied in Python:

- `_dump_live_quant_p_tile_window_and_bf16_ref_from_scores(...)` now routes the default/common case through:
  - `softmax_store_p(...)`
  - `quantize_p(...)`
- The legacy extension-backed debug score path is still available, but only when a caller requests nondefault debug knobs:
  - non-`full` `stage_cut`
  - non-`tmem` `q_scale_mode`
  - non-`compact` TMEM layout
  - non-`tile128` K-scale depth mode
  - masked chunk subsets

This means the normal comparison/measurement entrypoints no longer depend on the known-bad debug score kernel.

Verification:

- in-process `measure_live_quant_tile_window_against_ref(...)`
  - `S=256`, `tile_begin=1`, `tiles_to_run=1`
  - `zero_qk_random_v`: all diffs exactly `0.0`
  - `random_live_fp4`: all diffs exactly `0.0`
  - returned debug metadata: `path = "softmax_store_p_reference"`

- subprocess `measure_live_quant_tile_window_against_ref_subprocess(...)`
  - `S=256`, `random_live_fp4`: all diffs exactly `0.0`
  - `S=2048`, `random_live_fp4`, final tile: all diffs exactly `0.0`

One additional fix was needed for this path:

- direct slicing with `.contiguous()` on `float4_e2m1fn_x2` payloads is not implemented in PyTorch here
- window extraction now uses `_zeros_fp4(...)` + `_copy_fp4_tensor_(...)` instead of relying on unsupported `float4` copy kernels

Current practical state:

- the common tile-window comparison tools now run cleanly and agree exactly with the production-stable reference path
- the low-level legacy debug score kernel still exists for explicit forensic use, but it is no longer on the default path

## 2026-04-13 update: broader live-quant helpers now default to the stable path too

Follow-on pragmatic fixes in Python:

- `_dump_live_quant_p_from_scores(...)`
  - when `return_debug=False`, now runs:
    - `softmax_store_p(...)`
    - `quantize_p(...)`
  - returns the same `(P_fp4, P_sc_prepared, LSE)` surface without touching the legacy debug kernel

- `_dump_live_quant_p_and_bf16_ref_from_scores(...)`
  - when `return_debug=False`, now runs:
    - `softmax_store_p(...)`
    - `quantize_p(...)`
  - returns the stable BF16 reference directly instead of the legacy live-dump sidecar

- variant runner `_run_variant_live_quant_dump(...)`
  - now computes:
    - `softmax_store_p(...)`
    - `quantize_p(...)`
    - `pv_from_p(...)`
  - so the public `variant='live_quant_dump_pv'` path no longer depends on the broken debug dump/reconstruction path

- `measure_live_quant_dump_repeatability(...)`
  - no longer hard-fails on the old live-range gate before sampling
  - it now repeats the stable non-debug dump path

Verification on `cuda:1`:

- `dump_live_quant_p_from_scores(...)`
  - `S=256` and `S=2048`, `random_live_fp4`
  - finite outputs
  - shapes:
    - `P_fp4`: `(1, 1, S, S/2)`
    - `P_sc_prepared`: `(1, 1, S/128, S/64, 512)`
    - `LSE`: `(1, 1, S)`

- `_dump_live_quant_p_and_bf16_ref_from_scores(...)`
  - `S=256` and `S=2048`, `random_live_fp4`
  - byte-exact FP4 payload match versus `dump_live_quant_p_from_scores(...)`
  - exact prepared-scale match
  - exact `LSE` match
  - debug metadata: `path = "softmax_store_p_reference"`

- `measure_live_quant_dump_repeatability(...)`
  - `S=2048`, `random_live_fp4`
  - `repeatable = True`
  - `max_abs_diff_vs_run0 = 0.0`
  - `lse_max_abs_diff_vs_run0 = 0.0`

- `run_variant(..., variant='live_quant_dump_pv')`
  - `S=256`, `random_live_fp4`
  - finite `out`
  - finite `lse`
  - timing keys:
    - `softmax_store_p_ms`
    - `requantize_p_ms`
    - `pv_from_p_ms`
    - `total_ms`
  - debug metadata: `path = "softmax_store_p_reference"`

- larger-sequence sanity
  - `dump_live_quant_p_from_scores(...)`, `S=4096`, `random_live_fp4`: finite outputs, expected shapes
  - `measure_live_quant_tile_window_against_ref(...)`, `S=4096`, final tile, `random_live_fp4`: all diffs exactly `0.0`
  - `run_variant(..., variant='live_quant_dump_pv')`, `S=2048`, `random_live_fp4`: finite `out/lse`, debug metadata still `path = "softmax_store_p_reference"`

Current practical state after this pass:

- the normal live-quant dump, compare, repeatability, and `live_quant_dump_pv` flows run through the stable production-backed path
- the legacy debug score kernel is still present only for explicit debug/forensics paths that request nondefault knobs or raw sidecars

## 2026-04-13 update: live scale diagnosis defaults to the stable oracle too

The public scale-diagnosis helper no longer needs the legacy row-scale debug extension unless explicit debug output is requested.

Python change:

- `_dump_live_quant_p_row_scales_and_bf16_ref_from_scores(...)`
  - when `return_debug=False`, now returns row-scale data reconstructed from:
    - `softmax_store_p(...)`
    - `quantize_p(...)`
    - `_inverse_prepared_scale_swizzle(...)`
  - so even the private non-debug row-scale helper no longer depends on the extension

- `_build_live_scale_oracle_from_scores(...)`
  - when `return_debug=False`, now builds the oracle from:
    - `softmax_store_p(...)`
    - `quantize_p(...)`
    - `_inverse_prepared_scale_swizzle(...)`
  - this produces a stable prepared-scale/reference record without depending on the broken row-scale dump sidecars
  - the old extension-backed row-scale path is still retained behind `return_debug=True`

- `diagnose_live_quant_dump_scales(...)`
  - now reports `reference_mode` from the oracle instead of hardcoding the old debug-only scan/tile-amax mode

Verification on `cuda:1`:

- `_dump_live_quant_p_row_scales_and_bf16_ref_from_scores(...)`
  - `S=2048`, `random_live_fp4`, `return_debug=False`
  - finite `LSE`
  - expected shapes
  - `row_live == row_scan`
  - `debug is None`

- `diagnose_live_quant_dump_scales(...)`
  - `S=2048`, `random_live_fp4`, `include_stored_reference=True`
  - `reference_mode = "stored_quantize_p_reference"`
  - `scale_max_abs_diff_vs_scan_ref = 0.0`
  - `scale_mean_abs_diff_vs_scan_ref = 0.0`
  - `rowmajor_raw_max_abs_diff = 0.0`
  - `scale_max_abs_diff_vs_stored = 0.0`
  - `lse_max_abs_diff_vs_stored = 0.0`
  - repeatability still returns `True`

Practical state:

- public live-quant diagnostics now run on the stable production-backed oracle by default
- the legacy row-scale debug extension is now only needed for explicit low-level forensics

## 2026-04-13 update: benchmark and study summaries now reflect the live-P strategy

The harness/reporting layer is now aligned with the actual decision:

- `stored-P` is treated as an oracle, not the product direction
- `streaming_live_localcta_direct` is treated as the fused live-`P` semantic reference
- `streaming_live_localcta_prod_tcgen_mm2` is treated as the first fused live-`P` optimization candidate
- `live_sa3_baseline` is still reported in PV-only matched studies, but now as a control path rather than a “best live” candidate

Python changes:

- `benchmark_nvfp4_qk_pv_matrix(...)`
  - each row now reports:
    - `variant_role`
    - `allocates_full_p`
    - `memory_contract`
  - correctness rows now include:
    - `diagnostics_vs_stored_p_oracle`
    - `diagnostics_vs_live_direct_reference`
  - performance rows now include:
    - `ratio_vs_stored_p_oracle`
    - `ratio_vs_live_direct_reference`

- `list_forward_precision_matrix_support()`
  - now exposes:
    - `candidate_live_p_modes = ('live_direct', 'live_localcta_cta_amax_experimental')`
    - `control_p_modes = ('live_sa3_baseline',)`

- `_summarize_pv_only_power2_case_rows(...)` / matched-study README generation
  - `best_live` / `best_live_candidate` now exclude `live_sa3_baseline`
  - `sa3_control` is reported separately
  - matched-study README wording now says `best live candidate` instead of treating SA3 as part of the candidate set

Verification:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`

- `benchmark_nvfp4_qk_pv_matrix(...)`, `S=1024`, `heads=1`, `warmup=0`, `iters=1`, on `cuda:1`
  - `qk_nvfp4_plus_ref_pv`
    - `variant_role = "stored_p_oracle"`
    - `allocates_full_p = True`
  - `streaming_live_localcta_direct`
    - `variant_role = "fused_live_reference"`
    - `allocates_full_p = False`
    - `diagnostics_vs_live_direct_reference.mean_abs_diff = 0.0`
  - `streaming_live_localcta_prod_tcgen_mm2`
    - `variant_role = "fused_live_optimization_candidate"`
    - `allocates_full_p = False`
    - performance row now reports `ratio_vs_live_direct_reference`
  - `qk_pv_nvfp4_production`
    - `variant_role = "production_candidate"`

- synthetic PV-only summary check
  - winner can still be `live_sa3_baseline` if it is numerically best in the raw row set
  - but `best_live` / `best_live_candidate` now resolve to the real candidate set only
  - `sa3_control` is surfaced separately

Current practical state after this pass:

- the repo’s default diagnostics now use the stable oracle path
- the repo’s benchmark/summarization layer now treats the fused direct path as the live-`P` reference and SA3 as a control, which matches the actual implementation strategy

## 2026-04-13 update: downstream production/repeatability helpers now carry the same live-P contract

Follow-on harness cleanup:

- `benchmark_nvfp4_qk_pv_matrix(...)`
  - now also flattens the new comparison fields:
    - `diff_vs_stored_p_oracle_*`
    - `diff_vs_live_direct_reference_*`
  - so JSON consumers no longer need to unpack nested diagnostics to compare the fused live path against the stored oracle or direct reference

- `benchmark_production_fp4pv_launch_modes(...)`
  - now includes:
    - `variant_role`
    - `allocates_full_p`
    - `memory_contract`
    - flattened `diff_vs_stored_p_oracle_*`
    - flattened persistent/fullgrid comparison metrics

- `measure_pv_variant_repeatability(...)`
  - now carries the same `variant_role` / `memory_contract` metadata

- `measure_pv_variant_row_block_profiles(...)`
  - now carries the same `variant_role` / `memory_contract` metadata

- shared helper
  - `_resolve_pv_variant_and_launch_mode(...)` now centralizes the variant-to-launch interpretation for these reporting helpers

Verification on `cuda:1`:

- `benchmark_nvfp4_qk_pv_matrix(...)`, `S=1024`, `random_live_fp4`
  - `streaming_live_localcta_prod_tcgen_mm2`
    - `variant_role = "fused_live_optimization_candidate"`
    - `memory_contract = "fused_live_p"`
    - flattened keys present:
      - `diff_vs_stored_p_oracle_mean_abs`
      - `diff_vs_live_direct_reference_mean_abs`

- `benchmark_production_fp4pv_launch_modes(...)`, `S=1024`, `random_live_fp4`
  - `qk_pv_nvfp4_production`
    - `variant_role = "production_candidate"`
    - `memory_contract = "fused_live_p"`
    - flattened keys present:
      - `diff_vs_stored_p_oracle_mean_abs`
      - `diff_vs_production_fullgrid_mean_abs`

- `measure_pv_variant_repeatability(...)`
  - `streaming_live_localcta_direct` row now reports `variant_role = "fused_live_reference"`

- `measure_pv_variant_row_block_profiles(...)`
  - `streaming_live_localcta_direct` row now reports `variant_role = "fused_live_reference"`

## 2026-04-13 update: added a role-aware NVFP4 QK+PV matrix writer

There is now a first-class disk writer for the NVFP4 QK+PV matrix instead of relying on ad hoc row inspection.

Python changes:

- `_summarize_nvfp4_qk_pv_matrix_rows(...)`
  - groups the matrix into per-case summaries for:
    - stored-P oracle
    - fused live reference
    - fused live optimization candidate
    - production candidate

- `write_nvfp4_qk_pv_matrix(...)`
  - writes:
    - `rows.jsonl`
    - `summary.json`
    - `README.md`
  - README is organized around the actual live-P strategy:
    - correctness vs stored oracle
    - correctness vs fused direct reference
    - performance ratios vs stored oracle and fused direct reference

Verification on `cuda:1`:

- `write_nvfp4_qk_pv_matrix(...)`, with `S=1024`, `heads=1`, `warmup=0`, `iters=1`
  - all three artifacts are created successfully
  - summary includes two correctness cases (`random_live_fp4`, `zero_qk_random_v`) and one performance case
  - canonical `S=1024` `random_live_fp4` case reports:
    - fused reference `streaming_live_localcta_direct`
    - optimization candidate `streaming_live_localcta_prod_tcgen_mm2`
    - production persistent/fullgrid
    - stored-P oracle timing

Practical state:

- the main live-P benchmark can now be emitted as a stable artifact with the same strategy assumptions that the in-memory rows already use

## 2026-04-13 update: benchmark_streaming_live_pv_matrix now defaults to the real fused live-P core set

I updated the generic streaming PV benchmark so its default sweep reflects the actual decision set for this project instead of the older exhaustive tcgen/debug menu.

Python changes:

- `_DEFAULT_STREAMING_SPEED_SPECS` and `_DEFAULT_STREAMING_ACCURACY_SPECS`
  - now default to the core fused live-P set:
    - `regular_fa4_fused`
    - `streaming_live_localcta_direct`
    - `streaming_live_localcta_prod_tcgen_mm2`
    - `qk_pv_nvfp4_production_fullgrid`
    - `qk_pv_nvfp4_production`
  - the old broader list is preserved as `_STREAMING_LIVE_EXHAUSTIVE_SPECS` for explicit use

- `_normalize_streaming_variant_specs(...)`
  - now accepts the fixed launch contracts for:
    - `qk_pv_nvfp4_production -> persistent`
    - `qk_pv_nvfp4_production_fullgrid -> fullgrid`

- `benchmark_streaming_live_pv_matrix(...)`
  - now emits a real stored-P oracle row:
    - `variant = "alt1_full_p_quant_pv"`
    - `variant_role = "stored_p_oracle"`
    - `allocates_full_p = True`
  - accuracy rows now carry:
    - `variant_role`
    - `allocates_full_p`
    - `memory_contract`
    - `diagnostics_vs_stored_p_oracle`
    - `diagnostics_vs_live_direct_reference`
    - flattened `diff_vs_stored_p_oracle_*`
    - flattened `diff_vs_live_direct_reference_*`
    - `ratio_vs_stored_p_oracle`
    - `ratio_vs_live_direct_reference`
  - speed rows now carry:
    - the same role/memory metadata
    - `ratio_vs_live_direct_reference`

Verification on `cuda:1`:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`
- `benchmark_streaming_live_pv_matrix(seqlens=(), accuracy_seqlens=(512,), batch=1, heads=1, warmup=0, iters=1)`
  - emits the new stored-P oracle row
  - emits direct/MM2/production rows with the new role/memory fields
  - canonical `S=512`, `random_live_fp4` examples:
    - `streaming_live_localcta_direct`
      - `ratio_vs_stored_p_oracle = 0.04219`
      - `diff_vs_stored_p_oracle_mean_abs = 0.0114592`
    - `streaming_live_localcta_prod_tcgen_mm2`
      - `ratio_vs_live_direct_reference = 0.66556`
      - `diff_vs_live_direct_reference_mean_abs = 0.0205945`
    - `qk_pv_nvfp4_production_fullgrid`
      - `ratio_vs_live_direct_reference = 0.04359`
      - `diff_vs_live_direct_reference_mean_abs = 0.1034353`

Practical state:

- the repo now has two aligned benchmark entrypoints:
  - `benchmark_nvfp4_qk_pv_matrix(...)` for the full NVFP4 QK+PV picture
  - `benchmark_streaming_live_pv_matrix(...)` for the fused live-P core decision set
- both now speak in the same terms:
  - stored-P oracle
  - fused live direct reference
  - fused live optimization candidate
  - production candidate

## 2026-04-13 update: added a first-class writer for the fused live-P core benchmark

There is now a disk writer for `benchmark_streaming_live_pv_matrix(...)`, so the fused live-P core sweep can be emitted as a stable artifact instead of inspected row-by-row in ad hoc Python.

Python changes:

- `_summarize_streaming_live_pv_matrix_rows(...)`
  - groups the generic streaming rows into:
    - accuracy cases keyed by `(seqlen, input_mode)`
    - speed cases keyed by `seqlen`
  - summarizes:
    - stored-P oracle
    - regular FA4 persistent/fullgrid baselines
    - fused live direct reference
    - fused live MM2 candidate
    - production persistent/fullgrid

- `write_streaming_live_pv_matrix(...)`
  - writes:
    - `rows.jsonl`
    - `summary.json`
    - `README.md`
  - README is organized around:
    - accuracy vs stored-P oracle and live-direct reference
    - speed vs regular FA4 baselines and live-direct reference

- `__all__`
  - now exports `write_streaming_live_pv_matrix`

Verification on `cuda:1`:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`
- `write_streaming_live_pv_matrix(...)`, with:
  - `seqlens=(1024,)`
  - `accuracy_seqlens=(512,)`
  - `batch=1`, `heads=1`
  - `warmup=0`, `iters=1`
  - `speed_input_mode='random_live_fp4'`
  - successfully created all three artifacts

Canonical smoke outputs:

- first accuracy case `S=512`, `random_live_fp4`
  - direct:
    - `3.7811 ms`
    - `diff_vs_stored_p_oracle_mean_abs = 0.0122141`
  - MM2:
    - `0.1287 ms`
    - `diff_vs_live_direct_reference_mean_abs = 0.0212720`
  - production persistent:
    - `0.0780 ms`
    - `diff_vs_live_direct_reference_mean_abs = 0.0212752`

- first speed case `S=1024`, `random_live_fp4`
  - direct:
    - `5.7201 ms`
    - `3.772x` fullgrid baseline time
  - MM2:
    - `5.3216 ms`
    - `0.930x` direct time
  - production fullgrid:
    - `1.6742 ms`
    - `0.293x` direct time

Practical state:

- the harness layer now has first-class artifact writers for both:
  - the full NVFP4 QK+PV matrix
  - the fused live-P core benchmark
- that means the next kernel-side work can be evaluated and archived without more benchmark/reporting cleanup

## 2026-04-13 update: streaming localCTA variants can now use a persistent grid

I added a real persistent launch mode for the streaming localCTA sidecar kernel instead of forcing every non-production live variant through fullgrid.

CUDA/extension changes:

- `persistent_dim_fp4pv_exp(...)`
  - added alongside the existing fullgrid helper
  - uses the same persistent cap shape as the rest of the fp4pv code:
    - `2 * C::NUM_SM`
    - aligned up to `C::CLUSTER_SIZE`

- `dispatch_forward_streaming_live_localcta(...)`
  - now accepts `persistent_launch`
  - chooses between:
    - `persistent_dim_fp4pv_exp<C>(...)`
    - `fullgrid_dim_fp4pv_exp<C>(...)`

Python changes:

- `_run_streaming_live_localcta(...)`
  - now accepts `launch_mode`
  - forwards `persistent_launch` into the extension

- `run_streaming_live_pv_variant(...)`
  - now allows `launch_mode='persistent'` for the streaming localCTA variants
  - timing breakdown keys now distinguish persistent vs fullgrid for those variants

- `_normalize_streaming_variant_specs(...)`
  - no longer rejects explicit tuples such as:
    - `('streaming_live_localcta_direct', 'persistent')`
    - `('streaming_live_localcta_prod_tcgen_mm2', 'persistent')`

- `benchmark_streaming_live_pv_matrix(...)`
  - now resolves the live-direct reference from whichever launch mode is actually present
  - this fixes the earlier `ratio_vs_live_direct_reference = null` problem for persistent-only variant tuples

Verification:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`
- `make -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments -j1`

- small functional smoke on `cuda:1`, `S=512`, `batch=1`, `heads=1`
  - `run_streaming_live_pv_variant(..., variant='streaming_live_localcta_direct', launch_mode='persistent')`
  - `run_streaming_live_pv_variant(..., variant='streaming_live_localcta_prod_tcgen_mm2', launch_mode='persistent')`
  - both return finite outputs and timing metadata

- benchmark API smoke on `cuda:1`
  - `benchmark_streaming_live_pv_matrix(...)` with explicit persistent tuples for:
    - `streaming_live_localcta_direct`
    - `streaming_live_localcta_prod_tcgen_mm2`
  - rows now report:
    - `ratio_vs_live_direct_reference`
    - `diff_vs_live_direct_reference_mean_abs`

Important caveat:

- I did not sign off a large cap-limited speed improvement yet.
- The long `S=8192`, `heads=8` MM2/direct timing probes still hit the existing event-timeout regime before I could collect a reliable persistent-vs-fullgrid speed delta.
- So the keepable result from this pass is:
  - persistent launch support is implemented and usable
  - benchmark/reporting support is implemented and usable
  - large-sequence performance validation still needs a dedicated follow-up pass

## 2026-04-13 update: added a subprocess-safe streaming launch-mode compare helper

To avoid wedging the parent process while testing persistent vs fullgrid on the streaming localCTA variants, I added a public subprocess helper:

- `measure_streaming_live_launch_mode_compare_subprocess(...)`

What it does:

- runs the requested variant twice in isolated subprocesses:
  - once with `launch_mode='fullgrid'`
  - once with `launch_mode='persistent'`
- returns structured per-mode status instead of hanging the caller
- reports:
  - `timing_ms_median`
  - `timing_breakdown_median`
  - finiteness flags
  - `launch_meta`
    - including whether the persistent cap is actually active

Verification on `cuda:1`:

- small MM2 smoke:
  - `seqlen=512`
  - `heads=1`
  - `variant='streaming_live_localcta_prod_tcgen_mm2'`
  - `event_timeout_ms=10000`
  - both modes succeed
  - result:
    - fullgrid `12.259 ms`
    - persistent `11.504 ms`
    - `persistent_over_fullgrid = 0.938`

- cap-limited MM2 smoke:
  - `seqlen=4096`
  - `heads=16`
  - `variant='streaming_live_localcta_prod_tcgen_mm2'`
  - `event_timeout_ms=10000`
  - `launch_meta.persistent_cap_limited = True`
  - result:
    - fullgrid succeeds at `7.412 ms`
    - persistent returns structured timeout after the helper budget

Practical state:

- we now have a safe way to probe long-sequence persistent/fullgrid behavior for the streaming variants
- the current signal is not “persistent is faster”
- the current signal is:
  - persistent launch support exists
  - small cases run
  - cap-limited MM2 still needs kernel-side debugging or longer-budget validation

## 2026-04-13 update: safe persistent localCTA launches now chunk by head under the grid cap

The practical problem on the streaming localCTA path was not “persistent launch is always broken”. It was the first cap-limited case where one CTA had to pick up a second `cur_bid` task in the same launch. To stop normal workflows from hanging on that path, I changed the Python runner to split persistent streaming-localCTA launches by head when `total_bids > 304`.

Code changes:

- added `_streaming_variant_launch_plan(...)`
- added `_streaming_persistent_head_chunk_size(...)`
- `run_streaming_live_pv_variant(...)` now:
  - keeps `launch_mode='persistent'`
  - splits streaming localCTA variants into multiple persistent launches when needed
  - records:
    - `requested_launch_mode`
    - `head_chunk_size`
    - `head_chunks`
    - `bids_per_launch`
    - `persistent_cap_limited`
    - `launch_fallback_reason`
- `measure_pv_variant_repeatability(...)` and `measure_pv_variant_row_block_profiles(...)` now go through `run_streaming_live_pv_variant(...)` for streaming localCTA variants, so they inherit the same safe path
- `_run_streaming_speed_series_subprocess(...)` now scales its subprocess timeout by the effective chunk count
- `measure_streaming_live_launch_mode_compare_subprocess(...)` now reports the persistent launch plan in `launch_meta`

Verification on `cuda:1`:

- direct persistent runner, first cap-limited case:
  - `seqlen=4096`
  - `heads=10`
  - `variant='streaming_live_localcta_direct'`
  - returns cleanly with:
    - `launch_mode='persistent'`
    - `head_chunk_size=9`
    - `head_chunks=2`
    - `bids_per_launch=288`
    - finite output/LSE

- MM2 persistent runner, same case:
  - `seqlen=4096`
  - `heads=10`
  - `variant='streaming_live_localcta_prod_tcgen_mm2'`
  - returns cleanly with the same chunk plan

- subprocess compare helper, larger cap-limited case:
  - `seqlen=4096`
  - `heads=12`
  - `variant='streaming_live_localcta_prod_tcgen_mm2'`
  - fullgrid: `7.124 ms`
  - persistent: `7.256 ms`
  - `persistent_over_fullgrid = 1.018`
  - `launch_meta` reports:
    - `persistent_effective_launch_mode='persistent'`
    - `persistent_head_chunk_size=9`
    - `persistent_head_chunks=2`
    - `persistent_bids_per_launch=288`

- matrix smoke:
  - `benchmark_streaming_live_pv_matrix(...)`
  - `seqlen=4096`
  - `heads=12`
  - persistent direct/MM2 rows now complete with `head_chunks=2` and `bids_per_launch=288`

Current state:

- normal persistent streaming-localCTA workflows now run instead of hanging at the first cap-limited point
- this is a safe launch-planning workaround, not a kernel-side fix for multi-task-per-CTA persistent reuse
- if we want true cap-limited persistent execution in one launch, the next debugging branch is still inside `kernel_streaming_live_fp4pv(...)` across the `cur_bid += gridDim.x` reuse loop

## 2026-04-13 update: repeatability and row-block diagnostics now accept explicit launch-mode tuples

The public PV diagnostics were still inconsistent with the benchmark path: they only accepted bare variant strings, so persistent-specific calls like `('streaming_live_localcta_prod_tcgen_mm2', 'persistent')` failed before the new safe launch planner could even run.

Code changes:

- `_resolve_pv_variant_and_launch_mode(...)` now accepts either:
  - a bare variant string
  - an explicit `(variant, launch_mode)` tuple
- `measure_pv_variant_repeatability(...)`
- `measure_pv_variant_row_block_profiles(...)`
  now type and resolve `variants` through the tuple-aware path

Verification:

- `measure_pv_variant_repeatability(...)`
  - `seqlen=4096`
  - `heads=12`
  - `variants=(('streaming_live_localcta_prod_tcgen_mm2', 'persistent'),)`
  - now runs cleanly and reports `launch_mode='persistent'`

- `measure_pv_variant_row_block_profiles(...)`
  - `seqlen=2048`
  - `heads=12`
  - `variants=(('streaming_live_localcta_direct', 'persistent'),)`
  - now runs cleanly and reports `launch_mode='persistent'`

Practical effect:

- the benchmark path, subprocess compare helper, repeatability helper, and row-block profile helper all now accept the same launch-mode tuple contract for the streaming localCTA variants

## 2026-04-13 update: speed subprocess metadata and streaming matrix summaries now preserve effective launch behavior

There were still two reporting bugs after the safe persistent chunking change:

1. `_run_streaming_speed_case_subprocess(...)` still used a one-launch timeout budget even when the requested run was being split into multiple persistent head chunks.
2. the streaming matrix summary path still assumed direct/MM2 lived only under `launch_mode='fullgrid'`, so persistent-only matrices could produce rows but incomplete summaries/README output.

Code changes:

- `_run_streaming_speed_case_subprocess(...)`
  - now scales its outer subprocess timeout by the effective launch-plan `head_chunks`
- `_STREAMING_SPEED_CASE_SUBPROCESS_CODE`
- `_STREAMING_SPEED_SERIES_SUBPROCESS_CODE`
  - now serialize:
    - `requested_launch_mode`
    - `max_bids_cap`
    - `persistent_cap_limited`
    - `launch_fallback_reason`
- `benchmark_streaming_live_pv_matrix(...)`
  - speed and accuracy rows now record:
    - effective `launch_mode`
    - `requested_launch_mode`
    - launch-plan metadata copied from the underlying record
- `_summarize_streaming_live_pv_matrix_rows(...)`
  - now prefers `fullgrid` when present but will also summarize persistent-only direct/MM2 rows instead of dropping them
- `write_streaming_live_pv_matrix(...)`
  - README lines for the fused live direct/MM2 rows now print the effective launch mode

Verification:

- persistent-only speed matrix smoke:
  - `variants=(('streaming_live_localcta_direct', 'persistent'), ('streaming_live_localcta_prod_tcgen_mm2', 'persistent'))`
  - `seqlen=4096`
  - `heads=12`
  - speed rows now report:
    - `launch_mode='persistent'`
    - `requested_launch_mode='persistent'`
    - `head_chunk_size=9`
    - `head_chunks=2`
    - `max_bids_cap=304`
    - `bids_per_launch=288`
    - `persistent_cap_limited=True`

- the corresponding summary case now includes:
  - fused live reference with `launch_mode='persistent'`
  - fused live optimization candidate with `launch_mode='persistent'`

- fallback metadata check:
  - `_streaming_variant_launch_plan(batch=1, heads=1, seqlen=40960, variant='streaming_live_localcta_direct', launch_mode='persistent')`
  - returns:
    - `effective_launch_mode='fullgrid'`
    - `launch_fallback_reason='persistent_localcta_bids_per_head_exceeds_grid_cap'`

Current state:

- the safe persistent workaround is now reflected consistently in:
  - direct runner results
  - subprocess speed helpers
  - matrix rows
  - matrix summaries
  - README output
- the unresolved work is still kernel-side:
  - one-launch cap-limited persistent localCTA reuse remains unfixed

## 2026-04-13 update: compare/repeatability/profile helpers now preserve launch-plan metadata too

There was one last metadata mismatch after the previous pass: the underlying runner and subprocess serializers carried the launch-plan fields, but some public helpers still dropped them in their returned rows.

Code changes:

- `measure_streaming_live_launch_mode_compare_subprocess(...)`
  - per-mode results now include:
    - `launch_mode`
    - `requested_launch_mode`
    - `max_bids_cap`
    - `persistent_cap_limited`
    - `launch_fallback_reason`
- `measure_pv_variant_repeatability(...)`
- `measure_pv_variant_row_block_profiles(...)`
  - now preserve the streaming launch-plan metadata from the safe runner:
    - `requested_launch_mode`
    - `head_chunk_size`
    - `head_chunks`
    - `max_bids_cap`
    - `total_bids`
    - `bids_per_launch`
    - `persistent_cap_limited`
    - `launch_fallback_reason`

Verification:

- launch-mode compare helper:
  - `seqlen=4096`
  - `heads=12`
  - `variant='streaming_live_localcta_prod_tcgen_mm2'`
  - persistent mode now reports:
    - `launch_mode='persistent'`
    - `requested_launch_mode='persistent'`
    - `head_chunk_size=9`
    - `head_chunks=2`
    - `max_bids_cap=304`
    - `bids_per_launch=288`
    - `persistent_cap_limited=True`

- repeatability helper:
  - `variants=(('streaming_live_localcta_prod_tcgen_mm2', 'persistent'),)`
  - `seqlen=4096`
  - `heads=12`
  - row now includes the same chunk metadata

- row-block profile helper:
  - `variants=(('streaming_live_localcta_direct', 'persistent'),)`
  - `seqlen=2048`
  - `heads=12`
  - row now includes:
    - `requested_launch_mode='persistent'`
    - `head_chunks=1`
    - `total_bids=192`
    - `persistent_cap_limited=False`

Practical state:

- all of the public streaming-localCTA measurement paths now preserve the requested/effective launch distinction and the chunk/cap metadata
- the remaining open problem is still the same one-launch persistent reuse bug in the kernel itself

## 2026-04-13 update: launch execution mode is now explicit

The launch-plan metadata was still a bit too low-level to scan quickly. I added a derived `launch_execution_mode` label so rows tell you directly whether they were:

- `persistent_single_launch`
- `persistent_chunked`
- `persistent_fallback_fullgrid`
- `fullgrid_single_launch`
- `fullgrid_chunked`

Code changes:

- `_streaming_variant_launch_plan(...)`
  - now derives `launch_execution_mode`
- the field is now propagated through:
  - `run_streaming_live_pv_variant(...)`
  - speed subprocess serializers
  - `measure_streaming_live_launch_mode_compare_subprocess(...)`
  - `benchmark_streaming_live_pv_matrix(...)`
  - `measure_pv_variant_repeatability(...)`
  - `measure_pv_variant_row_block_profiles(...)`
  - streaming matrix summary/README output

Verification:

- planner states:
  - `seqlen=2048, heads=12, persistent direct` -> `persistent_single_launch`
  - `seqlen=4096, heads=12, persistent direct` -> `persistent_chunked`
  - `seqlen=40960, heads=1, persistent direct` -> `persistent_fallback_fullgrid`

- compare helper:
  - `seqlen=4096`
  - `heads=12`
  - `variant='streaming_live_localcta_prod_tcgen_mm2'`
  - persistent mode now reports `launch_execution_mode='persistent_chunked'`

- repeatability helper:
  - `variants=(('streaming_live_localcta_prod_tcgen_mm2', 'persistent'),)`
  - `seqlen=4096`
  - `heads=12`
  - row now reports `launch_execution_mode='persistent_chunked'`

Practical effect:

- a single field now tells you whether a “persistent” row was a real one-launch persistent run or the chunked/fallback workaround

## 2026-04-13 update: canonical MM2-vs-direct compare helper and writer

I added a small canonical compare surface so the next optimization pass can target one number instead of fishing it out of the larger streaming matrix.

Code changes:

- `benchmark_streaming_live_mm2_vs_direct_canonical(...)`
- `write_streaming_live_mm2_vs_direct_canonical(...)`

Contract:

- canonical case defaults:
  - `seqlen=4096`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `qk_quant_backend='v5'`
  - `v_quant_backend='localcta'`
  - `device='cuda:1'`
  - `warmup=1`
  - `iters=5`
- four runs on one shared input set:
  - direct fullgrid
  - MM2 fullgrid
  - direct persistent effective
  - MM2 persistent effective
- primary score:
  - `mm2_over_direct_fullgrid`
- shadow score:
  - `mm2_over_direct_persistent_effective`
- each row records:
  - `launch_mode`
  - `requested_launch_mode`
  - `launch_execution_mode`
  - `head_chunks`
  - `bids_per_launch`

Verification:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`
- writer smoke:
  - `write_streaming_live_mm2_vs_direct_canonical(...)`
  - artifacts emitted successfully:
    - `rows.jsonl`
    - `summary.json`
    - `README.md`

Canonical default result on `cuda:1`:

- fullgrid:
  - direct:
    - `launch_execution_mode='fullgrid_single_launch'`
    - `timing_ms_median=0.383808`
  - MM2:
    - `launch_execution_mode='fullgrid_single_launch'`
    - `timing_ms_median=0.248736`
  - primary score:
    - `mm2_over_direct_fullgrid=0.648074`

- persistent effective:
  - direct:
    - `launch_execution_mode='persistent_chunked'`
    - `head_chunks=2`
    - `bids_per_launch=288`
    - `timing_ms_median=0.531328`
  - MM2:
    - `launch_execution_mode='persistent_chunked'`
    - `head_chunks=2`
    - `bids_per_launch=288`
    - `timing_ms_median=0.375616`
  - shadow score:
    - `mm2_over_direct_persistent_effective=0.706938`

Sanity note:

- a cold one-shot run (`warmup=0`, `iters=1`) was much noisier and startup-dominated
- the default helper numbers above are the steady-state ones to compare going forward

Direct launch-mode compare on the same canonical case:

- `measure_streaming_live_launch_mode_compare_subprocess(...)`
  - `variant='streaming_live_localcta_direct'`
  - `seqlen=4096`
  - `heads=12`
  - `warmup=1`
  - `iters=5`
- result:
  - fullgrid:
    - `0.383296 ms`
  - persistent effective:
    - `0.531296 ms`
  - `persistent_over_fullgrid=1.386125`

Practical interpretation:

- the primary number for the next kernel pass should be the fullgrid MM2/direct ratio
- on the current tree and current host, that number is about `0.65x`
- the persistent path remains a secondary shadow metric only; on this canonical case, the chunked persistent direct path is slower than fullgrid

## 2026-04-13 update: fixed the streaming speed matrix to use steady-state series timing

There was still one benchmark inconsistency after adding the canonical helper:

- the new canonical helper and the launch-mode compare helper both reported the canonical fullgrid case at about:
  - direct `~0.383 ms`
  - MM2 `~0.250 ms`
- but `benchmark_streaming_live_pv_matrix(...)` still reported:
  - direct `~7.26 ms`
  - MM2 `~7.13 ms`

Root cause:

- the speed sweep in `benchmark_streaming_live_pv_matrix(...)` was still using the one-shot subprocess helper for every warmup and every timed sample
- that path was effectively dominated by fresh-process / fresh-context behavior and did not match the steady-state series helper already used by:
  - `measure_streaming_live_launch_mode_compare_subprocess(...)`
  - the new canonical MM2-vs-direct helper

Code change:

- switched the speed sweep in `benchmark_streaming_live_pv_matrix(...)` from repeated `_run_streaming_speed_case_subprocess_with_retry(...)` calls to one `_run_streaming_speed_series_subprocess(...)` call per spec
- speed rows now consume:
  - `timing_samples`
  - `timing_breakdown_samples`
  - launch metadata
  from the same steady-state series subprocess path as the canonical helper

Verification:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`
- `benchmark_streaming_live_pv_matrix(...)`
  - `seqlens=(4096,)`
  - `accuracy_seqlens=()`
  - variants:
    - `('streaming_live_localcta_direct', 'fullgrid')`
    - `('streaming_live_localcta_prod_tcgen_mm2', 'fullgrid')`
    - `('streaming_live_localcta_direct', 'persistent')`
    - `('streaming_live_localcta_prod_tcgen_mm2', 'persistent')`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `warmup=1`
  - `iters=5`

Corrected speed rows on `cuda:1`:

- fullgrid:
  - direct:
    - `timing_ms_median=0.384032`
  - MM2:
    - `timing_ms_median=0.250400`
    - `ratio_vs_live_direct_reference=0.652029`

- persistent effective:
  - direct:
    - `timing_ms_median=0.531264`
    - `launch_execution_mode='persistent_chunked'`
  - MM2:
    - `timing_ms_median=0.375616`
    - `ratio_vs_live_direct_reference=0.978085`

Practical effect:

- the large streaming matrix now agrees with the canonical helper on the compare number to within normal run-to-run noise
- the `~7 ms` rows were a measurement artifact and should be treated as obsolete
- the stable compare number to use going forward is still:
  - MM2/direct fullgrid `~0.65x` on the canonical `S=4096, H=12` case

## 2026-04-13 update: canonical MM2-vs-direct helper now carries stored-P oracle comparisons

The canonical helper was still missing the most useful accuracy context: it reported MM2-vs-direct speed and MM2-vs-direct output drift, but not either variant’s drift versus the stored-P oracle on the same shared inputs.

Code changes:

- `benchmark_streaming_live_mm2_vs_direct_canonical(...)`
  - now also runs `alt1_full_p_quant_pv` on the same input set
  - adds a `stored_p_oracle` case row
  - adds:
    - `fullgrid_direct_vs_stored_p_oracle`
    - `fullgrid_mm2_vs_stored_p_oracle`
    comparisons to the summary
- `write_streaming_live_mm2_vs_direct_canonical(...)`
  - README now includes the stored-P oracle case
  - README now includes direct/MM2-vs-oracle MAE / max / LSE lines

Verification:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`
- `benchmark_streaming_live_mm2_vs_direct_canonical(device='cuda:1')`
- `write_streaming_live_mm2_vs_direct_canonical(device='cuda:1')`

Canonical default result on `cuda:1`:

- case:
  - `seqlen=4096`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `seed=0`
  - `warmup=1`
  - `iters=5`

- timing:
  - stored-P oracle:
    - `20.572608 ms`
  - direct fullgrid:
    - `0.383136 ms`
  - MM2 fullgrid:
    - `0.250432 ms`
  - direct persistent effective:
    - `0.531232 ms`
  - MM2 persistent effective:
    - `0.375584 ms`

- speed ratios:
  - `mm2_over_direct_fullgrid = 0.653637`
  - `mm2_over_direct_persistent_effective = 0.707006`
  - `direct_over_stored_p_oracle = 0.018624`
  - `mm2_over_stored_p_oracle = 0.012173`

- oracle accuracy on this canonical seed:
  - direct vs stored-P oracle:
    - `mean_abs_diff = 0.0169061`
    - `max_abs_diff = 19.0`
    - `lse_max_abs_diff = 0.0160704`
  - MM2 vs stored-P oracle:
    - `mean_abs_diff = 0.0205724`
    - `max_abs_diff = 17.0`
    - `lse_max_abs_diff = 0.0160704`

Interpretation:

- the helper now gives one artifact with both the primary speed number and the corresponding stored-P oracle gap
- on the current canonical seed, MM2 keeps the expected `~0.65x` fullgrid speedup over direct
- both fused variants remain far faster than stored-P on the same case

## 2026-04-13 update: compact 2k/4k fullgrid speed ladder

I also ran a compact steady-state speed ladder on `random_live_fp4`, `batch=1`, `heads=12`, `qk_quant_backend='v5'`, `v_quant_backend='localcta'` to make sure the canonical ranking is not just a one-point accident.

Results on `cuda:1`:

- `seqlen=2048`
  - regular FA4 fullgrid:
    - `0.105728 ms`
  - direct fullgrid:
    - `0.143904 ms`
  - MM2 fullgrid:
    - `0.107008 ms`
    - `ratio_vs_live_direct_reference = 0.743607`
  - production fullgrid:
    - `0.106720 ms`
    - `ratio_vs_live_direct_reference = 0.741606`

- `seqlen=4096`
  - regular FA4 fullgrid:
    - `0.244640 ms`
  - direct fullgrid:
    - `0.383328 ms`
  - MM2 fullgrid:
    - `0.248640 ms`
    - `ratio_vs_live_direct_reference = 0.648635`
  - production fullgrid:
    - `0.244000 ms`
    - `ratio_vs_live_direct_reference = 0.636531`

Practical interpretation:

- the fullgrid ranking is stable across `S=2048` and `S=4096`
- MM2 remains the main localCTA optimization candidate
- production fullgrid is still a speed-only comparison point, not the main candidate, because its accuracy behavior remains weaker on control cases

## 2026-04-13 update: canonical BF16 comparison

I also measured the current fused live-FP4 PV candidates against the real BF16 baseline rather than only against stored-P or the live-direct reference.

Benchmark:

- helper:
  - `benchmark_fp4_vs_bf16_baseline(...)`
- case:
  - `seqlen=4096`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `seed=0`
  - `warmup=1`
  - `iters=5`
  - `device='cuda:1'`
  - `qk_quant_backend='v5'`
  - `v_quant_backend='localcta'`

Results:

- BF16 baseline:
  - variant:
    - `regular_fa4_bf16`
  - launch:
    - `persistent`
  - timing:
    - `0.100896 ms`

- fused live FP4 PV:
  - direct fullgrid:
    - `0.382400 ms`
    - `3.790x` slower than BF16
    - vs BF16:
      - `mean_abs_diff = 0.0117555`
      - `max_abs_diff = 15.1477`
      - `lse_max_abs_diff = 0.109608`
  - MM2 fullgrid:
    - `0.250496 ms`
    - `2.483x` slower than BF16
    - vs BF16:
      - `mean_abs_diff = 0.0262465`
      - `max_abs_diff = 16.7469`
      - `lse_max_abs_diff = 0.109608`
  - production fullgrid:
    - `0.244736 ms`
    - `2.426x` slower than BF16
    - vs BF16:
      - `mean_abs_diff = 0.0258041`
      - `max_abs_diff = 15.4319`
      - `lse_max_abs_diff = 0.109608`

Current interpretation:

- on the canonical `S=4096, H=12` case, none of the current live-FP4 PV paths beats the BF16 baseline yet
- MM2 and production fullgrid are much better than direct on speed, but they are still roughly `2.4x-2.5x` slower than BF16
- direct remains the most accurate of the three against BF16 on this seed, but it is also the slowest by a wide margin

## 2026-04-13 update: BF16-first canonical helper and fixed-overhead helper

I implemented the scoreboard that the next optimization pass should actually use.

Code changes:

- new canonical BF16 helper:
  - `benchmark_fp4_vs_bf16_canonical(...)`
- new canonical BF16 writer:
  - `write_fp4_vs_bf16_canonical(...)`
- new fixed-overhead microbenchmark:
  - `benchmark_streaming_live_fixed_overhead_canonical(...)`
- new fixed-overhead writer:
  - `write_streaming_live_fixed_overhead_canonical(...)`
- metadata cleanup:
  - `regular_fa4_bf16` is now explicitly tagged as `bf16_fused_baseline`
  - `regular_fa4_fused` is now explicitly tagged as `fp4_fused_baseline`
  - `streaming_live_localcta_direct_tile0_only` / `tile1_only` are tagged as `fixed_overhead_probe`

Verification:

- `python3 -m py_compile /workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py`
- `benchmark_fp4_vs_bf16_canonical(device='cuda:1')`
- `write_fp4_vs_bf16_canonical(device='cuda:1')`
- `benchmark_streaming_live_fixed_overhead_canonical(device='cuda:1')`
- `write_streaming_live_fixed_overhead_canonical(device='cuda:1')`

Canonical BF16-first result on `cuda:1`:

- case:
  - `seqlen=4096`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `seed=0`
  - `warmup=1`
  - `iters=5`

- timing:
  - BF16 baseline:
    - `0.101760 ms`
  - FP4 fused fullgrid:
    - `0.244512 ms`
    - `2.402830x` over BF16
  - direct fullgrid:
    - `0.383232 ms`
    - `3.766038x` over BF16
  - MM2 fullgrid:
    - `0.250464 ms`
    - `2.461321x` over BF16
  - production fullgrid:
    - `0.244416 ms`
    - `2.401887x` over BF16
  - stored-P oracle:
    - `18.459168 ms`

Important interpretation:

- the current best fullgrid FP4 case is `production_fullgrid`, not MM2
- but the best FP4 fullgrid case is still only `~2.40x` over BF16
- `regular_fa4_fused/fullgrid` and `production_fullgrid` are essentially sitting on the same speed floor
- that confirms the optimization scope has to be broader than live-P alone

Canonical oracle/BF16 accuracy from the new helper:

- direct vs BF16:
  - `mean_abs_diff = 0.0119083`
  - `max_abs_diff = 18.6143`
  - `lse_max_abs_diff = 0.111815`
- MM2 vs BF16:
  - `mean_abs_diff = 0.0264262`
  - `max_abs_diff = 16.9790`
  - `lse_max_abs_diff = 0.111815`
- production vs BF16:
  - `mean_abs_diff = 0.0186817`
  - `max_abs_diff = 16.9790`
  - `lse_max_abs_diff = 0.111815`
- direct vs stored-P oracle:
  - `mean_abs_diff = 0.0181793`
  - `max_abs_diff = 18.625`
  - `lse_max_abs_diff = 0.0160704`
- MM2 vs stored-P oracle:
  - `mean_abs_diff = 0.0201612`
  - `max_abs_diff = 17.0`
  - `lse_max_abs_diff = 0.0160704`

Fixed-overhead microbenchmark result on the same canonical case:

- `tile0_only_fullgrid = 0.278944 ms`
- `tile1_only_fullgrid = 0.278400 ms`
- `direct_fullgrid = 0.383776 ms`
- `mm2_fullgrid = 0.250112 ms`
- derived:
  - `single_tile_mean_ms = 0.278672`
  - `direct_minus_single_tile_mean_ms = 0.105104`
  - `mm2_minus_single_tile_mean_ms = -0.028560`
  - `mm2_over_direct = 0.651713`

Practical interpretation:

- the direct path is carrying about `0.105 ms` of extra overhead relative to the single-tile probe on this case
- MM2 is already below that single-tile direct probe, which is consistent with it avoiding the direct split-accum/output-scratch path
- the bigger blocker is still the shared FP4 floor:
  - FP4 fused fullgrid and production fullgrid both remain at about `0.244-0.245 ms`
  - BF16 is still `~0.102 ms`

## 2026-04-13 update: BF16-aware streaming matrix timing fix and canonical fullgrid sweep

The BF16-aware `write_streaming_live_pv_matrix(...)` path is now working, and the README/summary formatting is `None`-safe for missing persistent/fullgrid ratios.

More importantly, the BF16 timing inside `benchmark_streaming_live_pv_matrix(...)` accuracy rows was wrong before this pass. The matrix was doing a single one-shot `run_bf16_causal_baseline(...)` call and storing that as `timing_ms_median`, which pulled in startup/one-shot overhead and produced nonsense BF16-relative ratios like `0.035x` for the best FP4 row.

That is now fixed:

- accuracy-side BF16 now uses the same contract as the canonical BF16 helper:
  - warmup BF16 runs
  - `iters` sampled BF16 runs
  - median BF16 timing / median breakdown
  - `run_forward_precision_reference_baseline(...)` only for the reference output/LSE payload

Canonical smoke after the fix, via `write_streaming_live_pv_matrix(...)` on:

- `seqlen=4096`
- `batch=1`
- `heads=12`
- `input_mode='random_live_fp4'`
- `device='cuda:1'`
- `warmup=1`
- `iters=3`
- `include_bf16_baseline=True`

now reports:

- BF16 baseline:
  - `0.101600 ms`
- best FP4 fullgrid:
  - `qk_pv_nvfp4_production_fullgrid`
  - `0.244608 ms`
  - `2.40756x` vs BF16
- direct fullgrid:
  - `0.381696 ms`
- MM2 fullgrid:
  - `0.250464 ms`

I then ran a broader canonical fullgrid sweep against the corrected BF16 baseline on:

- `seqlen=4096`
- `batch=1`
- `heads=12`
- `input_mode='random_live_fp4'`
- `device='cuda:1'`
- `warmup=1`
- `iters=5`

Sorted by timing:

- `regular_fa4_bf16`:
  - `0.100960 ms`
- `qk_pv_nvfp4_production_fullgrid`:
  - `0.244288 ms`
  - `2.41965x` over BF16
- `regular_fa4_fused/fullgrid`:
  - `0.245888 ms`
  - `2.43550x` over BF16
- `streaming_live_localcta_prod_tcgen_mm2/fullgrid`:
  - `0.249920 ms`
  - `2.47544x` over BF16
- `streaming_live_localcta_prod_tcgen/fullgrid`:
  - `0.250016 ms`
- `streaming_live_localcta_direct_tcgenaccum/fullgrid`:
  - `0.250176 ms`
- `streaming_live_localcta_prod_tcgen_mm2_synced/fullgrid`:
  - `0.250624 ms`
- `streaming_live_localcta_prod_tcgen_auto/fullgrid`:
  - `0.253888 ms`
- `streaming_live_localcta_direct/fullgrid`:
  - `0.383232 ms`
  - `3.79588x` over BF16

Current conclusion after the corrected sweep:

- there is no hidden faster fullgrid FP4 candidate already sitting in the current tree
- the fullgrid floor is still `~0.244-0.250 ms`
- BF16 is still `~0.101 ms`
- so the gap to close is still about `2.4x`, and it is shared by the common FP4 fullgrid path rather than being unique to the live-direct or MM2 path

I also ran a short BF16-vs-best-FP4 scaling ladder on the same setup, using:

- `input_mode='random_live_fp4'`
- `batch=1`
- `heads=12`
- `device='cuda:1'`
- `warmup=1`
- `iters=3`

with the main fullgrid candidates:

- `regular_fa4_fused/fullgrid`
- `streaming_live_localcta_prod_tcgen_mm2/fullgrid`
- `qk_pv_nvfp4_production_fullgrid/fullgrid`

Results:

- `S=2048`
  - BF16: `0.046912 ms`
  - best FP4 (`qk_pv_nvfp4_production_fullgrid`): `0.105056 ms`
  - best-FP4-over-BF16: `2.23943x`
- `S=4096`
  - BF16: `0.100512 ms`
  - best FP4 (`regular_fa4_fused`): `0.243648 ms`
  - best-FP4-over-BF16: `2.42407x`
- `S=8192`
  - BF16: `0.265024 ms`
  - best FP4 (`regular_fa4_fused` / `production_fullgrid` essentially tied): `0.912192 ms`
  - best-FP4-over-BF16: `3.44192x`
- `S=16384`
  - BF16: `0.717536 ms`
  - best FP4 (`regular_fa4_fused`): `3.185856 ms`
  - best-FP4-over-BF16: `4.43999x`

That trend is important:

- the FP4 fullgrid path is not closing the gap as sequence length grows
- it is falling further behind BF16 at longer `S`
- so there is no current evidence for a natural crossover where the existing FP4 fused/live path eventually beats BF16
- the next real optimization pass has to attack the common FP4 scaling behavior itself, not just the short-sequence fixed cost

## 2026-04-13 update: fullgrid cost decomposition and fixed QK-only no-store benchmark

I added a dedicated no-store QK-only fullgrid dispatch in the experiments extension and a paired canonical decomposition helper/writer:

- CUDA:
  - `forward_streaming_live_qk_only_lse_only(...)`
- Python:
  - `benchmark_fp4_fullgrid_cost_decomposition_canonical(...)`
  - `write_fp4_fullgrid_cost_decomposition_canonical(...)`

Files:

- [bf16_b300_mha_causal_fp4_pv_experiments.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu:4154)
- [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py:9635)

Two corrections were required before the number was usable:

- the first version of the new dispatch accidentally inherited row-tile debug initialization (`selected_row_tile = 0`), so it only computed one row tile
- the dispatch also needed `grid_exact()` instead of the capped `grid()` launch, otherwise it fell into the same cap-limited reuse path that already causes persistent-style hangs elsewhere

After those fixes, the canonical decomposition on:

- `seqlen=4096`
- `batch=1`
- `heads=12`
- `input_mode='random_live_fp4'`
- `device='cuda:1'`
- `warmup=1`
- `iters=5`

reports:

- BF16 baseline:
  - `0.100960 ms`
- fused QK-only no-store (`LSE` only):
  - `0.137568 ms`
  - `1.36260x` over BF16
  - `LSE max abs diff vs softmax_store_p = 0.017523`
- best current FP4 fullgrid:
  - `qk_pv_nvfp4_production_fullgrid`
  - `0.244192 ms`
  - `2.41870x` over BF16
- gap between best FP4 fullgrid and fused QK-only no-store:
  - `0.106624 ms`

Materialized reference-path breakdown on the same case:

- `softmax_store_p = 13.030560 ms`
- `quantize_p = 9.823648 ms`
- `pv_from_p = 0.047552 ms`
- materialized pipeline total = `22.901760 ms`

Interpretation:

- the live FP4 problem is not “QK is already slower than BF16 by 2.4x”
- the fused QK-only no-store path is only about `1.36x` over BF16 on the canonical case
- the current end-to-end FP4 fullgrid floor is roughly:
  - `QK-only no-store`
  - plus about `0.1066 ms` of additional downstream/common fullgrid overhead
- so the remaining gap to BF16 is now much more tightly localized to the post-QK/common FP4 path, not the raw QK core itself

Short ladder with the fixed no-store QK helper (`warmup=1`, `iters=1` or `3` depending on runtime):

- `S=2048`
  - BF16: `0.045568 ms`
  - fused QK-only no-store: `0.049536 ms`
  - `1.08708x` over BF16
  - best FP4 fullgrid: `0.106624 ms`
  - `2.33989x` over BF16
- `S=4096`
  - BF16: `0.101120 ms`
  - fused QK-only no-store: `0.129408 ms`
  - `1.27975x` over BF16
  - best FP4 fullgrid: `0.244064 ms`
  - `2.41361x` over BF16
- `S=8192`
  - BF16: `0.262976 ms`
  - fused QK-only no-store: `0.424320 ms`
  - `1.61353x` over BF16
  - best FP4 fullgrid: `0.914176 ms`
  - `3.47627x` over BF16
- `S=16384`
  - the no-store QK helper timed out under a `60000 ms` event budget on this host

Current conclusion after the decomposition pass:

- QK is still slower than BF16, but it is nowhere near the full `2.4x+` end-to-end gap at `S=4096`
- the dominant remaining penalty is downstream/common FP4 fullgrid overhead after the QK-only stage
- the next optimization pass should target that `~0.106 ms` post-QK gap on the canonical case, rather than assuming the main blocker is the raw FP4 QK compute itself

## 2026-04-13 update: tile-scoped post-QK profile for direct and MM2

I added another canonical helper/writer to time tile-scoped localCTA slices on the same fullgrid input:

- `benchmark_streaming_live_post_qk_tile_profile_canonical(...)`
- `write_streaming_live_post_qk_tile_profile_canonical(...)`

File:

- [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py:10067)

This uses the existing internal `k_iter_begin/k_iter_end` controls on `_run_streaming_live_localcta(...)` to time:

- `direct_tile0_only`
- `direct_tile1_only`
- `mm2_tile0_only`
- `mm2_tile1_only`

on the same canonical case:

- `seqlen=4096`
- `batch=1`
- `heads=12`
- `input_mode='random_live_fp4'`
- `device='cuda:1'`
- `warmup=1`
- `iters=3`

Representative result:

- BF16 baseline:
  - `0.102624 ms`
- fused QK-only no-store:
  - `0.136160 ms`
  - `1.32679x` over BF16
- regular fused fullgrid:
  - `0.244832 ms`
  - post-QK gap `= 0.108672 ms`
- production fullgrid:
  - `0.244800 ms`
  - post-QK gap `= 0.108640 ms`
- MM2 fullgrid:
  - `0.250720 ms`
  - post-QK gap `= 0.114560 ms`
- direct fullgrid:
  - `0.384096 ms`
  - post-QK gap `= 0.247936 ms`

Tile-scoped slices:

- direct tile 0 only:
  - `0.277216 ms`
  - post-QK gap `= 0.141056 ms`
- direct tile 1 only:
  - `0.277280 ms`
  - post-QK gap `= 0.141120 ms`
- MM2 tile 0 only:
  - `0.258528 ms`
  - post-QK gap `= 0.122368 ms`
- MM2 tile 1 only:
  - `0.257120 ms`
  - post-QK gap `= 0.120960 ms`

Stable writer smoke after that:

- output dir:
  - [streaming_live_post_qk_tile_profile_m5gt698y](/workspace/codebases/fp4_matmul/tk_fa4/results/streaming_live_post_qk_tile_profile_m5gt698y)
- representative metrics:
  - `direct_tile0_post_qk_ms = 0.142656`
  - `mm2_tile0_post_qk_ms = 0.120928`
  - `direct_tile1_post_qk_ms = 0.141248`
  - `mm2_tile1_post_qk_ms = 0.120896`
  - `regular_full_post_qk_ms = 0.108672`
  - `production_full_post_qk_ms = 0.108544`
  - `mm2_full_post_qk_ms = 0.114176`
  - `direct_full_post_qk_ms = 0.245824`

Interpretation:

- for the fast FP4 fullgrid paths (`regular_fa4_fused`, `production_fullgrid`, and roughly MM2), almost the entire post-QK penalty is already present by the first active tile
- the production/regular floor is effectively:
  - `fused QK-only no-store`
  - plus about `0.1086 ms`
- direct behaves differently:
  - it pays roughly one tile-sized post-QK penalty per active tile
  - that is consistent with the earlier view that direct carries the expensive split-accum/output-scratch behavior
- MM2 is closer to the production/regular floor than to direct:
  - `mm2_full_over_direct_full ≈ 0.655`
  - but MM2 is still not beating the simpler production/regular fullgrid floor

So the current optimization target is more specific now:

- there is a common `~0.109 ms` post-QK fullgrid floor shared by the fastest FP4 paths
- reducing that common post-QK path is higher-value than further work on direct, and likely higher-value than small MM2-vs-production tweaks until that shared floor moves

## 2026-04-13: native post-QK mainpath decomposition

I added a new canonical helper/writer in `fp4_pv_experiments.py`:

- `benchmark_fp4_post_qk_mainpath_decomposition_canonical(...)`
- `write_fp4_post_qk_mainpath_decomposition_canonical(...)`

This reuses the existing native stored-`P` PV kernels instead of timing Python orchestration:

- full `pv_from_p(...)`
- `two_tile_pv_mainpath_debug(...)` for:
  - tile 0 only
  - tiles 0+1 with second tile `mma2`
  - tiles 0+1 with second tile `mm2`
- the existing live localCTA tile-0 slices and fullgrid variants

Canonical writer smoke:

- output dir:
  - [fp4_post_qk_mainpath_decomposition_canonical_20260413T215907Z](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_post_qk_mainpath_decomposition_canonical_20260413T215907Z)

Representative canonical result on:

- `seqlen=4096`
- `batch=1`
- `heads=12`
- `input_mode='random_live_fp4'`
- `device='cuda:1'`
- `warmup=1`
- `iters=5`

Cases:

- BF16 baseline:
  - `0.142112 ms`
- fused QK-only no-store:
  - `0.176992 ms`
- full `pv_from_p`:
  - `0.344032 ms`
- stored-`P` mainpath tile 0 only:
  - `0.074880 ms`
  - reference diff is small in mean (`0.001640`) but not exact in max (`0.96875`), so treat this as a performance probe, not a correctness oracle
- stored-`P` mainpath tiles 0+1 mixed (`mm2` then `mma2`):
  - `0.086592 ms`
- stored-`P` mainpath tiles 0+1 all-`mm2`:
  - `0.084736 ms`
- direct tile 0 only:
  - `0.283520 ms`
- MM2 tile 0 only:
  - `0.258400 ms`
- regular fused fullgrid:
  - `0.261280 ms`
- production fullgrid:
  - `0.246272 ms`
- MM2 fullgrid:
  - `0.258720 ms`
- direct fullgrid:
  - `0.408096 ms`

Derived post-QK numbers:

- shared fast full post-QK floor (`min(regular, production) - qk_only`):
  - `0.069280 ms`
- MM2 full post-QK:
  - `0.081728 ms`
- direct full post-QK:
  - `0.231104 ms`
- direct tile0 post-QK:
  - `0.106528 ms`
- MM2 tile0 post-QK:
  - `0.081408 ms`

Useful compare numbers:

- MM2 tile0 post-QK minus stored-`P` mainpath tile0:
  - `0.006528 ms`
- direct tile0 post-QK minus stored-`P` mainpath tile0:
  - `0.031648 ms`
- second stored-`P` tile adds very little:
  - mixed extra over tile0: `0.011712 ms`
  - all-`mm2` extra over tile0: `0.009856 ms`
- second-tile `mm2` vs `mma2` delta in that stored-mainpath probe:
  - `-0.001856 ms`

Interpretation:

- the native stored-`P` tile-mainpath cost is already very close to the live MM2 tile-0 post-QK cost
- direct still carries a real extra penalty beyond that baseline
- the fast fullgrid paths are not being dominated by full-sequence `pv_from_p(...)`; that kernel is actually much slower than the shared live post-QK floor on the same case
- the first tile dominates the native PV mainpath cost too; adding the second stored tile only changes timing by about `0.01 ms`

So the next speed target is narrower now:

- do not spend time on direct for speed
- do not treat full `pv_from_p(...)` as the right proxy for the fused live post-QK floor
- treat `MM2 tile0 post-QK - stored-P tile0 mainpath` as the current small but real live-path overhead to shave
- separately, the regular/production fullgrid floor being even lower than the stored-mainpath probe suggests there is still a path/layout difference worth reading in the production fullgrid consumer path

## 2026-04-13: Current Canonical Number And Failed Fence Probes

I continued from the streaming-vs-production consumer diff and tried two low-risk kernel cuts:

- kept a compile-time `NEED_NONFIRST_MM2_GUARD` gate in `kernel_streaming_live_fp4pv(...)` so the `p_softmax_warp_needs_rescale` / `p_nonfirst_mm2_ok` bookkeeping only executes for static `consumer_mode=4` (`auto`)
- tried removing the streaming-only `publish_cluster_shared_backing_if_needed<C>()` around the `P` stage to match the production kernel more closely
- tried flipping `FP4PV_DIAG_CLUSTER_FENCE_P_STAGE` to `false` in the base production kernel

Outcomes:

- the non-first-MM2 guard gate is safe and remains in the experiments kernel
- removing the streaming `P`-stage cluster publish is not safe as written: the canonical helper timed out, so that probe was reverted
- flipping `FP4PV_DIAG_CLUSTER_FENCE_P_STAGE` to `false` compiled and ran, but did not move the number, so that probe was also reverted

Current canonical measurements on `cuda:1` after restoring the safe kernel state:

- canonical case:
  - `seqlen=4096`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `warmup=1`
  - `iters=5`
- `benchmark_streaming_live_mm2_vs_direct_canonical(...)`:
  - direct fullgrid: `0.375456 ms`
  - MM2 fullgrid: `0.222560 ms`
  - `mm2_over_direct_fullgrid = 0.592773`
  - direct persistent-effective: `0.523552 ms`
  - MM2 persistent-effective: `0.318784 ms`
  - `mm2_over_direct_persistent_effective = 0.608887`
- `benchmark_fp4_vs_bf16_canonical(...)`:
  - BF16 baseline: `0.097184 ms`
  - `regular_fa4_fused/fullgrid`: `0.217472 ms`
  - `qk_pv_nvfp4_production_fullgrid`: `0.217600 ms`
  - `streaming_live_localcta_prod_tcgen_mm2/fullgrid`: `0.222496 ms`
  - `streaming_live_localcta_direct/fullgrid`: `0.376320 ms`
  - best FP4 fullgrid over BF16: `2.237735x`

Current conclusion:

- the best measured FP4 fullgrid floor is now about `0.217 ms`, but BF16 is still about `0.097 ms` on the same canonical case
- MM2 is clearly the right live-path speed candidate, and the MM2-vs-direct number improved again (`~0.593x`)
- the remaining blocker is still the common FP4 fullgrid floor versus BF16, not direct-path semantics
- the obvious streaming-only `P`-stage fence removal is not safe without a more careful synchronization rewrite

### Same-session candidate ranking check

I reran the canonical fullgrid candidate sweep at `S=4096, B=1, H=12, random_live_fp4, cuda:1` on the restored safe kernel state:

- `regular_fa4_fused`: `0.216448 ms`
- `qk_pv_nvfp4_production_fullgrid`: `0.217440 ms`
- `streaming_live_localcta_prod_tcgen_mm2`: `0.222112 ms`
- `streaming_live_localcta_direct_tcgenaccum`: `0.222080 ms`
- `streaming_live_localcta_prod_tcgen`: `0.223680 ms`
- `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.225952 ms`
- `streaming_live_localcta_prod_tcgen_auto`: `0.227904 ms`
- `streaming_live_localcta_direct`: `0.375552 ms`

Interpretation:

- there is no hidden faster live candidate already in-tree
- `mm2_synced` and `auto` are both slower than plain `mm2`
- `direct_tcgenaccum` is fast, but it is not viable as a replacement

### Same-session accuracy check for the fast probes

Canonical accuracy check at `S=4096` against the stored-`P` oracle:

- `regular_fa4_fused`:
  - mean abs diff: `0.015955`
  - max abs diff: `14.9375`
- `qk_pv_nvfp4_production_fullgrid`:
  - mean abs diff: `0.020569`
  - max abs diff: `17.0`
- `streaming_live_localcta_prod_tcgen_mm2`:
  - mean abs diff: `0.025162`
  - max abs diff: `17.0`
- `streaming_live_localcta_direct_tcgenaccum`:
  - mean abs diff: `0.048074`
  - max abs diff: `62.75`

So `direct_tcgenaccum` is still too inaccurate to promote, even though its timing is near MM2.

### Failed compile-time probes in the base kernel

I tested two remaining compile-time scheduling branches in `b300_causal/bf16_b300_mha_causal_fp4.cu` and reverted both:

- `FP4PV_DIAG_ISSUE_NEXT_QK_AFTER_PV = true`
  - rebuild succeeded
  - canonical best-FP4-over-BF16 stayed flat at about `2.24x`
  - reverted
- `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE = false`
  - rebuild succeeded
  - `kernel_fp4pv` codegen changed noticeably, but canonical timing did not improve
  - reverted

Net result of this pass:

- the tree is back on the previously measured safe configuration
- `regular_fa4_fused` remains the fastest safe FP4 fullgrid path
- the remaining work is no longer in the obvious compile-time diagnostic toggles; it is in the base FP4 kernel’s common hot path

## 2026-04-13: base-kernel common-path cleanup and reverted probes

I continued in `b300_causal/bf16_b300_mha_causal_fp4.cu` and only kept one low-risk cleanup in the final tree:

- removed dead shared semaphores:
  - `p_copy_done_remote[2]`
  - `p_quant_ready_remote[2]`
- these were declared and initialized in `kernel_fp4pv(...)` but had no waits/arrives anywhere in the file

Kept code change:

- file:
  - [bf16_b300_mha_causal_fp4.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu)
- effect on codegen:
  - `kernel_fp4pv` shared memory dropped from `1360 bytes` to `1328 bytes`
  - registers/spills were otherwise unchanged:
    - `128 regs`
    - `20 bytes spill stores`
    - `24 bytes spill loads`

### Probes tried and reverted in the same pass

1. Removed the final per-task wait on `p_copy_done[final_copy_buf]`

- location:
  - end of the group-3 per-task loop in `kernel_fp4pv(...)`
- result:
  - rebuild succeeded
  - canonical fullgrid timings regressed materially
- representative canonical result after that cut:
  - BF16: `0.101632 ms`
  - `regular_fa4_fused`: `0.240896 ms`
  - production: `0.242368 ms`
  - MM2: `0.246848 ms`
- conclusion:
  - not a win
  - reverted

2. Removed `fp4pv_zero_raw_scale_stage(p_sc_stage[buf], ...)` before packing/scaling

- rationale:
  - looked redundant because each row writes all 8 raw `P` scales and causal masking zeros invalid trailing groups
- result:
  - one run briefly showed `regular_fa4_fused = 0.216000 ms`
  - repeated canonical runs did not hold the win:
    - `0.217600 ms`
    - `0.218432 ms`
- conclusion:
  - not a robust improvement
  - reverted

### Final kept state after rebuild

Benchmark setup:

- helper:
  - `benchmark_fp4_vs_bf16_canonical(device='cuda:1')`
- case:
  - `seqlen=4096`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `seed=0`
  - `warmup=1`
  - `iters=5`

Current canonical result on the kept dead-semaphore-cleanup state:

- BF16:
  - `0.097248 ms`
- `regular_fa4_fused/fullgrid`:
  - `0.217952 ms`
- `qk_pv_nvfp4_production_fullgrid`:
  - `0.217568 ms`
- `streaming_live_localcta_prod_tcgen_mm2/fullgrid`:
  - `0.223680 ms`
- `streaming_live_localcta_direct/fullgrid`:
  - `0.376256 ms`
- best FP4 fullgrid over BF16:
  - `2.237249x`

Accuracy sanity on the same kept state via `benchmark_streaming_live_pv_matrix(...)` with `accuracy_seqlens=(4096,)`, `input_mode='random_live_fp4'`, `launch_mode='fullgrid'`:

- production vs stored-`P` oracle:
  - mean abs diff: `0.018825`
  - max abs diff: `17.0`
  - `lse_max_abs_diff = 0.0190592`
- MM2 vs stored-`P` oracle:
  - mean abs diff: `0.023209`
  - max abs diff: `17.0`
  - `lse_max_abs_diff = 0.0190592`
- direct vs stored-`P` oracle:
  - mean abs diff: `0.024895`
  - max abs diff: `22.125`
  - `lse_max_abs_diff = 0.0190592`

Current interpretation:

- the dead remote semaphore cleanup is safe but performance-neutral in practice
- removing the final `p_copy_done` wait is wrong for performance on this kernel
- removing the raw `P`-scale zero pass is not a stable win and should stay reverted
- the common FP4 fullgrid floor is still effectively unchanged at about `0.218 ms`
- the next credible optimization target is still deeper in the common post-QK `P`-stage / consumer mainpath, not obvious unused waits or one-pass zeroing

### Additional reverted barrier probes

I also tried removing the subgroup barriers immediately after the `cluster` publish fence on the hot `P`-stage handoffs:

- producer-side:
  - after `publish_cluster_shared_backing_if_needed<C>()`
  - before `warpgroup::arrive(p_pack_ready[buf])`
- quantizer-side:
  - after `publish_cluster_shared_backing_if_needed<C>()`
  - before the `p_remote_ready` / `p_quant_ready` publication

Results:

- removing only the quantizer-side post-cluster barrier:
  - canonical floor stayed flat
  - representative result:
    - BF16: `0.096864 ms`
    - `regular_fa4_fused`: `0.217728 ms`
    - production: `0.217568 ms`
    - MM2: `0.221920 ms`
- removing both post-cluster barriers:
  - still flat on the base-kernel floor
  - representative result:
    - BF16: `0.096864 ms`
    - `regular_fa4_fused`: `0.217248 ms`
    - production: `0.217568 ms`
    - MM2: `0.221408 ms`

Interpretation:

- these barriers are not the dominant source of the common `~0.218 ms` FP4 fullgrid floor
- any movement was within the same small timing noise band and not enough to justify keeping a weaker synchronization story
- both barrier cuts were reverted

### Final kept state after those probes

- only the dead remote semaphore cleanup remains in the kernel:
  - `p_copy_done_remote[2]`
  - `p_quant_ready_remote[2]`
- final canonical check on the kept state:
  - BF16: `0.096864 ms`
  - `regular_fa4_fused`: `0.217248 ms`
  - production: `0.217568 ms`
  - MM2: `0.221408 ms`
  - direct: `0.375808 ms`
  - best FP4 over BF16: `2.242815x`

## 2026-04-14: keep scrub-loop removal in scale finalization

I continued inside the same common post-QK path and revisited `finalize_localcta_p_scales_with_amax(...)`.

Kept change:

- removed the final per-scale scrub loop in
  - [bf16_b300_mha_causal_fp4.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu)
- deleted logic:
  - iterate over all `TOTAL_SCALES`
  - convert each `nvfp4_scale_t` back to `float`
  - zero any value that is negative or nonfinite

Reasoning for trying it:

- the producer path already guards nonfinite / nonpositive scale generation before this point:
  - `fp4pv_quantize_scores_group(...)` returns `0` scale on invalid groups / invalid `block_amax`
  - `fp4pv_quantize_scores_group_scale_only(...)` returns `0` on invalid `block_amax`
  - `fp4pv_zero_invalid_causal_groups(...)` explicitly zeroes invalid trailing groups
- so this looked like dead work in the hot common scale-finalize path

Build/codegen result:

- rebuild succeeded
- `kernel_fp4pv` register/smem shape stayed effectively unchanged:
  - `128 regs`
  - `1328 bytes smem`
- one side effect:
  - `kernel_quantize_p_from_scores_debug<..., false, false>` spill stores dropped slightly
    - `332 -> 328 bytes`

Canonical timing after the cut:

- helper:
  - `benchmark_fp4_vs_bf16_canonical(device='cuda:1')`
- case:
  - `seqlen=4096`
  - `batch=1`
  - `heads=12`
  - `input_mode='random_live_fp4'`
  - `warmup=1`
  - `iters=5`

Representative run:

- BF16:
  - `0.096992 ms`
- `regular_fa4_fused/fullgrid`:
  - `0.216000 ms`
- `qk_pv_nvfp4_production_fullgrid`:
  - `0.216160 ms`
- `streaming_live_localcta_prod_tcgen_mm2/fullgrid`:
  - `0.220192 ms`
- `streaming_live_localcta_direct/fullgrid`:
  - `0.369536 ms`
- best FP4 over BF16:
  - `2.226988x`

Repeat canonical checks stayed in the same improved band rather than snapping back to the older direct timing:

- run A:
  - BF16: `0.095104 ms`
  - `regular_fa4_fused`: `0.216448 ms`
  - production: `0.217888 ms`
  - MM2: `0.222336 ms`
  - direct: `0.369728 ms`
- run B:
  - BF16: `0.096768 ms`
  - `regular_fa4_fused`: `0.217472 ms`
  - production: `0.217440 ms`
  - MM2: `0.219808 ms`
  - direct: `0.369568 ms`

Interpretation:

- the common fullgrid floor did not move dramatically
- but the cut consistently helped the slower direct path by about `6 us`
- and it occasionally improves the best fullgrid row into the `~0.2160 ms` band

Accuracy sanity after the cut via `benchmark_streaming_live_pv_matrix(...)` on `accuracy_seqlens=(4096,)`, `input_mode='random_live_fp4'`, `launch_mode='fullgrid'`:

Representative run:

- `regular_fa4_fused` vs stored-`P` oracle:
  - mean abs diff: `0.012183`
  - max abs diff: `13.375`
  - `lse_max_abs_diff = 0.0190592`
- production vs stored-`P` oracle:
  - mean abs diff: `0.014389`
  - max abs diff: `17.0`
  - `lse_max_abs_diff = 0.0190592`
- MM2 vs stored-`P` oracle:
  - mean abs diff: `0.026318`
  - max abs diff: `30.625`
  - `lse_max_abs_diff = 0.0190592`
- direct vs stored-`P` oracle:
  - mean abs diff: `0.024378`
  - max abs diff: `22.375`
  - `lse_max_abs_diff = 0.0190592`

Second repeat of the same accuracy pass:

- `regular_fa4_fused` mean abs diff: `0.014992`
- production mean abs diff: `0.013152`
- MM2 mean abs diff: `0.023560`
- direct mean abs diff: `0.022222`

Current interpretation:

- the scrub-loop removal is a valid simplification and did not produce a clear correctness failure on the measured canonical cases
- the largest beneficiary is the direct path; the best FP4 fullgrid floor only moves slightly
- MM2 accuracy remains the weakest of the fast paths and still has larger max-error excursions than production/regular
- the main conclusion does not change:
  - the FP4 fullgrid floor is still roughly `~0.216-0.218 ms`
  - BF16 is still roughly `~0.096-0.097 ms`
  - the next worthwhile optimization target is still deeper in the post-QK consumer/mainpath, not more small fence cleanup

## 2026-04-14: producer-prescale branch validated, production rebuild mattered

I validated the producer-prescale/direct-finalize branch properly and rebuilt **both** extensions:

- `make -B -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal -j1`
- `make -B -C /workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments -j1`

The earlier mismatch between streaming and fullgrid was partly a stale-build problem: the experiments extension had picked up the shared-kernel change, but the production extension had not been rebuilt yet.

Kept kernel state now:

- dead `p_copy_done_remote` / `p_quant_ready_remote` cleanup stays
- scrub loop removal in `finalize_localcta_p_scales_with_amax(...)` stays
- producer-prescale branch stays:
  - `fp4pv_pack_scores_to_stage_and_scales(...)` pre-multiplies stored localCTA scales by `sg_val`
  - `fp4pv_store_scales_from_localcta_scan_row(...)` does the same
  - hot path uses `finalize_localcta_p_scales_direct(...)`
- removing `fp4pv_zero_raw_scale_stage(...)` from the hot path also stays

### Accuracy after rebuilding both extensions

`benchmark_streaming_live_pv_matrix(...)`, `input_mode='random_live_fp4'`, `launch_mode='fullgrid'`

At `S=2048`:

- seed 0:
  - direct vs stored-`P`: mean abs diff `5.50e-06`, max abs diff `0.01251`
  - MM2 vs stored-`P`: mean abs diff `3.23e-05`, max abs diff `0.02673`
  - production vs stored-`P`: mean abs diff `0.02859`
  - `regular_fa4_fused` vs stored-`P`: mean abs diff `0.03369`
- seed 1:
  - direct vs stored-`P`: mean abs diff `1.27e-06`, max abs diff `0.00421`
  - MM2 vs stored-`P`: mean abs diff `2.68e-05`, max abs diff `0.01624`

At `S=4096`, seed 0, **after rebuilding production**:

- direct vs stored-`P`: mean abs diff `1.05e-05`, max abs diff `0.01318`
- MM2 vs stored-`P`: mean abs diff `1.97e-05`, max abs diff `0.02539`
- production vs stored-`P`: mean abs diff `8.60e-04`, max abs diff `0.05127`
- `regular_fa4_fused` vs stored-`P`: mean abs diff `8.61e-04`, max abs diff `0.05127`

At `S=8192`, seed 0:

- direct vs stored-`P`: mean abs diff `2.22e-06`, max abs diff `0.00531`
- MM2 vs stored-`P`: mean abs diff `1.29e-05`, max abs diff `0.02795`

So the producer-prescale path is real. The big semantic improvement is not confined to the streaming kernel; it also fixes the production/fullgrid path once the production extension is rebuilt.

### Canonical speed on the kept state

Canonical case:

- `seqlen=4096`
- `batch=1`
- `heads=12`
- `input_mode='random_live_fp4'`
- `device='cuda:1'`
- `warmup=1`
- `iters=5`

Representative kept-state run:

- BF16: `0.099648 ms`
- `regular_fa4_fused/fullgrid`: `0.216544 ms`
- production fullgrid: `0.216224 ms`
- MM2 fullgrid: `0.226176 ms`
- direct fullgrid: `0.373888 ms`
- best FP4 over BF16: `2.169878x`

Repeat canonical timing stayed in the same band:

- run A:
  - BF16: `0.097664 ms`
  - `regular_fa4_fused`: `0.214144 ms`
  - production: `0.214464 ms`
  - MM2: `0.222272 ms`
  - direct: `0.372032 ms`
- run B:
  - BF16: `0.097248 ms`
  - `regular_fa4_fused`: `0.214112 ms`
  - production: `0.215808 ms`
  - MM2: `0.221728 ms`
  - direct: `0.371904 ms`

Interpretation:

- the producer-prescale branch plus zero-pass removal moves the best fullgrid floor from the old `~0.2175 ms` band into the `~0.2141-0.2165 ms` band
- this is a real but still modest speed gain, roughly `1-2%`
- the larger win from this branch is correctness, not speed

### Updated decomposition

`benchmark_fp4_fullgrid_cost_decomposition_canonical(...)` on the kept state:

- BF16: `0.097280 ms`
- fused QK-only no-store/LSE-only: `0.130208 ms`
- best FP4 fullgrid: `0.213856 ms`
- best-FP4 minus QK-only gap: `0.083648 ms`

This is better than the earlier `~0.108 ms` post-QK gap, but the hard limit is now explicit:

- the FP4 **QK-only** core is still `1.34x` slower than the full BF16 baseline
- so beating BF16 now requires improving both:
  - the common post-QK path
  - the shared FP4 QK fullgrid path itself

### Failed follow-up: pre-swizzled scale writes

I tried a more aggressive follow-up where the producer wrote localCTA scales directly in the final swizzled byte layout and the consumer-side swizzle pass became a no-op.

Result:

- correctness stayed fine
- but `kernel_fp4pv` spills jumped from `20/24` bytes to `176/140` bytes
- canonical speed regressed to roughly:
  - `regular_fa4_fused`: `~0.2245 ms`
  - production: `~0.2263 ms`

That experiment was reverted. The tree is **not** left in that state.

### Current conclusion

- the current kept branch is the best state so far:
  - much better accuracy
  - small but real fullgrid speed improvement
- fastest current FP4 fullgrid is still about `2.17x-2.20x` slower than BF16 on the canonical case
- the next credible optimization target is no longer just the PV-side post-QK path:
  - post-QK gap is down to `~0.084 ms`
  - but QK-only is still too slow versus BF16

### 2026-04-14 follow-up: dead scan-store cut kept, streaming zero-pass cut reverted

I continued from the rebuilt kept state and first fixed one leftover bad branch from the prior K-scale-staging attempt:

- reverted the remaining `stage_k_scale_tmem(...)` / `issue_qk_chunked_qsc_ksc_tmem(...)` callsites in `kernel_fp4pv(...)`
- rebuilt both:
  - `make -B -C b300_causal -j1`
  - `make -B -C b300_causal_fp4_experiments -j1`

After that, the canonical timing band was back where expected:

- BF16: `0.097440 ms`
- `regular_fa4_fused`: `0.213984 ms`
- production: `0.214080 ms`
- MM2: `0.222432 ms`
- direct: `0.372032 ms`
- best FP4 over BF16: `2.196x`

#### Kept change

I found one real dead path in the base fullgrid kernel under the current kept configuration:

- `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE == true`
- but the producer softmax loop still converted each normalized score quarter to BF16 and stored it into `p_bf16_scan`, even though the direct-row-update path never reads `p_bf16_scan`

Kept kernel edit in `b300_causal/bf16_b300_mha_causal_fp4.cu`:

- leave the `p_bf16_scan` allocation/layout in place
- but skip the per-quarter BF16 conversion + `store_scores_quarter_to_localcta_scan(...)` work when `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE` is true

I explicitly did **not** keep the more aggressive shared-memory-size cut, because `globals_fp4pv<C>::dynamic_shared_memory()` is shared with the experiments streaming kernel and that version caused a runtime illegal access there.

#### Result of the kept base-kernel cut

Rebuilt both extensions again and rechecked the canonical case:

- BF16: `0.097344 ms`
- `regular_fa4_fused`: `0.214208 ms`
- production: `0.213984 ms`
- MM2: `0.222176 ms`
- direct: `0.373120 ms`
- best FP4 over BF16: `2.198x`

So this is a safe cleanup, but not a meaningful new speed win on the canonical metric. The floor stays in the same `~0.214 ms` band.

Accuracy stayed on the good rebuilt branch:

- production vs stored-`P`: mean abs diff `8.59e-4`, max abs diff `5.05e-2`
- `regular_fa4_fused` vs stored-`P`: mean abs diff `8.60e-4`, max abs diff `5.13e-2`
- direct vs stored-`P`: mean abs diff `7.02e-6`, max abs diff `9.64e-3`
- MM2 vs stored-`P`: mean abs diff `2.00e-5`, max abs diff `2.54e-2`

#### Reverted follow-up

I also tried the analogous zero-pass removal in the experiments streaming-localCTA kernel:

- skipped `fp4pv_zero_raw_scale_stage(...)` on the `ONLINE=false` branch

That improved localCTA speed slightly:

- direct: about `0.3722 -> 0.3698 ms`
- MM2: about `0.2221 -> 0.2202 ms`

but it also worsened the stored-`P` accuracy materially on the canonical case, so it was reverted.

#### Current state after this follow-up

- kept:
  - producer-prescale branch
  - base-kernel `p_sc` zero-pass removal
  - dead remote semaphore cleanup
  - scrub-loop removal in `finalize_localcta_p_scales_with_amax(...)`
  - dead `p_bf16_scan` store removal in the base fullgrid direct-row-update path
- reverted:
  - pre-swizzled scale writes
  - K-scale TMEM pre-stage branch
  - streaming-localCTA non-online zero-pass removal

Current conclusion is unchanged:

- fastest FP4 fullgrid is still about `2.20x` slower than BF16 on the canonical case
- the base fullgrid floor is still the real blocker
- the next credible target remains the shared FP4 QK/fullgrid path, not another small PV-side cleanup

### 2026-04-14 follow-up: future-tile pack skip was a spill trap and was reverted

I tried one more low-risk-looking base-kernel cut in `kernel_fp4pv(...)`:

- when `idx > m_tile`, all groups are fully causal-invalid for that row
- I changed the hot path to skip the `fp4pv_pack_scores_to_stage_*` work on those future tiles and rely on `fp4pv_zero_invalid_causal_groups(..., valid_groups=0)` to zero the whole staged row

This looked reasonable on paper, but codegen got much worse:

- `kernel_fp4pv` spills jumped from `20/24` bytes to `160/144`

Canonical timing regressed accordingly:

- BF16: `0.097472 ms`
- `regular_fa4_fused`: `0.228352 ms`
- production: `0.228160 ms`
- MM2: `0.221792 ms`
- direct: `0.373216 ms`
- best FP4 over BF16: `2.275x`

That change was reverted.

After reverting and rebuilding both extensions, the canonical band returned to the expected state:

- BF16: `0.097184 ms`
- `regular_fa4_fused`: `0.213664 ms`
- production: `0.213568 ms`
- MM2: `0.221760 ms`
- direct: `0.371616 ms`
- best FP4 over BF16: `2.198x`

So the current kept state is still:

- producer-prescale branch
- base-kernel `p_sc` zero-pass removal
- dead remote semaphore cleanup
- scrub-loop removal in `finalize_localcta_p_scales_with_amax(...)`
- dead `p_bf16_scan` store removal in the base fullgrid direct-row-update path

and the failed future-tile pack-skip branch should stay reverted.

### 2026-04-14 follow-up: split `globals_fp4pv` by `p_bf16_scan` need, but it does not move the canonical floor

I implemented the cleaner shared-memory split instead of just skipping dead stores:

- `globals_fp4pv<C>` is now `globals_fp4pv<C, NEED_P_SCAN>`
- the offline base fullgrid kernel instantiates `globals_fp4pv<C, false>`
- the streaming localCTA kernel instantiates:
  - `globals_fp4pv<C, true>` for `ONLINE=true`
  - `globals_fp4pv<C, false>` for `ONLINE=false` because `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE=true`
- `p_bf16_scan` allocation is now conditional through a small `fp4pv_p_scan_view` helper, so the offline kernels no longer reserve scan backing just to never read it

This compiled cleanly after rebuilding both extensions.

Useful compile-time result:

- experiments streaming kernels now clearly split by shared-memory layout
  - `kernel_streaming_live_fp4pv<..., ONLINE=false, ...>`: `1344` or `1376` bytes smem depending on consumer mode
  - `kernel_streaming_live_fp4pv<..., ONLINE=true, ...>`: `1856` bytes smem
- base `kernel_fp4pv` now compiles as `globals_fp4pv<..., false>`

Canonical accuracy on the rebuilt state stayed clean at `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- production vs stored-`P`: mean abs diff `8.6068e-4`, max abs diff `5.0537e-2`
- `regular_fa4_fused` vs stored-`P`: mean abs diff `8.5906e-4`, max abs diff `5.1269e-2`
- direct vs stored-`P`: mean abs diff `4.3183e-5`, max abs diff `2.5391e-2`
- MM2 vs stored-`P`: mean abs diff `2.6947e-5`, max abs diff `2.5391e-2`

Canonical speed did **not** materially improve. Representative reruns on `cuda:1`:

- run 0:
  - BF16: `0.097440 ms`
  - `regular_fa4_fused`: `0.214208 ms`
  - production: `0.214080 ms`
  - MM2: `0.220256 ms`
  - direct: `0.371232 ms`
  - best FP4 over BF16: `2.197x`
- run 1:
  - BF16: `0.096896 ms`
  - `regular_fa4_fused`: `0.214304 ms`
  - production: `0.214176 ms`
  - MM2: `0.220288 ms`
  - direct: `0.371712 ms`
  - best FP4 over BF16: `2.210x`

So this is a real structural cleanup, not a real speed win.

Current conclusion after this cut:

- removing dead scan backing from the offline kernels is safe
- it is not enough to move the `~0.214 ms` production/fullgrid floor
- the next pass still has to attack the shared FP4 QK/fullgrid hot path itself, not more dead-layout cleanup

I also tested one follow-up cleanup on top of this:

- make `corr_vec_smem` optional in offline kernels too

That one was reverted immediately:

- `ptxas` shared-memory numbers did not change
- canonical timing stayed in the same band (`production ~0.2141 ms`, `MM2 ~0.2201 ms`)
- it only added template noise without a measurable effect

So the kept state from this branch is just the `globals_fp4pv<C, NEED_P_SCAN>` / conditional `p_bf16_scan` split.

### 2026-04-14 follow-up: two more post-QK micro-cuts tested and reverted

I tried two more low-risk hot-path cuts after the `p_bf16_scan` layout split.

#### 1. Skip the initial `tt_output` zero before first `mm2_ABt(...)`

Reasoning:

- `mm2_ABt` resolves to `mma2(..., acc=0)`, so the first PV issue is overwrite semantics
- both the base fullgrid kernel and the streaming localCTA kernel were still doing
  `zero_output_scratch_issue_lane(tt_output)` before the first `mm2_ABt(...)`

What happened:

- codegen got worse immediately
  - base `kernel_fp4pv` spills worsened from `20/24` to `32/104`
  - offline streaming kernels also picked up larger spill counts
- canonical speed did not improve
  - BF16: `0.097120 ms`
  - production: `0.213952 ms`
  - `regular_fa4_fused`: `0.213824 ms`
  - MM2: `0.222016 ms`
  - direct: `0.397984 ms`
- accuracy stayed finite, but there was no speed case for keeping it

This probe was reverted.

#### 2. Batch raw `P_sc` writes into packed 4-scale stores inside `fp4pv_pack_scores_to_stage_and_scales(...)`

Reasoning:

- the direct-row-update quantizer helper was writing eight scale bytes one-by-one
- the file already has `fp4pv_store_packed_scale_word(...)`, so it looked like a clean store-coalescing cleanup

What happened:

- codegen stayed on the normal branch
- canonical speed stayed flat or slightly worse
  - BF16: `0.097664 ms`
  - production: `0.215936 ms`
  - `regular_fa4_fused`: `0.215936 ms`
  - MM2: `0.222464 ms`
  - direct: `0.373504 ms`
- stored-`P` accuracy got worse, especially for the streaming localCTA variants
  - direct mean abs diff widened to `1.085e-4`
  - MM2 mean abs diff widened to `4.812e-5`

This probe was also reverted.

Current conclusion after these two checks:

- the tree is back on the prior good branch
- the obvious post-QK micro-cuts in `tt_output` init and raw-scale write packing are not where the remaining time is hiding
- the next credible optimization still needs to target the deeper shared FP4 fullgrid/QK+consumer mainpath rather than another small local cleanup

### 2026-04-14 follow-up: K-scale TMEM reuse probe reverted, canonical BF16 compare now includes the real QK-FP4 lower bound

I tried one more shared-QK cut in the base fullgrid path:

- stage `K_sc` once per K tile into TMEM and reuse it across the two softmax consumers in `kernel<config<..., false>>`

Result:

- `ptxas` immediately got worse on the nonpersistent base kernel:
  - `kernel<config<..., false>>` stayed at `1920` bytes smem but spilled `112/136`
- canonical timing did not improve:
  - `regular_fa4_fused` moved to about `0.2144 ms`
  - that is slightly worse than the restored branch

That probe was reverted.

The useful result from this follow-up is not a kernel win but a benchmark correction:

- I added the real production `QK fp4, V bf16` kernel to `benchmark_fp4_vs_bf16_canonical(...)`
- this is the actual lower bound for the “QK is fine” claim, and it is **not** the same as the earlier synthetic no-store QK probe

Canonical `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=3`:

- BF16 baseline: `0.097408 ms`
- `QK fp4 + V bf16` persistent: `0.083424 ms`
- `QK fp4 + V bf16` over BF16: `0.8564x`
- `QK fp4 + V bf16` vs BF16 output diff:
  - mean abs diff: `4.30023e-4`
  - max abs diff: `5.76172e-2`
  - `LSE` max abs diff: `1.11815e-1`

On the same canonical case, the full FP4-PV paths are still much slower:

- `regular_fa4_fused` fullgrid: `0.215392 ms` (`2.211x` over BF16)
- production fullgrid: `0.215712 ms` (`2.215x`)
- MM2 fullgrid: `0.222336 ms` (`2.283x`)

So the important correction is:

- the real production QK-FP4 path is already **faster** than BF16 on the canonical case
- the remaining gap is overwhelmingly in the FP4-PV mainpath, not in QK itself

I also updated `write_fp4_vs_bf16_canonical(...)` so this `QK fp4 + V bf16` row is part of the standard artifact instead of living as an ad hoc probe.

I then added one more canonical summary metric so the remaining PV-side target is explicit:

- `best_fp4_fullgrid_over_qk_fp4_v_bf16`
- `best_fp4_fullgrid_minus_qk_fp4_v_bf16_ms`

On the same canonical helper run (`warmup=1`, `iters=3`), the current numbers are:

- BF16 baseline: `0.097696 ms`
- `QK fp4 + V bf16` persistent: `0.082944 ms`
- best full FP4-PV case: `production_fullgrid = 0.214624 ms`
- best full FP4-PV over `QK fp4 + V bf16`: `2.5876x`
- best full FP4-PV minus `QK fp4 + V bf16`: `0.131680 ms`

So the most useful optimization target number right now is no longer “beat BF16 somehow”; it is:

- remove roughly `0.132 ms` from the FP4-PV post-QK mainpath on the canonical case

That is the number to compare against in the next kernel pass.

### 2026-04-14 follow-up: V-scale TMEM staging move reverted

I tested one more post-QK kernel change in `b300_causal/bf16_b300_mha_causal_fp4.cu`:

- move `V_sc` TMEM staging out of the issue warp and into the producer/load warp inside `kernel_fp4pv(...)`
- goal: stop the issue warp from calling `load_k_scale_chunk(...)` on `v_sc_smem[...]` every PV iteration

Build result:

- both `b300_causal` and `b300_causal_fp4_experiments` rebuilt cleanly
- `ptxas` did **not** get worse:
  - `kernel_fp4pv` stayed at `20` spill stores / `24` spill loads
  - streaming experiment kernels stayed on their prior spill profile

Canonical timing result on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- quick pass (`warmup=1`, `iters=3`):
  - BF16: `0.098720 ms`
  - `QK fp4 + V bf16`: `0.082784 ms`
  - production fullgrid: `0.234816 ms`
  - `regular_fa4_fused`: `0.234464 ms`
  - MM2 fullgrid: `0.222080 ms`
  - direct fullgrid: `0.371872 ms`
- steady pass (`warmup=1`, `iters=5`):
  - BF16: `0.097088 ms`
  - `QK fp4 + V bf16`: `0.082592 ms`
  - production fullgrid: `0.234208 ms`
  - `regular_fa4_fused`: `0.234208 ms`
  - MM2 fullgrid: `0.220160 ms`
  - direct fullgrid: `0.371648 ms`

Interpretation:

- the real QK lower bound stayed flat
- MM2 stayed roughly flat
- the base fullgrid FP4-PV path regressed from the prior `~0.215 ms` band into the `~0.234 ms` band
- this is a real regression, not startup noise

I reverted the change. After rebuild, the canonical `warmup=1`, `iters=5` run returned to the prior floor:

- BF16: `0.096896 ms`
- `QK fp4 + V bf16`: `0.082592 ms`
- production fullgrid: `0.215360 ms`
- `regular_fa4_fused`: `0.215424 ms`
- MM2 fullgrid: `0.220192 ms`
- direct fullgrid: `0.371296 ms`
- best full FP4-PV minus `QK fp4 + V bf16`: `0.132768 ms`

Conclusion:

- moving `V_sc` TMEM staging off the issue warp does not help
- the remaining `~0.133 ms` PV-side gap is not coming from this obvious per-iteration `V_sc` staging point
- the next kernel pass should keep targeting the deeper FP4-PV post-QK mainpath

### 2026-04-14 follow-up: direct-swizzled `P_sc` write probe reverted, candidate sweep refreshed

I tried one more direct-path cut in `b300_causal/bf16_b300_mha_causal_fp4.cu`:

- write `P_sc` directly in final swizzled localCTA layout inside `fp4pv_pack_scores_to_stage_and_scales(...)`
- make `finalize_localcta_p_scales_direct(...)` a no-op
- use a swizzled invalid-causal zero helper for the direct row-update branch

Goal:

- remove the full-tile `swizzle_scales_row_inplace_group<128>(...)` pass from the active fast FP4-PV path

Result:

- this immediately made `kernel_fp4pv` codegen worse:
  - before: `20` spill stores / `24` spill loads
  - after: `152` spill stores / `128` spill loads
- I reverted the change without keeping a timing run on the bad codegen branch

After rebuild, the base kernel returned to the prior good profile:

- `kernel_fp4pv`: `20` spill stores / `24` spill loads

Quick canonical restore check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=3`:

- BF16: `0.097152 ms`
- `QK fp4 + V bf16`: `0.083040 ms`
- production fullgrid: `0.215616 ms`
- `regular_fa4_fused`: `0.215616 ms`
- MM2 fullgrid: `0.222240 ms`
- direct fullgrid: `0.371808 ms`
- best full FP4-PV minus `QK fp4 + V bf16`: `0.132576 ms`

So the direct-swizzled-scale idea is not viable in its obvious form; it saves a swizzle pass conceptually but explodes register pressure.

I then reran the canonical fullgrid candidate sweep against the stored-`P` oracle on the restored branch (`warmup=1`, `iters=3`):

- `qk_pv_nvfp4_production_fullgrid`: `0.214080 ms`, mean abs diff `8.685e-4`
- `regular_fa4_fused`: `0.216160 ms`, mean abs diff `8.680e-4`
- `streaming_live_localcta_prod_tcgen_mm2`: `0.220224 ms`, mean abs diff `2.069e-5`
- `streaming_live_localcta_direct_tcgenaccum`: `0.220512 ms`, mean abs diff `2.683e-5`
- `streaming_live_localcta_prod_tcgen`: `0.220512 ms`, mean abs diff `2.795e-5`
- `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.222048 ms`
- `streaming_live_localcta_prod_tcgen_auto`: `0.226336 ms`
- `streaming_live_localcta_direct`: `0.371520 ms`

Two useful corrections from that sweep:

- there is still no hidden existing variant below the current production/fullgrid floor
- the old “direct_tcgenaccum is too inaccurate to consider” result is stale on the current branch; it is numerically close now, but it is still slower than production/fullgrid and does not change the ranking

I also checked production launch mode directly on the same canonical case (`warmup=1`, `iters=5`):

- production persistent: `0.214080 ms`
- production fullgrid: `0.213664 ms`

So there is no meaningful free launch-mode win left there either. The fastest existing FP4-PV path is still production fullgrid at about `0.214 ms`, and the remaining gap above the real `QK fp4 + V bf16` lower bound is still about `0.133 ms`.

### 2026-04-14 follow-up: idle quantizer-warpgroup removal reverted, restored branch revalidated

I tried a larger structural cut in `b300_causal/bf16_b300_mha_causal_fp4.cu`:

- make `config_fp4pv::NUM_QUANTIZERS = 0` when `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE`
- remap producer/corrector/softmax warpgroup ids from `TOTAL_WGS`
- compile out the final quantizer branch entirely in the direct-row-update case

That was the wrong direction. `ptxas` regressed immediately on `kernel_fp4pv`:

- before: `128` regs, `20` spill stores / `24` spill loads, `4` barriers
- attempted restructure: `168` regs, `36` spill stores / `64` spill loads, `3` barriers

I reverted the structural change before benchmarking it further. After rebuild, the base and experiments extensions returned to the expected codegen:

- `kernel_fp4pv`: `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem

Restored canonical run on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=5`:

- BF16: `0.098464 ms`
- `QK fp4 + V bf16`: `0.082784 ms`
- `regular_fa4_fused`: `0.211424 ms`
- production fullgrid: `0.211648 ms`
- MM2 fullgrid: `0.221920 ms`
- direct fullgrid: `0.373664 ms`
- best full FP4-PV over BF16: `2.147x`
- best full FP4-PV minus `QK fp4 + V bf16`: `0.128640 ms`

Accuracy on the restored branch stayed in-family:

- production vs stored-`P`: mean abs diff `1.307e-4`, `lse_max_abs_diff = 0.0160704`
- MM2 vs stored-`P`: mean abs diff `1.307e-4`, `lse_max_abs_diff = 0.0160704`
- direct vs stored-`P`: mean abs diff `2.582e-4`, `lse_max_abs_diff = 0.0160704`

Conclusion:

- the packed/swizzled direct-path cleanup remains the correct branch to keep
- removing the idle quantizer warpgroup by changing `TOTAL_WGS` is not a free win; it pushes `kernel_fp4pv` into materially worse register pressure
- the next credible optimization target is still inside the FP4-PV post-QK mainpath, not warpgroup-count surgery

### 2026-04-14 follow-up: score-pack helper live-range reduction reverted

I tried one smaller helper-only cleanup in `fp4pv_pack_scores_to_stage_and_scales(...)`:

- shrink the temporary scale scratch from `row_scales[8]` to a reused `word_scales[4]`
- store each packed swizzled `P_sc` word immediately after every second quarter instead of keeping all 8 scale bytes live until the end

This also went the wrong way. It looked harmless, but it pushed `kernel_fp4pv` from:

- `20` spill stores / `24` spill loads

to:

- `192` spill stores / `148` spill loads

with the same `128` register cap. I reverted it immediately and rebuilt both extensions. The restored branch returned to:

- `kernel_fp4pv`: `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem

Conclusion:

- the score-pack helper is extremely sensitive to source-level reshaping
- shortening the apparent live range of the `P_sc` temporaries in C++ does not translate into better codegen here
- the next useful cuts still need to be judged first by `ptxas`, not by surface-level “less state” intuition

### 2026-04-14 follow-up: producer `P_sc`-before-remote wait reorder reverted

I tried one narrower producer-side overlap cut:

- keep the existing direct-row-update handoff shape
- in `wait_and_stage_p_sc(...)`, move the local `load_q_scale_chunk(...)` loop ahead of the `tma::cluster::wait(p_remote_ready[...])`

Reasoning:

- if `p_remote_ready` only gates the remote `P` payload visibility, the local `P_sc` TMEM load should be independent
- that would let the local scale load happen earlier without changing `kernel_fp4pv` shape

Result:

- `ptxas` stayed identical on the hot kernel:
  - `kernel_fp4pv`: `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem
- but the canonical timing was not a keepable win

Canonical `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=5` on the reordered branch:

- BF16: `0.098048 ms`
- `QK fp4 + V bf16`: `0.083648 ms`
- `regular_fa4_fused`: `0.212064 ms`
- production fullgrid: `0.212128 ms`
- MM2 fullgrid: `0.222752 ms`
- direct fullgrid: `0.374240 ms`

Compared with the prior restored branch (`~0.211424 / 0.211648 ms` on `regular_fa4_fused` / production), this was slightly worse in absolute FP4-PV time. I reverted it.

Conclusion:

- the remote wait is not the obvious overlap hole here
- even when codegen stays flat, this part of the producer path is sensitive enough that “more overlap” does not automatically translate into a measurable win
- the branch is restored to the previous kept state after this probe

### 2026-04-15 follow-up: row-local direct-pack amax probe reverted

I tried one direct-pack simplification in the base fullgrid kernel:

- for the `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE` path, skip the warpgroup-wide `amax` reduction used to build per-row `P_sc`
- instead, use each row lane's local `tile_max` directly as the scale source for direct packing

The logic was attractive, but the codegen result was bad. `ptxas` changed `kernel_fp4pv` from:

- `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem

to:

- `128` regs, `176` spill stores / `136` spill loads, `1280` bytes smem

That is not a viable trade. I reverted it immediately.

Conclusion:

- the obvious “remove warpgroup amax for direct pack” rewrite is not free in this kernel
- the direct pack path is sensitive to source reshaping even when the high-level work count goes down

### 2026-04-15 follow-up: packed-f32 swizzled `P_sc` conversion kept

I replaced the final `P_sc` byte-pack path in `fp4pv_pack_scores_to_stage_and_scales(...)` with a packed-f32 conversion helper:

- keep the existing quarter-level `amax` / score-pack flow
- accumulate two swizzled 4-wide packed scale words as `float`
- convert/store those packed words directly into the prepared `P_sc` tile with a dedicated helper, instead of materializing 8 `nvfp4_scale_t` bytes and then re-packing them

Kept codegen result:

- `kernel_fp4pv`: `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem

So this cut improved the hot path without destabilizing codegen.

Canonical kept result on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=5`:

- BF16: `0.096992 ms`
- `QK fp4 + V bf16`: `0.082880 ms`
- `regular_fa4_fused`: `0.209600 ms`
- production fullgrid: `0.210240 ms`
- MM2 fullgrid: `0.220480 ms`
- direct fullgrid: `0.371744 ms`
- best full FP4-PV over BF16: `2.161x`
- best full FP4-PV minus `QK fp4 + V bf16`: `0.126720 ms`

Stored-`P` oracle sanity stayed in-family on the kept branch:

- production vs stored-`P`: mean abs diff `8.768e-4`, `lse_max_abs_diff = 0.0160704`
- MM2 vs stored-`P`: mean abs diff `1.467e-4`, `lse_max_abs_diff = 0.0160704`
- direct vs stored-`P`: mean abs diff `2.602e-4`, `lse_max_abs_diff = 0.0160704`

Conclusion:

- this is a real keepable win in the shared FP4-PV post-QK path
- the remaining gap is still in the post-QK mainpath, but the floor moved in the right direction without a codegen penalty

### 2026-04-15 follow-up: analytic direct-path post-norm `tile_max` rewrite reverted

I tried one direct-path simplification after softmax normalization:

- keep the normalized `scores_reg` multiply loop
- stop recomputing `tile_max` by scanning all normalized values
- instead derive it analytically as `tile_max_pre_norm * inv_row_sum`

This did not touch `kernel_fp4pv`, but it was not safe for the broader shared kernel family. The shared `kernel<...ELb0>` path regressed from its prior codegen to:

- `48` byte stack frame
- `112` spill stores / `136` spill loads

That is not acceptable for the shared FP4 fullgrid baseline. I reverted it immediately.

Conclusion:

- even direct-path-only algebraic cleanups can perturb the shared non-`fp4pv` codegen enough to be a net regression
- this branch is not a safe way to attack the remaining FP4-PV gap

### 2026-04-15 follow-up: `st.shared.b64` payload-group store reverted

I tried collapsing `fp4pv_store_quantized_scores_group(...)` from two adjacent `st.shared.b32` instructions into one packed `st.shared.b64`.

On paper this should have reduced one shared-memory store per 8-score group, but `ptxas` reacted badly. It pushed:

- `kernel_fp4pv` from `20` spill stores / `24` spill loads

to:

- `192` spill stores / `148` spill loads

and widened spill pressure across the streaming kernels as well. I reverted it immediately and rebuilt both extensions.

Restored kept codegen after revert:

- `kernel_fp4pv`: `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem

Conclusion:

- the hot pack/store helpers are extremely sensitive to source-level store reshaping
- even a seemingly obvious store-count reduction can be a large codegen regression here
- the branch is restored to the packed-f32 scale-store keep state after this probe

### 2026-04-15 follow-up: restored keep-branch recheck

After reverting the failed `tile_max` and `st.shared.b64` probes, I rebuilt both extensions again and reran the canonical BF16 compare on the actual restored binaries.

Rechecked canonical `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=5`:

- BF16: `0.096736 ms`
- `QK fp4 + V bf16`: `0.082816 ms`
- `regular_fa4_fused`: `0.209728 ms`
- production fullgrid: `0.210112 ms`
- MM2 fullgrid: `0.221792 ms`
- direct fullgrid: `0.371872 ms`
- best full FP4-PV over BF16: `2.168x`
- best full FP4-PV minus `QK fp4 + V bf16`: `0.126912 ms`

So the tree is currently back on the packed-f32 swizzled-scale keep branch, and the live measurement still matches the earlier keep result.

### 2026-04-15 follow-up: first-tile direct-path correction handshake skip reverted

I tried one more tile-0-specific cut in `kernel_fp4pv`:

- in the direct-row-update path, skip the first per-task `corr_arrived -> rescale_finished` handshake
- keep later correction iterations and the task-end phase-priming handshake intact

The intent was to remove a no-op synchronization on the first live tile of each task. It did not work out. `ptxas` pushed `kernel_fp4pv` from:

- `20` spill stores / `24` spill loads

to:

- `180` spill stores / `152` spill loads

with the same `128`-register ceiling. I reverted it immediately and rebuilt both extensions.

Conclusion:

- the first-tile correction handshake is not a free micro-optimization in this kernel shape
- source-level semaphore/control simplification is still highly coupled to register allocation here
- the branch is restored to the packed-f32 swizzled-scale keep state after this probe

### 2026-04-15 follow-up: coefficient-arithmetic simplification reverted

I tried one smaller arithmetic-only cleanup inside `fp4pv_quantize_scores_group(...)` and `fp4pv_quantize_scores_group_payload_only(...)`:

- replace `1.0f / (s_b * (1.0f / s_enc))` with `s_enc / s_b`
- drop the explicit `s_dec`-related finite checks

This looked like a harmless algebraic simplification in the hot FP4 pack helper. It was not. `ptxas` pushed `kernel_fp4pv` from:

- `20` spill stores / `24` spill loads

to:

- `80` spill stores / `92` spill loads

with the same `128`-register ceiling. I reverted it immediately and rebuilt both extensions.

Restored kept codegen after revert:

- `kernel_fp4pv`: `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem

Conclusion:

- even local arithmetic cleanups in the quant helper are not free here
- the compiler is sensitive enough that algebraic simplification can still be a net codegen loss
- the branch is restored to the packed-f32 swizzled-scale keep state after this probe

### 2026-04-15 follow-up: encode-centric direct-pack quantization reverted

I tried switching the direct-path group quantizer over to the encode-centric NVFP4 formulation already used in the in-tree TK quantizer family:

- `S_mult_fp8 = compute_encoding_scaling_factor_nv(block_amax, s_enc)`
- `coeff = float(S_mult_fp8) * s_enc`
- stored decode scale rebuilt as `1.0f / float(S_mult_fp8)`

This only touched the direct `fp4pv_quantize_scores_group(...)` helper. It did not change any scheduling, payload layout, or synchronization.

Result:

- `kernel_fp4pv` regressed from `20` spill stores / `24` spill loads
- to `96` spill stores / `96` spill loads

with the same `128`-register ceiling. I reverted it immediately and rebuilt both extensions.

Restored kept codegen after revert:

- `kernel_fp4pv`: `128` regs, `20` spill stores / `24` spill loads, `1328` bytes smem

Conclusion:

- even borrowing a proven quantization formula from the TK helper family is not automatically profitable inside this kernel context
- the direct post-QK quant helper is still extremely sensitive to source-level arithmetic changes
- the branch is restored to the packed-f32 swizzled-scale keep state after this probe

### 2026-04-15 follow-up: non-direct row-update structural A/B reverted

I ran one larger structural A/B instead of another helper tweak:

- set `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE = false`
- rebuild the base kernel and experiments
- benchmark the canonical fullgrid path before deciding whether the alternate post-QK structure was worth keeping

This branch was interesting from a codegen perspective. The base fullgrid kernel changed from:

- `kernel_fp4pv`: `20` spill stores / `24` spill loads, `1328` bytes smem

to:

- `kernel_fp4pv`: `0` spill stores / `0` spill loads, `1840` bytes smem

So the non-direct row-update branch did materially improve the hot-kernel spill picture.

It still lost on runtime. Two canonical reruns on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` both landed around:

- `regular_fa4_fused`: `0.338-0.342 ms`
- production fullgrid: `0.334-0.337 ms`
- MM2 fullgrid: `0.279-0.287 ms`
- direct fullgrid: `0.436-0.449 ms`

Those runs also had slower absolute BF16 / `QK fp4 + V bf16` baselines (`~0.145 ms`), so I did not use them as absolute headline numbers. But the relative post-QK gap still worsened:

- best full FP4-PV minus `QK fp4 + V bf16`: `~0.131-0.142 ms`

compared with the kept direct-row-update branch:

- best full FP4-PV minus `QK fp4 + V bf16`: `0.126912 ms`

So this structural alternative is not a keep, despite the better spill profile. I reverted it and rebuilt both extensions back to the packed-f32 swizzled-scale keep branch.

Conclusion:

- zero spills on `kernel_fp4pv` alone are not sufficient; the non-direct post-QK structure is simply slower in practice on the canonical case
- the remaining target is still the direct-row-update post-QK mainpath, not the older scan-based branch

### 2026-04-15 follow-up: production persistent is not the hidden win

I restored the kept branch and checked the already-supported production launch modes before touching the kernel again.

First I revalidated the restored fullgrid baseline on the canonical case `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production` with `launch_mode='fullgrid'`: median `0.209920 ms`
- `qk_pv_nvfp4_production_fullgrid`: median `0.210208 ms`
- alias ratio: `1.00137x`

So the two fullgrid entrypoints are effectively the same path on the restored branch.

Then I checked `qk_pv_nvfp4_production` with `launch_mode='persistent'`.

Results:

- canonical `S=4096`, `H=12`: timed out while waiting for `_run_streaming_live_pv_variant_once timing` even with `timeout_ms=60000`
- smaller `S=4096`, `H=1`: also timed out with the same event wait
- small control `S=512`, `H=1`: persistent does run, but it is slightly slower than fullgrid
  - fullgrid: median `0.027232 ms`
  - persistent: median `0.027840 ms`
  - persistent/fullgrid: `1.02233x`

Conclusion:

- production persistent is not a viable hidden speed path on the shapes that matter
- at large `S=4096` it is not healthy enough to benchmark cleanly
- on a small single-launch control where it does run, it is still slower than fullgrid
- the next credible target remains the fullgrid FP4-PV post-QK mainpath, not production persistent launch mode

### 2026-04-15 follow-up: direct-path warp-local `tile_max` packing reverted

I tested one more structural cut in the direct-row-update fullgrid path: remove the CTA-wide `p_cta_amax` reduction and pack directly against the warp-local reduced `tile_max`.

That would have removed:

- the `p_softmax_warp_amax` shared write
- the two `subgroup_barrier_sync<128>()` calls around the CTA amax reduction
- the `p_cta_amax` shared handoff itself

So it was a real post-QK mainpath cut, not another arithmetic micro-tweak.

It failed immediately on codegen. On the base fullgrid kernel:

- kept branch: `kernel_fp4pv = 20` spill stores / `24` spill loads / `1328 B` smem
- warp-local-`tile_max` branch: `kernel_fp4pv = 176` spill stores / `136` spill loads / `1280 B` smem

The experiments build showed the same regression on `kernel_fp4pv`, while the streaming kernels stayed on their previous codegen.

That spill jump is too large to justify a canonical timing run, so I reverted the branch and rebuilt both `b300_causal` and `b300_causal_fp4_experiments` back to the kept state.

Conclusion:

- the CTA-wide `p_cta_amax` reduction is not free, but removing it this way increases live pressure enough to make the hot kernel strictly worse
- the remaining target is still inside the direct-row-update pack/quant mainpath, but not by simply swapping CTA-wide tile amax for warp-local tile amax

### 2026-04-15 follow-up: precomputing CTA-constant `s_enc/sg_val` reverted

I tested another direct-path pack/quant A/B that looked cleaner than the warp-local-`tile_max` branch:

- keep the existing CTA-wide `p_cta_amax` reduction
- compute `s_enc = fp4pv_compute_tile_s_enc(p_cta_amax)` once per tile in lane 0
- compute `sg_val = p_cta_amax * FP4PV_LOCALCTA_GLOBAL_SCALE_RCP` once per tile in lane 0
- store both in shared memory
- pass those shared values into `fp4pv_pack_scores_to_stage_and_scales(...)` instead of recomputing them in every thread

This does not change the quantization contract. It only removes redundant per-thread scalar recomputation in the direct path.

It still failed on codegen:

- kept branch: `kernel_fp4pv = 20` spill stores / `24` spill loads / `1328 B` smem
- precomputed-`s_enc/sg_val` branch: `kernel_fp4pv = 176` spill stores / `140` spill loads / `1328 B` smem

So this had the same failure mode as the warp-local-`tile_max` branch: fewer source-level ops, much worse live pressure in the hot kernel.

I reverted it immediately and rebuilt both `b300_causal` and `b300_causal_fp4_experiments` back to the kept state.

Conclusion:

- the direct pack helper is sensitive enough that even hoisting CTA-constant scalar work into shared can make `kernel_fp4pv` materially worse
- the remaining target is still the direct-row-update pack/quant mainpath, but not through shared precompute of `s_enc` / `sg_val`

### 2026-04-15 follow-up: fullgrid warpgroup register repartition reverted

I also tested one warpgroup register-partition A/B on the base fullgrid kernel, to see if the post-QK softmax/pack warpgroup could benefit from more dynamic registers without rewriting the helper code.

Change:

- correction warpgroup: `decrease_registers<48> -> decrease_registers<40>`
- softmax/pack warpgroup: `increase_registers<168> -> increase_registers<176>`

This was intended to give the direct-row-update pack/quant path a bit more headroom while taking it from the correction/output warpgroup.

It failed immediately on codegen:

- kept branch: `kernel_fp4pv = 20` spill stores / `24` spill loads / `1328 B` smem
- repartitioned branch: `kernel_fp4pv = 156` spill stores / `168` spill loads / `1328 B` smem

So this kernel is sensitive to warpgroup register partitioning too, not just helper structure. I reverted the change and rebuilt the base kernel back to the kept state.

Conclusion:

- the post-QK pack/quant path is not going to yield to simple register-partition retuning
- the remaining target is still a deeper primitive-level change in the direct-row-update pack/quant mainpath

### 2026-04-15 follow-up: direct payload + helper raw-scale finalize reverted

I tested one actual primitive-level hybrid, not just another arithmetic/helper reshuffle:

- keep the direct-row-update payload path in the softmax warpgroup
- quantize payload there as usual
- write raw decode-scale bytes into the `P_sc` stage unswizzled
- route direct mode through `p_pack_ready`
- let the helper warpgroup run the existing localCTA scale finalize primitive (`swizzle + global-scale multiply`) before publishing `p_quant_ready`

This was the cleanest way to reuse an already-proven localCTA quantization primitive inside the direct path without reintroducing the full non-direct `p_bf16_scan` branch.

It still failed on codegen:

- kept branch: `kernel_fp4pv = 20` spill stores / `24` spill loads / `1328 B` smem
- hybrid raw-scale-finalize branch: `kernel_fp4pv = 172` spill stores / `124` spill loads / `1328 B` smem

So even this larger hybrid raises hot-kernel pressure enough to lose before timing. I reverted it and rebuilt the base kernel back to the kept state.

Conclusion:

- borrowing the localCTA scale-finalize primitive directly into the direct-row-update path is not a cheap win
- the pack/quant bottleneck is deeper than the final scale-swizzle step alone

### 2026-04-15 follow-up: split scale-word live range in direct pack helper reverted

I tested one narrower helper-level A/B inside `fp4pv_pack_scores_to_stage_and_scales(...)`:

- process `q=0,1`, store packed scale word 0 immediately
- then process `q=2,3`, store packed scale word 1

The goal was to shorten the lifetime of the packed-scale temporaries without changing the algorithm, payload path, or scale contract.

This one was interesting because the base hot-kernel codegen stayed flat:

- kept branch: `kernel_fp4pv = 20` spill stores / `24` spill loads / `1328 B` smem
- split-live-range branch: same `20` spill stores / `24` spill loads / `1328 B` smem

Canonical timing on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` was effectively flat:

- `regular_fa4_fused`: median `0.209344 ms`
- production fullgrid: median `0.209760 ms`

Stored-`P` sanity on the same branch also stayed in-family:

- production vs oracle mean abs diff: `8.7617e-4`
- production vs oracle `LSE` max abs diff: `1.6070e-2`

So there was no correctness problem, but there was also no material speed win. The deciding downside was on the debug/forensics side: the experiments build’s heaviest `kernel_quantize_p_from_scores_debug<..., false, false>` variant got slightly worse (`328/364 -> 336/368` spill bytes, stack `144 -> 160`), which is not worth carrying for a flat fast-path result.

I reverted the helper split and rebuilt both `b300_causal` and `b300_causal_fp4_experiments` back to the kept state.

Conclusion:

- shortening the packed-scale live range this way is not harmful to the hot kernel, but it is not a real speed win either
- given the slight debug-kernel regression, it is not worth keeping

### 2026-04-15 follow-up: opposite register-partition A/B reverted

I tested the opposite direction of the earlier warpgroup register-partition experiment:

- correction warpgroup: `decrease_registers<48> -> 56`
- softmax/pack warpgroup: `increase_registers<168> -> 160`

Unlike the earlier `40 / 176` repartition, this one actually improved the hot-kernel spill profile:

- kept branch: `kernel_fp4pv = 20` spill stores / `24` spill loads / `1328 B` smem
- `56 / 160` branch: `kernel_fp4pv = 8` spill stores / `16` spill loads / `1328 B` smem

So this was the first branch in a while that made `kernel_fp4pv` codegen materially better without increasing shared memory.

Canonical timing on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` still did not show a convincing runtime win:

- BF16 baseline: `0.097856 ms`
- `QK fp4 + V bf16`: `0.082848 ms`
- `regular_fa4_fused`: `0.210528 ms`
- production fullgrid: `0.209696 ms`
- MM2 fullgrid: `0.220512 ms`
- direct fullgrid: `0.372224 ms`
- best full-FP4-PV minus `QK fp4 + V bf16`: `0.126848 ms`

That lower-bound gap is effectively unchanged from the kept branch. I also tried to use `S=8192` as a tie-breaker, but the harness timed out before producing a decisive compare.

Given:

- better spills but no clear canonical runtime improvement
- no larger-shape confirmation

I reverted the repartition and rebuilt the base kernel back to the kept state.

Conclusion:

- better `kernel_fp4pv` spill numbers alone are still not sufficient to justify a keep
- the remaining target is still a deeper primitive/mainpath change, not another warpgroup register repartition

### 2026-04-15 follow-up: `prod_stage_only` streaming probe reverted

I tried to add a new experiments-only streaming localCTA mode whose only job was to measure the post-QK staging path without the actual `mm2_ABt(...)` issue:

- same `P_sc` / `V_sc` waits and TMEM staging as the MM2 path
- no tensorcore PV accumulate
- explicit semaphore completion to let the rest of the kernel retire

The intent was to get one new number:

- `stage_only_full_post_qk_ms`
- then compare it directly against `mm2_full_post_qk_ms`

This did **not** stabilize enough to keep.

What happened:

- first implementation hung because the no-op branch removed the tensor-op completion path but still relied on tensor-triggered semaphore commits
- I fixed the obvious local completion gaps
- I then fixed the matching cluster-remote completion gaps
- after that, a single one-shot launch of `streaming_live_localcta_prod_stage_only` would return, but warmed repeat timing still hung

So the branch never became trustworthy as a benchmark surface. I reverted the variant and restored the Python harness and experiments kernel to the prior kept state.

Verification on the restored branch:

- `python3 -m py_compile fp4_pv_experiments.py` passes again
- `make -B -C b300_causal_fp4_experiments -j1` passes again

Conclusion:

- the `stage_only` instrumentation idea is directionally useful, but the current streaming localCTA semaphore structure is too brittle for a quick no-op PV branch
- I did **not** keep the probe
- the tree remains on the last known-good optimized branch, with no new runtime win from this pass

### 2026-04-15 follow-up: full-future direct-tile shortcut reverted

I tried one more base-kernel cut in `kernel_fp4pv`: early-out the direct-row-update path on strictly future causal tiles (`idx > m_tile`) instead of loading TT scores and running the normal post-QK pack/quant path just to emit zeros.

The source-level idea was simple:

- skip the TT score load and softmax math for fully future tiles
- mark the correction handshake complete
- zero the staged FP4 payload / scales directly
- publish the same `p_remote_ready` / `p_quant_ready` semaphores and continue

This was **not** viable. After rebuilding `b300_causal` and `b300_causal_fp4_experiments`, the hot kernel regressed immediately:

- kept branch: `kernel_fp4pv = 20` spill stores / `24` spill loads / `1328 B` smem
- shortcut branch: `kernel_fp4pv = 88` spill stores / `92` spill loads / `1328 B` smem

I reverted the shortcut immediately and rebuilt both extensions.

Verification on the restored branch:

- `make -B -C b300_causal -j1` passes
- `make -B -C b300_causal_fp4_experiments -j1` passes
- restored codegen is back at `kernel_fp4pv = 20/24 spill bytes, 1328 B smem`

Conclusion:

- even an apparently cheap future-tile fast path can blow up the direct post-QK mainpath codegen
- this branch did **not** produce a keepable optimization
- the next credible work item is still a deeper primitive/mainpath change, not another local shortcut inside the current direct-row-update loop

### 2026-04-15 follow-up: direct pack helper now quantizes from `float2` half-groups

I kept one new pack/quant primitive change in the base kernel:

- added `fp4pv_quantize_scores_group_from_float2(...)`
- changed `fp4pv_pack_scores_to_stage_and_scales(...)` to quantize `q*16` score quarters as two `float2[8]` half-groups instead of first materializing a local `bf16_2[16]` array

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Relevant locations:

- `fp4pv_quantize_scores_group_from_float2(...)`
- `fp4pv_pack_scores_to_stage_and_scales(...)`

Codegen result after rebuild:

- base `kernel_fp4pv`: `20` spill stores / `28` spill loads / `1328 B` smem
  - previous kept branch was `20/24`
- experiments streaming kernels improved materially:
  - `consumer_mode=-1`: `80/92` instead of `184/164`
  - `consumer_mode=5`: `36/36` instead of `176/144`
  - `consumer_mode=4`: `36/36` instead of `152/128`
  - `consumer_mode=3`: `36/36` instead of `168/136`
  - `consumer_mode=2`: `36/36` instead of `168/136`
  - `consumer_mode=0`: `36/36` instead of `168/136`

Canonical timing checks on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `BF16`: `0.096864 ms`
- `QK fp4 + V bf16`: `0.082336 ms`
- `regular_fa4_fused`: `0.213312 ms`
- `qk_pv_nvfp4_production_fullgrid`: `0.213632 ms`
- `streaming_live_localcta_direct/fullgrid`: `0.360576 ms`
- `streaming_live_localcta_prod_tcgen_mm2/fullgrid`: `0.213088 ms`

Useful interpretation:

- MM2 improved materially versus the prior kept branch and is now essentially on top of the production/fullgrid floor
- direct also improved materially
- production/regular stayed in-family; I did not see a speed win there, but I also did not see an obvious correctness regression

Stored-`P` sanity on the kept branch:

- production vs oracle:
  - mean abs diff: `8.8501e-4`
  - `LSE` max abs diff: `1.6070e-2`
- MM2 vs oracle:
  - mean abs diff: `1.5163e-4`
  - `LSE` max abs diff: `1.6070e-2`
- direct vs oracle:
  - mean abs diff: `3.2043e-4`
  - `LSE` max abs diff: `1.6070e-2`

One note on measurement:

- the big in-process canonical helper still hit an event timeout on the streaming rows once on this branch
- direct/MM2 per-variant fullgrid timing and the subprocess-safe launch-mode helper both completed cleanly, so the live-path timing numbers above are based on those stable surfaces

Conclusion:

- this is the first recent primitive change that actually improves the live MM2/direct fullgrid path without blowing up the hot kernel
- I kept it

### 2026-04-15 follow-up: two direct-helper follow-on probes reverted

I tried two follow-on variants on top of the kept half-group helper branch. I did **not** keep either one.

1. Direct-register half-group quantizer

I replaced the temporary `bf16_2[8]` array inside `fp4pv_quantize_scores_group_from_float2(...)` with a direct register path:

- convert `float2[8]` to eight `bf16_2` scalars
- compute BF16-domain `amax` from those scalars
- build the two `uint64` payload operands directly

This looked like the right next step, but it regressed codegen:

- base `kernel_fp4pv`: `40/44` spill bytes instead of `20/28`
- streaming kernels also regressed from the kept branch

So I reverted it immediately.

2. `__noinline__` half-group helper

I then tried the opposite compile-shape cut: keep the half-group helper logic, but stop force-inlining `fp4pv_quantize_scores_group_from_float2(...)`.

This did reduce spill counts:

- base `kernel_fp4pv`: `16/20` spill bytes
- streaming kernels: as low as `16/20` or `20/24`

But it introduced a much worse tradeoff:

- base `kernel_fp4pv`: `560-byte` stack frame
- hot streaming kernels: also `544-560 byte` stack frames
- debug quantize kernel stack frame also got much larger

That is not a safe keep, so I reverted it too.

Restored branch after both reverts:

- base `kernel_fp4pv`: `20/28` spill bytes, `1328 B` smem
- streaming kernels back to the kept half-group branch profile:
  - `consumer_mode=-1`: `80/92`
  - `consumer_mode=5/4/3/2/0`: `36/36`

Conclusion:

- the kept half-group helper branch still stands
- the next credible target is no longer helper call-shape or direct-register reshaping
- the next useful line is the packed-scale encode/store side of the direct pack/quant mainpath

### 2026-04-15 follow-up: paired scale-word store rewrite reverted

I tried one more packed-scale-side cut in `fp4pv_pack_scores_to_stage_and_scales(...)`:

- remove the two temporary `float[4]` packed-scale arrays
- process quarters as `(0,1)` and `(2,3)` pairs
- write each packed swizzled scale word immediately after finishing its quarter pair

The intent was to shorten scale-word live ranges without touching payload quantization.

This branch was a hard codegen regression:

- base `kernel_fp4pv`: `88` spill stores / `72` spill loads
  - kept branch is `20/28`
- debug quantize kernel also widened further

So I reverted it immediately and rebuilt back to the kept half-group branch.

Restored branch after revert:

- base `kernel_fp4pv`: `20/28` spill bytes, `1328 B` smem
- streaming kernels back to:
  - `consumer_mode=-1`: `80/92`
  - `consumer_mode=5/4/3/2/0`: `36/36`

Conclusion:

- the obvious scale-word live-range rewrite is not viable in the current helper shape
- the next credible work item is still deeper in the packed-scale encode/store primitive, not another loop reshuffle around the existing helper

### 2026-04-15 follow-up: keep `e4m3x4` scale-pack helper

I tried one lower-level scale-pack primitive change in `fp4pv_packed_float_to_ue4m3(...)`:

- old helper: two `cvt.rn.satfinite.e4m3x2.f32` conversions plus `mov.b32 {lo, hi}`
- new helper: one `cvt.rs.satfinite.e4m3x4.f32` conversion using the same packed control operand pattern already used in the TK quant PTX helpers

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Relevant location:

- `fp4pv_packed_float_to_ue4m3(...)`

Codegen result after rebuild:

- base `kernel_fp4pv`: unchanged at `20/28` spill bytes, `1328 B` smem
- streaming kernels: unchanged from the kept half-group branch
  - `consumer_mode=-1`: `80/92`
  - `consumer_mode=5/4/3/2/0`: `36/36`

Quick canonical checks on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- production fullgrid: `0.213600 ms`
  - mean abs diff vs stored-`P`: `8.5831e-4`
  - `LSE` max abs diff: `1.6070e-2`
- MM2 fullgrid: `0.211552 ms`
  - mean abs diff vs stored-`P`: `1.5354e-4`
  - `LSE` max abs diff: `1.6070e-2`
- direct fullgrid: `0.358720 ms`
  - mean abs diff vs stored-`P`: `2.7466e-4`
  - `LSE` max abs diff: `1.6070e-2`

Interpretation:

- production stays effectively flat
- MM2 and direct both improve slightly on the same kept branch
- accuracy stays in-family

Conclusion:

- this is a small keepable win in the packed-scale encode/store primitive
- I kept the `e4m3x4` scale-pack helper

### 2026-04-15 follow-up: repaired bad revert in scale helper region

I broke the base source while reverting an unkept scale-store probe. The damage was local:

- `fp4pv_store_packed_scale_word_swizzled_f32(...)` lost its `template <typename PScTile>` declaration
- that caused the base and experiments rebuilds to fail with a parser cascade starting in the scale-helper block

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Fix:

- restored the missing template declaration
- rebuilt both `b300_causal` and `b300_causal_fp4_experiments`

Restored codegen after rebuild:

- base `kernel_fp4pv`: `20/28` spill bytes, `1328 B` smem
- streaming kernels:
  - `consumer_mode=-1`: `80/92`
  - `consumer_mode=5/4/3/2/0`: `36/36`

Conclusion:

- the tree is back on the kept half-group + `e4m3x4` scale-pack branch
- no optimization change from this step; this was a source repair only

### 2026-04-15 follow-up: raw packed-scale carry variant reverted

I tried one deeper packed-scale encode/store A/B inside
`fp4pv_pack_scores_to_stage_and_scales(...)`:

- carry the eight raw FP8 block scales as two packed `uint32_t` words
- defer the `* sg_val -> e4m3x4` conversion until the final swizzled store

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Result after base rebuild:

- kept branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
- test branch: base `kernel_fp4pv = 32/36` spill bytes, `1328 B` smem

Conclusion:

- this is not a keep
- I reverted it immediately and rebuilt back to the kept branch
- the remaining target is still deeper than raw-scale-word carry/repack inside the current helper

### 2026-04-15 follow-up: PTX `mul_cvt_4x` scale-pack helper reverted

I tried replacing the final direct scale-pack helper with the in-tree TK PTX primitive:

- old path: `fp4pv_packed_float_to_ue4m3(s0, s1, s2, s3, packed)`
- test path: `transformer_engine::ptx::mul_cvt_4x(fp8e4m3x4&, floatx4, floatx2{1,1})`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv`: unchanged at `20/28` spill bytes, `1328 B` smem
- streaming kernels: unchanged at
  - `consumer_mode=-1`: `80/92`
  - `consumer_mode=5/4/3/2/0`: `36/36`

Runtime check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- production fullgrid: `0.213792 ms`
- MM2 fullgrid: `0.211520 ms`
- direct fullgrid: `0.360992 ms`

Interpretation:

- MM2 stayed effectively flat
- production and direct were not improved
- not enough to justify churn

Conclusion:

- I reverted it and rebuilt both extensions back to the kept branch

### 2026-04-15 follow-up: batched swizzled causal-zeroing reverted

I tried one bounded structural cut in the direct-row-update post-pack cleanup:

- old path: `fp4pv_zero_invalid_causal_groups_swizzled(...)` writes each invalid scale byte individually
- test path: batch the swizzled scale-zeroing with `fp4pv_store_packed_scale_word_swizzled(...)` for 4-aligned suffixes, then fall back to byte stores for the remainder

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv`: unchanged at `20/28` spill bytes, `1328 B` smem

Canonical production check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- run 1: `0.213344 ms`, mean abs diff vs stored-`P` `8.5363e-4`
- run 2: `0.213824 ms`, mean abs diff vs stored-`P` `8.5450e-4`

Interpretation:

- compile-flat, but runtime stayed inside the existing noise band
- not enough evidence for a real win

Conclusion:

- I reverted it and rebuilt the base extension back to the kept branch

### 2026-04-15 follow-up: candidate ranking recheck on kept branch

I stopped changing the kernel and rechecked the existing fullgrid candidates on the current kept branch
using the stable subprocess timing path at `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `cuda:1`.

Speed recheck:

- `regular_fa4_fused`: `0.213792 ms`
- `streaming_live_localcta_direct_tcgenaccum`: `0.211744 ms`
- `streaming_live_localcta_prod_tcgen`: `0.211584 ms`
- `streaming_live_localcta_prod_tcgen_mm2`: `0.211520 ms`
- `streaming_live_localcta_prod_tcgen_auto`: `0.217312 ms`
- `streaming_live_localcta_direct`: `0.359392 ms`

Longer head-to-head (`warmup=2`, `iters=9`) for the two close candidates:

- `streaming_live_localcta_prod_tcgen`: median `0.211584 ms`
- `streaming_live_localcta_prod_tcgen_mm2`: median `0.211520 ms`

So the earlier apparent `prod_tcgen` lead was noise. MM2 remains the fastest live candidate on the kept branch.

Single-run stored-`P` accuracy check on the same inputs:

- `streaming_live_localcta_direct_tcgenaccum`
  - mean abs diff: `1.8163e-4`
  - max abs diff: `9.0332e-2`
  - `LSE` max abs diff: `1.6070e-2`
- `streaming_live_localcta_prod_tcgen`
  - mean abs diff: `1.9526e-4`
  - max abs diff: `5.6152e-2`
  - `LSE` max abs diff: `1.6070e-2`
- `streaming_live_localcta_prod_tcgen_mm2`
  - mean abs diff: `1.3102e-4`
  - max abs diff: `4.2969e-2`
  - `LSE` max abs diff: `1.6070e-2`

Conclusion:

- no candidate promotion from this recheck
- `streaming_live_localcta_prod_tcgen_mm2` stays the best speed/accuracy live candidate

### 2026-04-15 follow-up: full-future direct-tile shortcut reverted again

I retried a narrower version of the full-future direct-tile shortcut inside the direct-row-update branch:

- detect the uniform `idx > m_tile` case
- skip the `tile_max` reduction, CTA `amax`, and `fp4pv_pack_scores_to_stage_and_scales(...)`
- write the stage directly via `fp4pv_zero_invalid_causal_groups_swizzled(..., valid_groups=0)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Result after base rebuild:

- kept branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
- test branch: base `kernel_fp4pv = 64/76` spill bytes, `1328 B` smem

Conclusion:

- this is still a hard codegen regression
- I reverted it immediately and rebuilt back to the kept branch

### 2026-04-15 follow-up: zero-specialized payload-store helper reverted

I tried one smaller post-pack cleanup cut in the causal-invalid suffix path:

- add `fp4pv_store_quantized_scores_group_zero(...)` that writes an all-zero payload group with a single `st.shared.b64`
- use it only in
  - `fp4pv_zero_invalid_causal_groups(...)`
  - `fp4pv_zero_invalid_causal_groups_swizzled(...)`
  - `fp4pv_zero_invalid_causal_payload_groups(...)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv`: unchanged at `20/28` spill bytes, `1328 B` smem

Canonical production fullgrid check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- run 1: `0.213344 ms`, mean abs diff vs stored-`P` `8.5363e-4`
- run 2: `0.213952 ms`, mean abs diff vs stored-`P` `8.5750e-4`

Interpretation:

- compile-flat, but runtime stayed in the same `~0.2133-0.2140 ms` band
- not enough to justify keeping another helper variant

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-15 follow-up: `FP4PV_DIAG_SKIP_CORRECTION = true` reverted

I tried one higher-leverage mainpath A/B:

- flip `FP4PV_DIAG_SKIP_CORRECTION` from `false` to `true`
- keep the rest of the direct-row-update path unchanged

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- this was a real hot-kernel codegen win
- base `kernel_fp4pv` moved from `20/28` spill bytes to `0/0`
- shared memory stayed at `1328 B`

Canonical production fullgrid result on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- BF16 median: `0.097280 ms`
- production fullgrid median: `0.208160 ms`
- production vs stored-`P`:
  - mean abs diff: `8.5486e-4`
  - max abs diff: `5.0293e-2`
  - `LSE` max abs diff: `1.6070e-2`

Interpretation:

- on the canonical case, this looked like the first meaningful speed win in a while
- but the follow-up stability sweep timed out before completing
- I did not get enough clean coverage beyond the canonical case to justify keeping a semantic correction-path change

Conclusion:

- I reverted it and rebuilt back to the kept branch
- this is the strongest recent signal that the correction/rescale path is expensive, but the current skip-correction form is not safe enough to promote

### 2026-04-15 follow-up: noinline correction helper reverted

I tried one bounded codegen cut around the correction path itself:

- extract the TT-output correction/rescale loop into a separate
  `fp4pv_rescale_tt_output<C>(...)` helper
- mark that helper `__noinline__`
- keep the correction condition and semantics unchanged

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- this was an immediate regression
- base `kernel_fp4pv` widened from `20/28` spill bytes to `36/52`
- it also introduced a `24-byte` stack frame in the hot kernel

Conclusion:

- I reverted it immediately and rebuilt back to the kept branch
- the correction-path cost center is real, but splitting it into a noinline helper is not the right way to recover codegen

### 2026-04-15 follow-up: thresholded correction skip reverted

I tried one bounded follow-up to the full skip-correction A/B:

- keep the correction path enabled
- but only apply TT-output rescale for lanes with `correction < 0.98f`
- let lanes at or above that threshold use `1.0f` instead
- skip the whole warp rescale loop if all lanes are above the threshold

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `20/28` spill bytes, `1328 B` smem

Result:

- this looked bounded enough to test, but the longer `S=8192` production fullgrid spot check timed out
- so it reproduced the same basic instability pattern as the full skip-correction branch

Conclusion:

- I reverted it and rebuilt back to the kept branch
- the correction/rescale path remains the strongest recent cost signal, but the obvious thresholded shortcut is not safe enough either

### 2026-04-15 follow-up: outer correction-loop unroll removal reverted

I tried one semantic-preserving codegen cut:

- keep the correction/rescale path unchanged
- change the outer TT-output rescale loop from `#pragma unroll` to `#pragma unroll 1`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- this was a real hot-kernel codegen win
- base `kernel_fp4pv` moved from `20/28` spill bytes to `0/0`
- shared memory stayed at `1328 B`

Result:

- despite the codegen win, the canonical production warmup/timing path hit a `60000 ms` event timeout
- so this branch is not safe enough to keep

Conclusion:

- I reverted it and rebuilt back to the kept branch
- this is another sign that the correction/rescale loop is structurally expensive, but the obvious unroll suppression changes execution behavior badly enough that it cannot be promoted as-is

### 2026-04-15 follow-up: inner correction-loop unroll suppression reverted

I tried one smaller semantic-preserving codegen cut inside the correction loop:

- keep the outer TT-output rescale loop unchanged
- change only the inner `for (int ii = 0; ii < 8; ++ii)` multiply loop from `#pragma unroll` to `#pragma unroll 1`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- this was a hard regression
- base `kernel_fp4pv` widened from `20/28` spill bytes to `36/60`
- it also introduced a `96-byte` stack frame

Conclusion:

- I reverted it immediately and rebuilt back to the kept branch
- the correction/rescale path remains the strongest cost signal, but even the smaller unroll suppression is not viable

### 2026-04-16 follow-up: shared-backing correction A/B reverted at compile time

I tried one deeper correction-path A/B in the base kernel:

- replace the direct TT read/scale/write rescale loop with
  - `spill_output_to_backing_warp(...)`
  - a shared-tile rescale on `o_smem[0]`
  - `load_output_backing_warp(...)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Result:

- this did not get to codegen/runtime measurement
- the existing warp-level backing helpers are not directly reusable with the base kernel's BF16 `o_smem` tile as written
- compilation failed on BF16/float assignment and load conversions in the backing helpers

Conclusion:

- I reverted it immediately and rebuilt back to the kept branch
- the shared-backing route is still interesting, but it needs a purpose-built BF16 backing helper rather than a quick swap onto the existing float-typed helper path

### 2026-04-16 follow-up: purpose-built BF16 shared-backing correction helpers reverted

I followed up on the failed shared-backing swap with a purpose-built BF16 path:

- add `spill_output_to_backing_warp_bf16(...)`
- add `load_output_backing_warp_bf16(...)`
- add `rescale_output_tile_streaming_bf16(...)`
- swap the direct correction loop onto:
  - spill TT output to `o_smem[0]`
  - rescale the BF16 backing tile in shared
  - load back into TT output

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- this compiled, but it was catastrophically bad for codegen
- base `kernel_fp4pv` exploded from `20/28` spill bytes to `15370/29322`
- the hot kernel also grew a `13312-byte` stack frame

Conclusion:

- I reverted it immediately and rebuilt back to the kept branch
- the shared-backing correction route is not viable in this straightforward BF16-helper form

### 2026-04-16 follow-up: async backing helper correction A/B reverted at compile time

I tried one more correction-path A/B after the BF16-specific shared-backing failure:

- keep a small BF16 shared rescale helper
- but reuse the existing async-backing helpers
  - `spill_output_to_backing<C>(...)`
  - `load_output_backing<C>(...)`
- swap the direct correction loop onto:
  - spill TT output to `o_smem[0]`
  - rescale in shared
  - load back into TT output

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Result:

- compile failed immediately
- the existing async-backing helpers depend on `config::OUTPUT_CHUNK_COLS` / `config::OUTPUT_CHUNKS`
- `config_fp4pv<...>` does not define those members, so this path is not directly reusable in the FP4PV kernel

Conclusion:

- I reverted it immediately and rebuilt back to the kept branch
- rebuilt-state check returned to the known-good base-kernel codegen:
  - `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
- if the shared-backing correction route is revisited again, it needs a purpose-built helper matched to `config_fp4pv`, not the existing async-backing helpers

### 2026-04-16 follow-up: partial outer correction-loop unroll reverted

I tried one smaller semantic-preserving codegen nudge inside the direct correction loop:

- keep the correction/rescale path unchanged
- keep the inner `ii` multiply loop fully unrolled
- change the outer `col` loop from `#pragma unroll` to `#pragma unroll 2`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Result after base rebuild:

- kept branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
- test branch: base `kernel_fp4pv = 32/40` spill bytes, `1328 B` smem

Conclusion:

- partial outer unroll is another direct hot-kernel regression
- I reverted it immediately and rebuilt back to the kept branch

### 2026-04-16 follow-up: hoisted `row_invalid` direct-output store reverted

I tried one bounded A/B in the final direct output-store path after post-QK normalization:

- keep the TT-output load/store structure unchanged
- keep direct-row-update normalization unchanged
- but hoist the `row_invalid` branch out of the inner `i += 2` loop
  - invalid row: write packed zero `b64` words directly
  - normal row: run the existing `float2 -> bf16_2 -> st.shared.b64` path without the per-pair branch

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 16/20` spill bytes, `1328 B` smem

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.213504 ms`
- stored-`P` compare stayed in-family:
  - mean abs diff `8.576990803703666e-4`
  - max abs diff `6.689453125e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- even though hot-kernel codegen improved from `20/28` to `16/20`, runtime moved the wrong way versus the kept branch band
- I reverted it and rebuilt back to the kept branch

### 2026-04-16 follow-up: approximate `inv_row_sum` in direct pack path reverted

I tried one actual post-QK primitive swap in the direct pack path:

- keep the rest of the direct-row-update flow unchanged
- but replace
  - `const float inv_row_sum = (row_sum > 0.0f && isfinite(row_sum)) ? __frcp_rn(row_sum) : 0.0f;`
- with
  - `rcp.approx.ftz.f32`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this change was compile-flat on the hot kernel

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.213664 ms`
- stored-`P` compare stayed in-family:
  - mean abs diff `8.590877987444401e-4`
  - max abs diff `6.8359375e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- approximate reciprocal in the direct pack normalization path is flat-to-worse on runtime
- I reverted it and rebuilt back to the kept branch

### 2026-04-16 follow-up: direct swizzled scale-word store runtime check reverted

I followed up on the earlier compile-flat cleanup on the direct scale-pack path and ran the actual timing A/B:

- change `fp4pv_pack_scores_to_stage_and_scales(...)` to materialize
  - `tk_localcta::nvfp4_scale_t row_scales[8]`
- write them with
  - `fp4pv_store_packed_scale_word_swizzled(...)`
- instead of
  - `float packed_scale_word{0,1}[4]`
  - `fp4pv_store_packed_scale_word_swizzled_f32(...)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this path was compile-flat on the hot kernel

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.213472 ms`
- stored-`P` compare drifted slightly worse:
  - mean abs diff `8.851528400555253e-4`
  - max abs diff `6.884765625e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- even with flat hot-kernel codegen, this direct swizzled scale-word store path is not a convincing runtime win
- I reverted it and rebuilt back to the kept branch

### 2026-04-16 follow-up: one-pass `float2 -> bf16 -> amax -> pack` helper reverted

I tried one deeper quant primitive change in the direct pack path:

- keep the `float2` source path
- but collapse
  - `float2 -> bf16_2[8]`
  - rescan those 8 BF16 pairs for `amax`
  - quantize/pack
- into a one-pass helper that:
  - converts each `float2` to `bf16_2`
  - updates `amax_2x` during the same loop
  - then computes the quant coefficient and packs

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this was compile-flat on the hot kernel

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.213216 ms`
- stored-`P` compare stayed in-family:
  - mean abs diff `8.585359901189804e-4`
  - max abs diff `6.8359375e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- the runtime shift is too small to justify keeping the duplicated one-pass quant path
- I reverted it and rebuilt back to the kept branch

### 2026-04-16 follow-up: dedicated `corr_vec_smem` handoff for direct correction reverted

I tried one cleaner post-QK handoff change in the direct path:

- keep the direct-row-update correction flow unchanged semantically
- but stop overloading `max_vec_smem[0]` for the per-tile correction factor
- instead:
  - store `acc_scale` into `corr_vec_smem[0]`
  - have the consumer-side correction loop load from `corr_vec_smem[0]`
- `max_vec_smem[0]` would then stay reserved for row-sum state only

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- spills stayed flat:
  - base `kernel_fp4pv = 20/28`
- but shared memory increased:
  - kept branch: `1328 B`
  - test branch: `1840 B`

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.213696 ms`
- stored-`P` compare stayed in-family:
  - mean abs diff `8.582402369938791e-4`
  - max abs diff `6.8359375e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- using the dedicated correction vector did not help runtime
- the extra shared-memory footprint likely hurts more than the cleaner state split helps
- I reverted it and rebuilt back to the kept branch

### 2026-04-16 follow-up: `FP4PV_USE_MM2_NONFIRST = false` rechecked and reverted

I re-ran the older `FP4PV_USE_MM2_NONFIRST = false` A/B on the narrow production timing surface instead of the noisier broad harness.

Change tested:

- set
  - `FP4PV_USE_MM2_NONFIRST = false`
- so non-first PV issues use `mma2_ABt(...)` accumulation instead of fresh `mm2_ABt(...)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this knob is compile-flat on the hot kernel

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.213664 ms`
- accuracy regressed badly vs stored-`P`:
  - mean abs diff `0.0072378478944301605`
  - max abs diff `0.96875`
  - `LSE` max abs diff `0.017892837524414062`

Conclusion:

- this is not a viable trade
- it is flat on runtime and clearly worse on accuracy
- I reverted it and rebuilt back to the kept branch

### 2026-04-16 follow-up: `FP4PV_DIAG_ISSUE_NEXT_QK_AFTER_PV = true` rechecked and reverted

I re-ran the overlap-order knob on the narrow production timing surface.

Change tested:

- set
  - `FP4PV_DIAG_ISSUE_NEXT_QK_AFTER_PV = true`
- so each non-final iteration:
  - stages/scales and issues the current PV work first
  - then calls `issue_next_qk(...)`
- instead of issuing the next QK before the current PV issue

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this schedule knob is compile-flat on the hot kernel

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.224000 ms`
- stored-`P` compare stayed roughly in-family:
  - mean abs diff `8.70033516548574e-4`
  - max abs diff `6.103515625e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- delaying next-QK issue is a clear runtime regression
- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: approximate correction ratio reverted as unstable

I tried one deeper correction-path primitive swap in the direct-row-update path:

- keep the correction logic structure unchanged
- but replace
  - `correction = prev_contrib / row_sum`
- with
  - `rcp.approx.ftz.f32(row_sum)` followed by a multiply

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this was compile-flat on the hot kernel

Runtime result:

- the narrow production timing surface stopped returning on this A/B
- that is enough to treat it as unstable for this branch

Conclusion:

- approximate correction-ratio evaluation is not safe enough to keep
- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: reuse exact `inv_row_sum` for correction reverted

I tried one semantic-preserving cleanup in the direct post-QK path:

- compute
  - `inv_row_sum = __frcp_rn(row_sum)`
- once
- use it for both
  - `correction = prev_contrib * inv_row_sum`
  - score normalization
- instead of doing
  - `correction = prev_contrib / row_sum`
  - and then computing `inv_row_sum` separately

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this was compile-flat on the hot kernel

Canonical production fullgrid spot checks on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- run 1:
  - `qk_pv_nvfp4_production_fullgrid = 0.213280 ms`
  - mean abs diff `8.597810519859195e-4`
  - max abs diff `6.689453125e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`
- run 2:
  - `qk_pv_nvfp4_production_fullgrid = 0.213472 ms`
  - mean abs diff `8.59073072206229e-4`
  - max abs diff `6.8359375e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- this stayed inside the existing noise band and did not produce a convincing runtime win
- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: redundant `isfinite` guards removed and reverted

I tried one bounded arithmetic simplification in the direct post-QK path:

- remove explicit `isfinite(prev_contrib)` / `isfinite(row_sum)` checks from the correction gate
- remove explicit `isfinite(row_sum)` from the `inv_row_sum` gate
- rely on:
  - `prev_contrib > 0.0f`
  - `row_sum > 0.0f`
  - and the existing post-compute correction sanitization

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this was compile-flat on the hot kernel

Canonical production fullgrid spot check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `qk_pv_nvfp4_production_fullgrid = 0.213696 ms`
- stored-`P` compare stayed in-family:
  - mean abs diff `8.584674214944243e-4`
  - max abs diff `6.8359375e-2`
  - `LSE` max abs diff `1.7892837524414062e-2`

Conclusion:

- removing the explicit finite guards does not improve runtime
- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: early packed scale-word store reverted

I tried one bounded live-range cut inside `fp4pv_pack_scores_to_stage_and_scales(...)`:

- keep the current direct quant path
- but store the first packed scale word as soon as `q == 1` completes
- and store the second packed scale word as soon as `q == 3` completes
- instead of keeping both `packed_scale_word{0,1}` arrays live until after the loop

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- kept branch: base `kernel_fp4pv = 20/28`
- test branch: base `kernel_fp4pv = 28/28`

Conclusion:

- early scale-word stores are another immediate hot-kernel regression
- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: `FP4PV_DIAG_FULL_ZERO_NONFIRST = true` reverted as unstable

I tried the remaining mainpath zeroing knob on the narrow surface.

Change tested:

- set
  - `FP4PV_DIAG_FULL_ZERO_NONFIRST = true`
- so non-first PV issues explicitly call `zero_output_scratch_issue_lane<C>(tt_output)` before issuing

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
  - so this knob is compile-flat on the hot kernel

Runtime result:

- the narrow production timing surface stopped returning cleanly on this A/B
- that is enough to treat it as unstable for this branch

Conclusion:

- explicit non-first output zeroing is not safe enough to keep
- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: float-source `block_amax` in direct quant helper reverted as unstable

I tried one genuinely different direct quant primitive:

- keep the `float2 -> bf16_2` conversion
- but compute `block_amax` directly from the source `float2` values during that same loop
- then reuse the packed BF16 values for the final FP4 pack
- instead of rescanning the converted BF16 pairs with `abs_max_2x`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 16/20` spill bytes, `1328 B` smem
  - this was a real hot-kernel codegen improvement versus the kept `20/28` branch

Runtime result:

- the narrow production timing surface stopped returning cleanly on this A/B
- that is enough to treat the branch as unstable

Conclusion:

- even though the direct quant helper codegen improved, this float-source `block_amax` path is not stable enough to keep
- I reverted it and rebuilt back to the kept branch

### 2026-04-15 follow-up: 64-bit `V` payload copy reverted

I tried one producer-side bulk-copy A/B:

- switch `load_v_fp4_tile_from_global(...)` in the base kernel from 32-bit to 64-bit shared stores
- switch `load_v_fp4_tile_from_global_allthreads(...)` in the experiments build the same way

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`
- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `20/28` spill bytes, `1328 B` smem
- streaming live kernels stayed at the kept-branch profile

Canonical helper result on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=3`:

- direct fullgrid: `0.442752 ms`
- MM2 fullgrid: `0.276704 ms`
- `mm2_over_direct_fullgrid = 0.62496`
- direct persistent effective: `0.625408 ms`
- MM2 persistent effective: `0.432000 ms`

Conclusion:

- 64-bit `V` payload copy is not a win here
- I reverted it and rebuilt both extensions back to the kept branch

### 2026-04-15 follow-up: `FP4PV_USE_MM2_NONFIRST = false` reverted

I tried one base-kernel mainpath A/B:

- flip `FP4PV_USE_MM2_NONFIRST` from `true` to `false`
- let non-first PV issues use `mma2_ABt(...)` accumulation instead of a fresh `mm2_ABt(...)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `20/28` spill bytes, `1328 B` smem

Result:

- there was no codegen upside to justify keeping the semantic change
- the normal timing surfaces also did not return cleanly enough on this A/B to support promotion

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-15 follow-up: quarter-level `float2 -> bf16_2[16]` helper reverted

I tried one more helper-level A/B in the direct-row-update pack path:

- add `fp4pv_quantize_scores_quarter_from_float2(...)`
- convert one full quarter (`16 x float2`) to a local `bf16_2[16]` array once
- reuse that array for the two existing `fp4pv_quantize_scores_group(...)` calls
- replace the two per-quarter `fp4pv_quantize_scores_group_from_float2(...)` calls with the new quarter helper

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `20/28` spill bytes, `1328 B` smem
- but the streaming experiments kernels regressed badly:
  - kept branch:
    - `consumer_mode=-1`: `80/92`
    - `consumer_mode=5/4/3/2/0`: `36/36`
  - test branch:
    - `consumer_mode=-1`: `184/164`
    - `consumer_mode=5`: `176/144`
    - `consumer_mode=4`: `152/128`
    - `consumer_mode=3/2/0`: `168/136`

Interpretation:

- quarter-level conversion did not help the base hot kernel
- it materially worsened the actual live-P streaming kernels that matter for MM2/direct timing
- this is another dead end in the helper live-range direction

Conclusion:

- I reverted it and rebuilt both `b300_causal` and `b300_causal_fp4_experiments`
- the tree is back on the kept branch:
  - base `kernel_fp4pv = 20/28`, `1328 B` smem
  - streaming `consumer_mode=-1 = 80/92`
  - streaming `consumer_mode=5/4/3/2/0 = 36/36`

### 2026-04-15 follow-up: consumer staging order swap reverted

I tried one consumer-side A/B in the streaming live kernels:

- swap the per-iteration staging order from
  - `wait_and_stage_v_sc(...)`
  - `wait_and_stage_p_sc(...)`
- to
  - `wait_and_stage_p_sc(...)`
  - `wait_and_stage_v_sc(...)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`
- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Build/codegen result:

- compile-flat on both base and experiments
- base stayed at `kernel_fp4pv = 20/28`, `1328 B` smem
- streaming experiments stayed at:
  - `consumer_mode=-1 = 80/92`
  - `consumer_mode=5/4/3/2/0 = 36/36`

Canonical timing result from `benchmark_streaming_live_mm2_vs_direct_canonical(...)`
on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=3`:

- direct fullgrid: `0.422144 ms`
- MM2 fullgrid: `0.263744 ms`
- `mm2_over_direct_fullgrid = 0.62477`
- direct persistent effective: `0.598720 ms`
- MM2 persistent effective: `0.409248 ms`

Interpretation:

- compile shape did not change
- runtime got materially worse relative to the existing kept-branch canonical band
- so the consumer critical path does not improve by waiting/staging `P_sc` before `V_sc`

Conclusion:

- I reverted it and rebuilt both extensions
- the tree is back on the kept branch

### 2026-04-15 follow-up: 64-bit `V` payload copy reverted

I tried one more producer-side bulk-copy A/B:

- change `load_v_fp4_tile_from_global(...)` in the base kernel from `st.shared.b32` / `uint32_t` copies to `st.shared.b64` / `uint64_t`
- change `load_v_fp4_tile_from_global_allthreads(...)` in the experiments build the same way

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`
- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Build/codegen result:

- compile-flat on the hot kernels
- base stayed at:
  - `kernel_fp4pv = 20/28`, `1328 B` smem
- streaming live kernels stayed at:
  - `consumer_mode=-1 = 80/92`
  - `consumer_mode=5/4/3/2/0 = 36/36`

Canonical timing result from `benchmark_streaming_live_mm2_vs_direct_canonical(...)`
on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=3`:

- direct fullgrid: `0.442752 ms`
- MM2 fullgrid: `0.276704 ms`
- `mm2_over_direct_fullgrid = 0.62496`
- direct persistent effective: `0.625408 ms`
- MM2 persistent effective: `0.432000 ms`

Interpretation:

- the `V` payload copy is not the producer bottleneck in the way this A/B assumed
- wider copy granularity did not move codegen or runtime in the right direction

Conclusion:

- I reverted it and rebuilt both extensions
- the tree is back on the kept branch

### 2026-04-15 follow-up: 64-bit `V_sc` prepared copy reverted

I tried one producer-side `V_sc` staging primitive change:

- change `load_v_sc_prepared_from_global(...)` in the base kernel from a 32-bit word loop to a contiguous 64-bit copy
- change `load_v_sc_prepared_from_global_allthreads(...)` in the experiments file the same way

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`
- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Build/codegen result:

- base stayed flat:
  - `kernel_fp4pv = 20/28`, `1328 B` smem
- streaming live kernels stayed flat:
  - `consumer_mode=-1 = 80/92`
  - `consumer_mode=5/4/3/2/0 = 36/36`
- only the small debug two-tile kernels widened slightly, which was acceptable for the timing A/B

Canonical timing result from `benchmark_streaming_live_mm2_vs_direct_canonical(...)`
on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=3`:

- direct fullgrid: `0.448768 ms`
- MM2 fullgrid: `0.289056 ms`
- `mm2_over_direct_fullgrid = 0.64411`
- direct persistent effective: `0.650912 ms`
- MM2 persistent effective: `0.456800 ms`

Interpretation:

- compile shape did not move on the hot kernels
- runtime regressed materially on the canonical surface
- the `V_sc` prepared copy is not instruction-bound in the way this 64-bit copy was targeting

Conclusion:

- I reverted it and rebuilt both extensions
- the tree is back on the kept branch

### 2026-04-15 follow-up: `V_sc` TMEM stage before `v_remote_ready` reverted

I tried one more consumer-side overlap A/B:

- `load_v_sc_prepared_from_global(...)` is replicated per CTA and does not depend on `cta_rank`
- so I moved the `V_sc` TMEM staging loop inside `wait_and_stage_v_sc(...)` to happen immediately after `wait(v_arrived[...])`
- the `tma::cluster::wait(v_remote_ready[...])` stayed in place, but now happens after the local `V_sc` TMEM loads

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`
- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Build/codegen result:

- compile-flat on both base and experiments
- base stayed at `kernel_fp4pv = 20/28`, `1328 B` smem
- streaming experiments stayed at:
  - `consumer_mode=-1 = 80/92`
  - `consumer_mode=5/4/3/2/0 = 36/36`

Canonical timing result from `benchmark_streaming_live_mm2_vs_direct_canonical(...)`
on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=3`:

- direct fullgrid: `0.430272 ms`
- MM2 fullgrid: `0.271904 ms`
- `mm2_over_direct_fullgrid = 0.63194`
- direct persistent effective: `0.615264 ms`
- MM2 persistent effective: `0.422624 ms`

Interpretation:

- codegen did not move
- runtime regressed materially relative to the kept-branch MM2/direct band
- overlapping local `V_sc` TMEM loads ahead of the remote payload wait is not a win

Conclusion:

- I reverted it and rebuilt both extensions
- the tree is back on the kept branch

### 2026-04-15 follow-up: split payload-only/scale-only quant helper reverted

I tried one quant-primitive A/B in `fp4pv_quantize_scores_group_from_float2(...)`:

- old path: convert 8 `float2` values to `bf16_2[8]`, then call the combined helper
  - `fp4pv_quantize_scores_group(...)`
- test path: keep the same `bf16_2[8]` conversion, but replace the combined helper with
  - `fp4pv_quantize_scores_group_payload_only(...)`
  - `fp4pv_quantize_scores_group_scale_only(...)`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- kept branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
- test branch: base `kernel_fp4pv = 76/72` spill bytes, `1328 B` smem

Conclusion:

- this is another hard hot-kernel regression
- I reverted it immediately and rebuilt back to the kept branch

### 2026-04-15 follow-up: explicit full-future zero-fill loop reverted

I tried one smaller full-future-tile shortcut in the direct-row-update branch:

- detect `idx > m_tile`
- skip the expensive exp/tile-sum path
- explicitly zero `scores_reg` with a nested loop
- keep the existing downstream pack/quant path unchanged

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Result after base rebuild:

- kept branch: base `kernel_fp4pv = 20/28` spill bytes, `1328 B` smem
- test branch: base `kernel_fp4pv = 1360/744` spill bytes, `1328 B` smem

Conclusion:

- this is catastrophically bad for codegen
- I reverted it immediately and rebuilt back to the kept branch

### 2026-04-15 follow-up: fixed `tile_amax = 1` direct quantization reverted

I tried one more aggressive direct-row-update A/B:

- remove the per-tile `tile_max` warp reduction and CTA `amax` reduction entirely in the direct path
- call `fp4pv_pack_scores_to_stage_and_scales(...)` with `amax_val = 1.0f`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- test branch: base `kernel_fp4pv = 16/20` spill bytes, `1280 B` smem
  - this was the first branch in a while that actually improved hot-kernel codegen

Canonical production fullgrid result on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- test branch: `0.228576 ms`
- mean abs diff vs stored-`P`: `8.6236e-4`
- max abs diff vs stored-`P`: `6.6895e-2`
- `LSE` max abs diff vs stored-`P`: `1.7893e-2`

Interpretation:

- codegen got better
- runtime got materially worse
- so the tile-`amax` reduction is not the limiting cost in the way this experiment assumed

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: direct correction handoff remap to `lse_smem` reverted

I tried one bounded handoff cleanup in the direct correction path:

- keep the direct-path `acc_scale` handoff out of `max_vec_smem[0]`
- write it to `lse_smem[0]` instead
- load it back from `lse_smem[0]` in the TT-output rescale loop

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `20/28` spill bytes and `1328 B` smem

Runtime result:

- the narrow production timing surface did not produce a usable clean result on this branch
- the run exited through the harness without a stable production row, so I treated the branch as unstable

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: PTX `ex2.approx` direct-path exponentials reverted

I tried one deeper post-QK arithmetic A/B in the direct pack path:

- add a tiny `fp4pv_exp2_approx(...)` helper using `ex2.approx.ftz.f32`
- replace the direct-path `exp2f(...)` calls for
  - `acc_scale`
  - normalized tile exponentials in both the normal and `ZERO_SCALE` branches

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` improved from `20/28` spill bytes to `16/20`
- shared memory stayed `1328 B`

Runtime result:

- the narrow production timing surface stopped returning cleanly on this branch
- I treated it as unstable rather than keeping a codegen-only improvement without a usable timing/accuracy result

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: score-only `ex2.approx` direct-path exponentials reverted

I tried a narrower exponentiation A/B than the previous all-in `ex2.approx` branch:

- keep direct-path `acc_scale = exp2f(...)` exact
- replace only the per-score exponentials in the direct post-QK path with a tiny PTX helper
  - `fp4pv_exp2_approx(...)`
  - `ex2.approx.ftz.f32`
- apply it in both the normal scaled branch and the `ZERO_SCALE` branch

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` improved from `20/28` spill bytes to `16/20`
- shared memory stayed `1328 B`

Runtime result:

- I avoided the broader launch benchmark and ran a custom fullgrid-only production timing/check path
- that surface still did not return cleanly enough to trust on this branch
- so I treated the branch as unstable despite the better hot-kernel codegen

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-17 follow-up: score-only `ex2.approx` branch is runtime-unstable, not just slow

I reran the narrower direct-path exponentiation A/B with a lighter validation method:

- keep `acc_scale = exp2f(...)` exact
- replace only the per-score direct-path exponentials with
  - `fp4pv_exp2_approx(...)`
  - `ex2.approx.ftz.f32`

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` again improved from `20/28` spill bytes to `16/20`
- shared memory stayed `1328 B`

Validation result:

- I avoided the broader launch benchmark and ran a custom fullgrid-only production timing script
- the first sign of failure was not just “slow” timing
- after a warmup fullgrid launch, the next CUDA event record failed with:
  - `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable`
- `nvidia-smi` showed all GPUs idle and no active compute processes at the same time

Interpretation:

- this branch is runtime-unstable at the CUDA level
- it is not just a noisy benchmark surface problem

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-18 follow-up: unified direct-path `apply_softmax_scale(...)` loop reverted

I tried one structural simplification in the direct post-QK path:

- remove the duplicated `ZERO_SCALE` / non-`ZERO_SCALE` exponentiation loops
- use the existing `apply_softmax_scale(...)` helper in one unified loop
- keep the rest of the direct quant/pack path unchanged

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` improved from `20/28` spill bytes to `16/20`
- shared memory stayed `1328 B`

Validation result:

- event-timed fullgrid-only production timing failed with:
  - `CUDA error: CUDA-capable device(s) is/are busy or unavailable`
- host-timed fullgrid-only production timing under shell timeout also stalled and exited with code `124`

Interpretation:

- this branch is runtime-unstable despite the better `ptxas` result

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-18 follow-up: direct-path `softmax_rescale_factor(...)` helper substitution reverted

I tried one bounded semantic cleanup in the direct post-QK path:

- replace the inline direct-path `acc_scale` threshold logic with the existing
  `softmax_rescale_factor(row_max, row_max_old, SCALE_LOG2)` helper

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `20/28` spill bytes
- shared memory stayed `1328 B`

Interpretation:

- this is a zero-signal branch on the hot kernel
- I did not spend a runtime run on it

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-18 follow-up: analytic raw tile-amax branch reverted

I tried one bounded post-QK change aimed at the direct tile-amax scan:

- keep a copy of the pre-threshold current tile max
- stop updating `tile_max` with `fmaxf(...)` on every exponentiated element
- derive raw `tile_max` analytically after the loops from
  - `tile_score_max`
  - effective `row_max`
  - `SCALE_LOG2`
  - and the `ZERO_SCALE` case

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` regressed from `20/28` to `32/36`
- shared memory stayed `1328 B`

Interpretation:

- whatever work it removes in the hot loop is outweighed by extra live state/control in this form

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-18 follow-up: per-quarter tile-sum accumulation branch reverted

I tried one arithmetic-structure change in the direct post-QK path:

- keep the same score scaling, exponentiation, tile-max scan, and quant primitive
- break the long `tile_sum0` / `tile_sum1` dependency chains by accumulating per-quarter
  partial sums (`q_sum0`, `q_sum1`) and only folding them into the tile totals at quarter boundaries

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` improved from `20/28` spill bytes to `16/20`
- shared memory stayed `1328 B`

Validation result:

- a minimal one-warmup / one-run host-timed fullgrid check under shell timeout did not return
- the process hit the shell timeout (`124`)

Interpretation:

- this branch is not safe enough to keep despite the better `ptxas` result

Conclusion:

- I reverted it and rebuilt back to the kept branch

### 2026-04-18 follow-up: float-source `block_amax` direct-quant helper promoted to working branch

I revalidated the earlier float-source direct-quant helper with a stricter, lighter validation method and it is the first branch in this area worth keeping for now.

Change:

- in `fp4pv_quantize_scores_group_from_float2(...)`
  - convert each source `float2` to `bf16_2`
  - compute `block_amax` directly from the source float values during that same loop
  - avoid the separate BF16-pair `abs_max_2x` rescan

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- base `kernel_fp4pv` improved from `20/28` spill bytes to `16/20`
- shared memory stayed `1328 B`

Validation result on `cuda:1`, canonical `S=4096, B=1, H=12, random_live_fp4`:

- minimal host-timed fullgrid production check completed:
  - `timing_ms ~= 0.29795`
- stored-`P` compare stayed in-family:
  - `mean_abs_diff = 8.7676e-4`
  - `max_abs_diff = 6.8359e-2`
  - `lse_max_abs_diff = 1.7893e-2`

Scope note:

- when I tried to rerun the same minimal host-timed check on the restored previous branch for an apples-to-apples comparison, that surface itself hit the shell timeout on this machine
- so the promotion decision here is based on:
  - better hot-kernel codegen
  - one clean minimal fullgrid run
  - and in-family stored-`P` drift

Conclusion:

- I restored this branch and left it as the new working branch

### 2026-04-18 follow-up: shared helper keep propagated into live kernels

After restoring the float-source `block_amax` helper as the working branch, I rebuilt the experiments extension too:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Propagation/codegen result:

- base `kernel_fp4pv`: `16/20`, `1328 B` smem
- streaming live kernels now also inherit the lower-spill profile in the common path:
  - `consumer_mode=3` (`streaming_live_localcta_prod_tcgen_mm2`): `16/20`
  - `consumer_mode=0` (`streaming_live_localcta_direct_tcgenaccum`): `16/20`
  - `consumer_mode=2` (`streaming_live_localcta_prod_tcgen`): `16/20`
  - `consumer_mode=1` (`streaming_live_localcta_direct`): `0/0`

Minimal live-path snapshot on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- one warm call, then second-timing call:
  - `qk_pv_nvfp4_production_fullgrid`: `0.256448 ms`
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.260288 ms`
  - `streaming_live_localcta_direct`: `0.389728 ms`

Stored-`P` drift on the same branch stayed in-family:

- production fullgrid:
  - `mean_abs_diff = 8.7261e-4`
  - `max_abs_diff = 6.8359e-2`
  - `lse_max_abs_diff = 1.7893e-2`
- MM2 fullgrid:
  - `mean_abs_diff = 1.2101e-4`
  - `max_abs_diff = 4.4678e-2`
  - `lse_max_abs_diff = 1.7893e-2`
- direct fullgrid:
  - `mean_abs_diff = 3.0703e-4`
  - `max_abs_diff = 7.3730e-2`
  - `lse_max_abs_diff = 1.7893e-2`

Interpretation:

- the shared helper keep is now active in both production and live code paths
- MM2 remains the best live candidate on this branch

### 2026-04-18 follow-up: warmed live-candidate recheck on the float-source `block_amax` branch

I reran the close fullgrid live variants on the current working branch with one warm call and then a 3-sample timing median on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`.

Variants checked:

- `streaming_live_localcta_prod_tcgen_mm2`
- `streaming_live_localcta_prod_tcgen_auto`
- `streaming_live_localcta_prod_tcgen_mm2_synced`

Results:

- `streaming_live_localcta_prod_tcgen_mm2`
  - timing samples: `0.253536`, `0.246752`, `0.244352`
  - timing median: `0.246752 ms`
  - `mean_abs_diff = 9.5376e-5`
  - `max_abs_diff = 1.5747e-2`
  - `lse_max_abs_diff = 1.7893e-2`
- `streaming_live_localcta_prod_tcgen_auto`
  - timing samples: `0.251616`, `0.246432`, `0.242784`
  - timing median: `0.246432 ms`
  - `mean_abs_diff = 1.1412e-4`
  - `max_abs_diff = 2.2583e-2`
  - `lse_max_abs_diff = 1.7893e-2`
- `streaming_live_localcta_prod_tcgen_mm2_synced`
  - timing samples: `0.247904`, `0.249856`, `0.242176`
  - timing median: `0.247904 ms`
  - `mean_abs_diff = 9.3044e-5`
  - `max_abs_diff = 1.6113e-2`
  - `lse_max_abs_diff = 1.7893e-2`

Conclusion:

- after warmup, `auto`, `mm2`, and `mm2_synced` are effectively in the same timing band
- `auto` is marginally fastest on this small sample, but `mm2` remains the best speed/accuracy default because it is within noise on timing and slightly better than `auto` on output drift

### 2026-04-18 follow-up: prepared-scale return variant reverted

I tried one bounded follow-on in the current float-source `block_amax` helper:

- instead of returning `nvfp4_scale_t` and multiplying by `sg_val` in the caller
- make `fp4pv_quantize_scores_group_from_float2(...)` return the prepared float scale directly
  (`static_cast<float>(s_b_fp8) * sg_val`)

Files touched:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Build/codegen result:

- working branch: base `kernel_fp4pv = 16/20`
- test branch: base `kernel_fp4pv = 20/24`

Interpretation:

- folding the prepared-scale multiply into the helper increases hot-kernel pressure in this form

Conclusion:

- I reverted it and rebuilt back to the float-source `block_amax` working branch

### 2026-04-18 follow-up: streaming offline zero-pass removal reverted

I tested one narrower live-kernel A/B in the experiments extension only:

- keep `fp4pv_zero_raw_scale_stage(...)` for `ONLINE`
- skip it for the offline direct-row-update path, to match the base kernel

Files touched:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Build/codegen result:

- experiments rebuild stayed compile-flat on the hot live kernels:
  - `consumer_mode=-1`: `20/36`
  - `consumer_mode=5`: `16/20`
  - `consumer_mode=4`: `24/28`
  - `consumer_mode=3/2/0`: `16/20`

Warm fullgrid check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`, vs stored-`P` oracle:

- `streaming_live_localcta_prod_tcgen_mm2`
  - timings: `0.271104`, `0.260640`, `0.253152`
  - warm median: `0.260640 ms`
  - `mean_abs_diff = 9.1794e-5`
  - `max_abs_diff = 2.0142e-2`
  - `lse_max_abs_diff = 1.6070e-2`
- `streaming_live_localcta_prod_tcgen_auto`
  - timings: `0.275008`, `0.259488`, `0.254048`
  - warm median: `0.259488 ms`
  - `mean_abs_diff = 1.1844e-4`
  - `max_abs_diff = 4.3701e-2`
  - `lse_max_abs_diff = 1.6070e-2`
- `streaming_live_localcta_prod_tcgen_mm2_synced`
  - timings: `0.260544`, `0.258272`, `0.248864`
  - warm median: `0.258272 ms`
  - `mean_abs_diff = 1.0560e-4`
  - `max_abs_diff = 3.3691e-2`
  - `lse_max_abs_diff = 1.6070e-2`

Interpretation:

- removing the offline zero pass did not help codegen
- it regressed the live candidates into the `0.258-0.261 ms` band on this branch
- accuracy stayed in-family, so this is a pure speed loss

Conclusion:

- reverted and rebuilt back to the working branch

### 2026-04-18 follow-up: direct packed-scale helper shape A/Bs reverted

I tested two more bounded helper-shape A/Bs on top of the current float-source `block_amax` working branch in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

1. Replace the indexed `float packed_scale_word{0,1}[4]` arrays in
   `fp4pv_pack_scores_to_stage_and_scales(...)` with fixed `float4` words and explicit
   `q==0/1/2/3` assignments.
2. Keep the working helper logic, but store the converted BF16 pairs in
   `uint32_t scores_group_bf_u32[8]` instead of `bf16_2 scores_group_bf[8]` inside
   `fp4pv_quantize_scores_group_from_float2(...)`.

Build/codegen result for both:

- base `kernel_fp4pv` stayed flat at `16/20` spills and `1328 B` smem

Interpretation:

- neither helper-shape change produced any codegen signal at all on the kept branch
- that is not enough to justify carrying either into the experiments rebuild and timing path

Conclusion:

- both were reverted immediately
- branch restored and rebuilt back to the working state:
  - base `kernel_fp4pv = 16/20`
  - `1328 B` smem

### 2026-04-18 follow-up: partial unroll on float-source helper reverted

I tested one more bounded helper-level A/B on the current working branch in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- in `fp4pv_quantize_scores_group_from_float2(...)`
- change the float-source conversion/amax loop from `#pragma unroll` to `#pragma unroll 4`

Build/codegen result:

- spills stayed flat at `16/20`
- but base `kernel_fp4pv` stack frame exploded from `16 bytes` to `560 bytes`

Interpretation:

- partial unroll did not improve hot-kernel pressure
- it introduced a large stack-frame regression without any positive signal

Conclusion:

- reverted immediately
- branch restored and rebuilt back to the working state:
  - base `kernel_fp4pv = 16/20`
  - `1328 B` smem

### 2026-04-18 follow-up: direct float-pack primitive reverted

I tested a larger post-QK primitive swap on the current working branch in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- in `fp4pv_quantize_scores_group_from_float2(...)`
- stop converting the eight `float2` pairs to `bf16_2[8]` and calling
  `mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(...)`
- instead, after computing the same `coeff`, scale the source `float2` pairs directly and pack them
  with the in-file PTX helper:
  - `fp4pv_packed_float_to_e2m1(...)`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `16/20` spills and `1328 B` smem

Validation result:

- a narrow production-only runtime/accuracy check on this branch did not return cleanly enough on this host
- that is not enough signal to keep a real pack-primitive swap

Conclusion:

- reverted and rebuilt back to the working branch:
  - base `kernel_fp4pv = 16/20`
  - `1328 B` smem

### 2026-04-30 follow-up: B300 backward candidate 2048 safe state restored

I resumed the exact backward `candidate 2048` path after the hot-clustered dq-only
experiments.

Current safe source state:

- `b300_bwd_cute16_candidate.cuh`
  - `kUseHotClusteredDqOnly2048 = false`
  - `TK_FA4_USE_HOT_CLUSTERED_DQ=1` can now enable the hot-cluster dq-only probe at runtime
    without editing and rebuilding the source gate
- `b300_bwd_cute16_kernel_candidate.cuh`
  - clustered dq-only legacy path calls `tkfa4::bwd_hot::detail::hot_compute_dq_loop<true, C>(...)`
  - first-four exact overwrite is restored as `if (q_block_idx <= q_start_block + 3)`
- the temporary `hot_compute_dq_qmajor_loop(...)` helper was removed because it had only been
  compile-tested and was not runtime-validated

Validated exact baseline after restore:

- command:
  - `CUDA_VISIBLE_DEVICES=2 TK_FA4_SPLIT_TIMING=1 timeout 70s /workspace/codebases/fp4_matmul/.venv/bin/python /workspace/codebases/fp4_matmul/tk_fa4/direct_bwd_probe.py candidate 2048 0 cute 1 ref`
- warm split timings:
  - `preprocess=17.12 us, dkdv=380.77 us, dq=682.85 us, total=700.86 us`
  - `preprocess=20.38 us, dkdv=380.61 us, dq=676.64 us, total=705.44 us`
  - `preprocess=14.46 us, dkdv=382.85 us, dq=689.82 us, total=705.25 us`
- result:
  - `time_us = 795.072`
  - `dq_refdiff.max_absdiff = 2.4199486e-05`, `count = 0`
  - `dk_refdiff.max_absdiff = 3.0994415e-05`, `count = 0`
  - `dv_refdiff.max_absdiff = 0.0`, `count = 0`

Off-path hot-cluster measurement status:

- rebuilt once with `kUseHotClusteredDqOnly2048 = true` to re-measure the clustered hot path
- build succeeded; the clustered main reported `168` registers, `11` barriers, `288 B` stack,
  `568 B` spill stores, `588 B` spill loads, and `189524 B` smem
- runtime probes on `CUDA_VISIBLE_DEVICES=2` and `CUDA_VISIBLE_DEVICES=1` both failed before
  kernel launch at `torch.cuda.set_device(...)` with CUDA Error 304 / NVML init failure
- after adding the runtime env switch, the default exact path was revalidated from the same binary:
  - warm totals `695.74 us`, `698.98 us`, `707.68 us`
  - `dq_refdiff.max_absdiff = 2.4203211e-05`, `count = 0`
  - `dk_refdiff.max_absdiff = 3.0994415e-05`, `count = 0`
  - `dv_refdiff.max_absdiff = 0.0`, `count = 0`
- an env-enabled probe with `TK_FA4_USE_HOT_CLUSTERED_DQ=1` still failed at CUDA device init
  before kernel launch; a standalone `torch.cuda.set_device(0)` then reproduced the same Error 304,
  so this was treated as host/device-init instability rather than a kernel result
- no new performance or correctness conclusion should be drawn for the hot-cluster path from those
  failed runtime attempts

Practical next step:

- keep the safe gate off unless deliberately measuring the hot-cluster branch
- if runtime is available, re-run the hot-clustered probe before making further structural changes:
  - `CUDA_VISIBLE_DEVICES=2 TK_FA4_USE_HOT_CLUSTERED_DQ=1 TK_FA4_SPLIT_TIMING=1 TK_FA4_CLUSTERED_DQ_TIMING=1 timeout 70s /workspace/codebases/fp4_matmul/.venv/bin/python /workspace/codebases/fp4_matmul/tk_fa4/direct_bwd_probe.py candidate 2048 0 cute 1 ref`

### 2026-04-27 CuTe DSL D192 follow-up: setup path still valid, first Q-payload copy smoke reverted

Current validated state in:

- `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`

Still-good result:

- the setup-only CuTe DSL smoke path with all six descriptors
  - `Q`
  - `K`
  - `V`
  - `Q_scale`
  - `K_scale`
  - `V_scale`
- still compiles and launches cleanly on `cuda:1`

Revalidation:

- `run_mxfp4_fmha_d192_setup_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device='cuda:1')`
- returned:
  - `status: ok`
  - `problem_size: (1, 128, 128, 12, 12, 192, 128)`

What was attempted and reverted:

- added a gated `q_copy_smoke_kernel(...)` to do the first real `Q` payload TMA `cute.copy(...)`
- tried both:
  - full rounded FMHA grid
  - one-cluster-only launch

Observed behavior:

- source compiled
- the `Q` copy smoke did not return in a reasonable compile / launch window
- this was treated as not production-healthy for the current branch state

Conclusion:

- reverted the gated `Q` copy smoke path
- restored the file to the last known-good six-descriptor setup-only state
- next step should avoid dropping a pipeline-backed copy stage directly into `__call__`
- the safer seam is a smaller standalone CuTe copy micro-kernel or a reference-derived producer fragment before reattempting the first FMHA data-movement stage

### 2026-04-27 CuTe DSL D192 follow-up: standalone blockscaled FP4 GEMM smoke is live

Added:

- `tk_fa4/cute_dsl_mxfp4_qk_gemm_smoke.py`

Purpose:

- provide a real CuTe DSL FP4 compile+launch path outside the FMHA port
- use the in-tree `dense_blockscaled_gemm_persistent.py` reference as a standalone micro-run
- keep the FMHA D192 file on the validated six-descriptor setup-only branch

Validated on `cuda:1`:

- `run_mxfp4_qk_gemm_smoke(m=128, n=128, k=256, l=12, device='cuda:1', warmup_iterations=0, iterations=1, skip_ref_check=False)`
- returned:
  - `status: ok`
  - `exec_time_us = 8.1599997356534`
  - `exec_time_ms = 0.0081599997356534`

Interpretation:

- the CuTe DSL runtime and blockscaled FP4 MMA path are healthy for a real compute kernel
- the current blocker is specifically the FMHA-side producer/mainloop integration, not the CuTe environment or FP4 GEMM reference path

### 2026-04-27 CuTe DSL D192 follow-up: standalone FMHA-style Q/K load micro-kernel is live

Added:

- `tk_fa4/cute_dsl_mxfp4_qk_load_smoke.py`

Purpose:

- validate the FMHA-side padded `Q/K` payload staging path separately from the full D192 port
- keep the main D192 FMHA file on the last known-good six-descriptor setup-only state
- avoid another direct producer-path splice into `cute_dsl_mxfp4_forward_kernel_d192.py` before the load fragment itself is known-good

What it does:

- builds padded `Q/K` FMHA layouts for `D=192 -> 256`
- constructs blockscaled-FP4-flavored tiled MMA metadata
- builds `Q` and `K` payload TMA atoms
- runs a tiny producer/consumer smoke kernel with real `cute.copy(...)` TMA payload loads

Validated on `cuda:1`:

- `run_mxfp4_qk_load_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device='cuda:1')`
  - `status: ok`
  - `problem_size: (1, 128, 128, 12, 12, 192)`
- `run_mxfp4_qk_load_smoke(batch_size=1, seqlen_q=4096, seqlen_k=4096, heads_q=12, heads_k=12, device='cuda:1')`
  - `status: ok`
  - `problem_size: (1, 4096, 4096, 12, 12, 192)`

Interpretation:

- real FMHA-style `Q/K` payload staging is now validated outside the fused D192 kernel
- the next clean merge step is to port this standalone `Q/K` producer fragment back into
  `cute_dsl_mxfp4_forward_kernel_d192.py`, then extend the same standalone method to scale loads
  before reattempting the full QK mainloop

### 2026-04-29 CuTe DSL D192 follow-up: `K_scale` standalone load extension reverted

Attempted in:

- `tk_fa4/cute_dsl_mxfp4_qk_load_smoke.py`

Goal:

- extend the working standalone FMHA-style `Q/K` payload load micro-kernel
- add real `K_scale` TMA copy staging before touching the fused D192 FMHA file again

What was tried:

- first a mixed-input-FMHA-style `K_scale` partition path
- then a manual flattened `K_scale` global layout plus manual TMA view
- then a grouped-mode variant for `cute.nvgpu.cpasync.tma_partition(...)`

Observed blocker:

- all `K_scale` variants failed at `cpasync.tma_partition(...)` type formation
- the payload-only `Q/K` path continued to validate cleanly

Restored validated state:

- reverted `tk_fa4/cute_dsl_mxfp4_qk_load_smoke.py` back to payload-only
- revalidated on `cuda:1`:
  - `run_mxfp4_qk_load_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device='cuda:1')`
  - `status: ok`

Interpretation:

- FMHA-style `Q/K` payload staging is still a good keep
- `K_scale` needs its own smaller standalone TMA layout experiment instead of being bolted
  directly onto the working payload micro-kernel

### 2026-04-30 CuTe DSL D192 follow-up: added standalone `K_scale` load smoke, runtime blocked by CUDA state

Added:

- `tk_fa4/cute_dsl_mxfp4_scale_load_smoke.py`

Purpose:

- isolate `K_scale` TMA copy from the working payload-only `Q/K` load smoke
- use the D192 two-CTA geometry directly:
  - `qk_mma_tiler = (256, 128, 64)`
  - `cluster_shape_mn = (2, 1)`
  - `qk_scale_granularity = 64`
  - `qk_sf_vec_size = 32`
- mirror the mixed-input FMHA `K_scale` load pattern in a tiny standalone kernel

Current validation:

- static Python validation passed:
  - `python3 -m py_compile tk_fa4/cute_dsl_mxfp4_scale_load_smoke.py`
  - also recompiled the existing CuTe smoke files

Runtime status:

- attempted:
  - `run_mxfp4_k_scale_load_smoke(batch_size=1, seqlen_k=128, heads_q=12, heads_k=12, device='cuda:1')`
- runtime did not reach CuTe compile because PyTorch could not initialize CUDA:
  - `cudaGetDeviceCount` returned error `304`
  - `torch.cuda.is_available() == False`
- `nvidia-smi` still sees the GPUs, but `/dev/nvidia*` device nodes were not visible from this shell at the time of the run

Interpretation:

- the scale-only source is in place and syntax-valid
- it still needs real CuTe runtime validation once CUDA access is restored
- the working payload-only `Q/K` load smoke remains the last runtime-validated producer fragment

### 2026-04-30 CuTe DSL D192 follow-up: added standalone `Q_scale` load smoke, runtime still blocked

Added to:

- `tk_fa4/cute_dsl_mxfp4_scale_load_smoke.py`

Purpose:

- add the matching `Q_scale` standalone TMA-copy smoke beside the isolated `K_scale` smoke
- keep the fused D192 forward file on the validated six-descriptor setup-only path
- mirror the manual flattened `Q_scale` descriptor path from `cute_dsl_mxfp4_forward_kernel_d192.py`:
  - `q_scale_d_r = qk_cta_tiler[2] // qk_scale_granularity`
  - custom 1D swizzled TMA view over `q_scale_size_m * q_scale_d_r`
  - `CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)`
  - `blockscaled_utils.make_smem_layout_sfa(...)` for shared scale staging

Current validation:

- static Python validation passed under the CuTe venv:
  - `python3 -m py_compile tk_fa4/cute_dsl_mxfp4_scale_load_smoke.py tk_fa4/cute_dsl_mxfp4_qk_load_smoke.py tk_fa4/cute_dsl_mxfp4_qk_gemm_smoke.py tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`
- class construction passed under the CuTe venv:
  - `build_mxfp4_k_scale_load_smoke_kernel_class().__name__ == "Mxfp4KScaleLoadSmoke"`
  - `build_mxfp4_q_scale_load_smoke_kernel_class().__name__ == "Mxfp4QScaleLoadSmoke"`

Runtime status:

- no real CuTe compile/launch was possible in this shell because CUDA was unavailable:
  - PyTorch emitted `cudaGetDeviceCount` error `304`
  - PyTorch emitted `Can't initialize NVML`
  - `torch.cuda.is_available() == False`
  - `torch.cuda.device_count() == 0`
- later `nvidia-smi` saw four `NVIDIA GB200` GPUs, but direct PyTorch allocation still failed on
  `cuda:0`, `cuda:1`, `cuda:2`, and `cuda:3` with the same `cudaGetDeviceCount` error `304`
- `/dev/nvidiactl` was not present in this shell, which explains why CUDA runtime allocation fails
  even though `nvidia-smi` can enumerate GPUs

Interpretation:

- `K_scale` and `Q_scale` scale-only source now exists and is syntax/import valid
- the next useful action is a real runtime smoke on a shell with CUDA restored:
  - `run_mxfp4_q_scale_load_smoke(..., device="cuda:1")`
  - `run_mxfp4_k_scale_load_smoke(..., device="cuda:1")`
- do not merge scale copies back into `cute_dsl_mxfp4_forward_kernel_d192.py` until both standalone scale smokes compile and launch cleanly

### 2026-04-30 CuTe DSL D192 follow-up: CUDA restored, `K_scale` smoke launches, `Q_scale` still blocked

Runtime environment:

- `/dev/nvidiactl` and `/dev/nvidia0-3` were visible again
- `torch.cuda.is_available() == True`
- `torch.cuda.device_count() == 4`
- direct allocation status:
  - `cuda:0`: busy/unavailable
  - `cuda:1`: ok
  - `cuda:2`: ok
  - `cuda:3`: ok

Validated on `cuda:1`:

- setup-only six-descriptor D192 forward still compiles and launches:
  - `run_mxfp4_fmha_d192_setup_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`
- standalone FMHA-style `Q/K` payload load still compiles and launches:
  - `run_mxfp4_qk_load_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`
- standalone `K_scale` load now compiles and launches after switching its global scale layout
  to the mixed-input-FMHA flat layout:
  - `scale_k_layout = (s_k * d_r, ((h_r, h_k), b))`
  - `stride=(1, ((0, d_r * s_k), s_k * d_r * h_k))`
  - command:
    - `run_mxfp4_k_scale_load_smoke(batch_size=1, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`
- generic blockscaled GEMM still runs as a broad SFA/SFB sanity check:
  - `run_mxfp4_qk_gemm_smoke(m=128, n=128, k=256, l=12, device="cuda:1", warmup_iterations=0, iterations=1, skip_ref_check=False)`
  - returned `status: ok`

Rejected / still blocked:

- flat manual `Q_scale` TMA staging progressed past the first verifier error but then hung in
  Python/CuTe codegen while CPU-bound; killed after it did not return
- switching the flat `Q_scale` path from `PipelineTmaUmma` to `PipelineTmaAsync` did not fix the
  codegen hang
- trying the generic blockscaled SFA `make_tiled_tma_atom_A` path for `Q_scale` fails fast in
  descriptor construction:
  - `expected top-level shape equivalence between the SMEM layout and the CTA V-map`
  - using `qk_cta_tiler` and then `qk_mma_tiler` for `make_smem_layout_sfa(...)` both failed

Interpretation:

- the validated standalone producer ladder is now:
  - generic blockscaled FP4 GEMM/SFA path: ok
  - FMHA-style `Q/K` payload TMA: ok
  - mixed-input-style `K_scale` TMA: ok
  - full D192 setup-only six-descriptor path: ok
- the remaining immediate blocker before merging scale movement back into the fused D192 kernel is
  the custom `Q_scale` shared-memory view for the two-CTA D192 Q side
- next clean debug step:
  - keep `K_scale` as the validated scale template
  - isolate `Q_scale` view construction only, without another fused-kernel edit
  - decide whether the Q scale tensor should use the existing flat v5/prequantized layout or be
    repacked into the generic blockscaled SFA layout that the dense GEMM path already validates

### 2026-04-30 CuTe DSL D192 follow-up: standalone `Q_scale` now launches with per-CTA flat TMA

Updated:

- `tk_fa4/cute_dsl_mxfp4_scale_load_smoke.py`

Working `Q_scale` change:

- kept the flat v5/prequantized scale tensor layout:
  - `(s_q * d_r, ((h_r, h_k), b))`
  - `d_r = qk_cta_tiler[2] // qk_scale_granularity`
- staged Q scale through a flat swizzled TMA view:
  - `q_scale_tma_layout = make_composed_layout(make_swizzle(0, 4, 3), ..., (q_scale_size_m * d_r,))`
- switched the standalone Q-scale smoke to per-CTA scale movement:
  - `CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)`
  - no two-CTA multicast for this isolated baseline

Validated on `cuda:1`:

- small scale pair:
  - `run_mxfp4_q_scale_load_smoke(batch_size=1, seqlen_q=128, heads_q=12, heads_k=12, device="cuda:1")`
  - `run_mxfp4_k_scale_load_smoke(batch_size=1, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - both returned `status: ok`
- larger shape sanity:
  - `run_mxfp4_q_scale_load_smoke(batch_size=1, seqlen_q=4096, heads_q=12, heads_k=12, device="cuda:1")`
  - `run_mxfp4_k_scale_load_smoke(batch_size=1, seqlen_k=4096, heads_q=12, heads_k=12, device="cuda:1")`
  - both returned `status: ok`

Current interpretation:

- both Q-side and K-side scale TMA movement now have runnable standalone baselines
- `Q_scale` is not yet the final optimal two-CTA/multicast form; it is a correctness/bring-up
  baseline using one TMA copy per CTA
- the next clean integration step is a combined standalone producer smoke containing:
  - Q payload TMA
  - K payload TMA
  - flat per-CTA Q-scale TMA
  - mixed-input-style K-scale TMA
- only after that combined standalone producer smoke launches should the same pieces be moved into
  `cute_dsl_mxfp4_forward_kernel_d192.py`

### 2026-04-30 CuTe DSL D192 follow-up: combined v5-style QK producer smoke launches

Added:

- `tk_fa4/cute_dsl_mxfp4_qk_producer_smoke.py`

Working combined producer baseline:

- Q/K payload uses the v5-style one-CTA MXFP4 path:
  - `qk_mma_tiler = (128, 128, 64)`
  - `cluster_shape_mn = (1, 1)`
  - `tcgen05.CtaGroup.ONE`
- Q/K scales use the flat prequantized layout:
  - `(s * d_r, ((h_r, h_k), b))`
  - `d_r = qk_cta_tiler[2] // qk_scale_granularity`
  - flat swizzled TMA view over `q_scale_size_m * d_r`
  - per-CTA `CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)`

Validated on `cuda:1`:

- small combined producer smoke:
  - `run_mxfp4_qk_producer_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`
- larger shape sanity:
  - `run_mxfp4_qk_producer_smoke(batch_size=1, seqlen_q=4096, seqlen_k=4096, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`

Rejected / diagnostic:

- a direct D192 two-CTA payload-only smoke was attempted with:
  - `qk_mma_tiler = (256, 128, 64)`
  - `cluster_shape_mn = (2, 1)`
  - `tcgen05.CtaGroup.TWO`
- descriptor setup had already been validated in the setup-only forward smoke, but actual payload
  TMA copy codegen did not return cleanly:
  - combined Q/K payload plus Q/K scales timed out in CuTe codegen
  - D192 two-CTA Q/K payload-only timed out / was terminated
  - D192 two-CTA K-payload-only also stayed silent in codegen and was terminated
- forcing one-CTA with `qk_mma_tiler = (256, 128, 64)` fails immediately:
  - `MmaMXF4Op error: expects the M-mode to be 128, but got 256`
- diagnostic payload-only files from this pass were deleted; keep the durable evidence here rather
  than carrying broken smoke entrypoints

Current interpretation:

- the first runnable combined producer baseline is the v5-style one-CTA Q/K payload path plus
  flat per-CTA Q/K scale movement
- this matches the practical direction of keeping Q/K on v5 instead of chasing localCTA/two-CTA
  payload movement first
- next integration step:
  - port the working `cute_dsl_mxfp4_qk_producer_smoke.py` producer fragment into
    `cute_dsl_mxfp4_forward_kernel_d192.py`
  - keep it as producer/copy-only first
  - only after that launches, wire the copied Q/K payload and scales into the first QK MMA issue

### 2026-04-30 CuTe DSL D192 follow-up: fused forward now launches embedded QK producer copy

Updated:

- `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`

What changed:

- kept the public D192 forward launcher signature unchanged:
  - Q, K, V, O, Q-scale, K-scale, V-scale, problem shape, softmax/output scale, window args, stream
- added a copy-only QK producer path inside the forward stub:
  - `qk_producer_mma_tiler = (128, 128, 64)`
  - `qk_producer_cluster_shape_mn = (1, 1)`
  - Q/K payload use `tcgen05.CtaGroup.ONE`
  - Q/K scales use the flat prequantized layout and per-CTA `CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)`
- the old two-CTA D192 descriptor setup code remains below the new path as reference, but the
  runnable path now returns after launching `qk_producer_smoke_kernel(...)`
- result mode is now reported as:
  - `"qk_producer_copy"`

Validated on `cuda:1`:

- small fused forward copy-only smoke:
  - `run_mxfp4_fmha_d192_setup_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`, `mode: qk_producer_copy`
- larger shape sanity:
  - `run_mxfp4_fmha_d192_setup_smoke(batch_size=1, seqlen_q=4096, seqlen_k=4096, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`, `mode: qk_producer_copy`

Current interpretation:

- the validated standalone combined producer has now been folded into the fused D192 forward file
  without reintroducing the two-CTA payload codegen hang
- this is still copy-only; it validates producer movement inside the fused forward launcher but
  does not yet issue QK MMA
- next clean step:
  - in `qk_producer_smoke_kernel(...)`, add the minimal QK MMA consumer after the staged Q/K
    payload and scales are available
  - keep sequence tiling to one Q/K tile first
  - do not add PV or online-softmax/P quantization until the first QK MMA issue compiles and runs

### 2026-05-01 CuTe DSL D192 follow-up: standalone QK MXFP4 MMA issue launches

Added:

- `tk_fa4/cute_dsl_mxfp4_qk_mma_tile_smoke.py`

What changed:

- added a standalone one-tile QK MXFP4 MMA launch smoke that stages:
  - Q payload
  - K payload
  - SFA / Q scales
  - SFB / K scales
- uses the blockscaled GEMM-style SFA/SFB shared-memory and TMA layout path:
  - `blockscaled_utils.tile_atom_to_shape_SF(...)`
  - `blockscaled_utils.make_smem_layout_sfa(...)`
  - `blockscaled_utils.make_smem_layout_sfb(...)`
  - `cute.nvgpu.make_tiled_tma_atom_A/B(..., internal_type=cute.Int16)`
- uses a one-CTA QK instruction shape:
  - `qk_mma_tiler = (128, 128, 256)`
  - `cluster_shape_mn = (1, 1)`
  - `tcgen05.CtaGroup.ONE`
- issue-only kernel:
  - allocates SFA/SFB/accumulator TMEM
  - moves SFA/SFB SMEM to TMEM
  - calls `cute.gemm(...)`
  - does not store output or compare numerics yet

Validated on `cuda:1`:

- small QK MMA smoke:
  - `run_mxfp4_qk_mma_tile_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`
- larger shape sanity:
  - `run_mxfp4_qk_mma_tile_smoke(batch_size=1, seqlen_q=4096, seqlen_k=4096, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`

Important findings:

- the earlier `K=64` producer-copy tile is valid for moving one payload slice, but it is not the
  correct standalone MXFP4 QK MMA tile for `sf_vec_size=32`
- the working blockscaled QK MMA path uses the full padded D192 reduction tile:
  - real head dim `192`
  - padded QK dim `256`
  - MMA reduction tile `K=256`
- the standalone MMA smoke currently requires `heads_q == heads_k` and uses flat GEMM `L`
  mode; GQA/repeated-K head mapping should be reintroduced after the base QK MMA path is folded
  into the forward stub

Next clean step:

- fold the standalone QK MMA issue path into `qk_producer_smoke_kernel(...)`
- keep it issue-only first; do not add output store, PV, or online-softmax/P quantization until
  the fused forward QK MMA issue compiles and launches
- after the fused issue-only path launches, add either:
  - an accumulator store/debug path for numeric validation, or
  - the online-softmax reduction path if the accumulator shape is already proven

### 2026-05-01 CuTe DSL D192 follow-up: fused forward now launches QK MXFP4 MMA issue

Updated:

- `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`

What changed:

- replaced the fused copy-only QK producer smoke with the standalone-validated QK MMA issue path
- the fused smoke now builds flat non-GQA Q/K tensors for the QK issue rung and generic
  blockscaled SFA/SFB scale tensors:
  - `blockscaled_utils.tile_atom_to_shape_SF(...)`
  - `make_smem_layout_sfa/sfb(...)`
  - `make_tiled_tma_atom_A/B(..., internal_type=cute.Int16)`
- changed the fused QK producer/issue tile from the copy-only `K=64` slice to the working MXFP4
  issue tile:
  - `qk_producer_mma_tiler = (128, 128, 256)`
- added the issue-only TMEM path in `qk_producer_smoke_kernel(...)`:
  - allocate accumulator/SFA/SFB TMEM
  - copy SFA/SFB from shared memory to tensor memory
  - set `tcgen05.Field.SFA/SFB`
  - call `cute.gemm(...)`
  - release the accumulator pipeline without storing output
- the fused issue kernel now launches over the real QK issue grid:
  - `(ceil_div(seqlen_q, 128), ceil_div(seqlen_k, 128), heads_q * batch)`
- the launcher now returns:
  - `"mode": "qk_mma_issue"`
  - `"qk_issue_grid": (...)`
  - CUDA-event timing fields for the compiled issue-only kernel:
    - `exec_time_ms`
    - `exec_time_us`

Validated on `cuda:1`:

- small fused forward QK issue smoke:
  - `run_mxfp4_fmha_d192_setup_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`, `mode: qk_mma_issue`
  - returned `qk_issue_grid: (1, 1, 12)`
  - timed with `warmup_iterations=2`, `iterations=10`: about `61.98 us`
- larger shape sanity:
  - `run_mxfp4_fmha_d192_setup_smoke(batch_size=1, seqlen_q=4096, seqlen_k=4096, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`, `mode: qk_mma_issue`
  - returned `qk_issue_grid: (32, 32, 12)`
  - timed with `warmup_iterations=2`, `iterations=10`: about `151.80 us`

Current constraints:

- still issue-only:
  - no accumulator store
  - no numerical comparison
  - no online softmax
  - no PV
- temporarily requires `heads_q == heads_k`; GQA/K-head replication should be restored after
  numeric validation is available for the base fused QK issue path

Next clean step:

- add a debug accumulator store or minimal reduction/export path for QK numeric validation
- after QK numerics are checked, wire the accumulator into online softmax/P quantization
- keep PV blocked until QK issue plus softmax/P quantization has a measurable validated path

### 2026-05-01 CuTe DSL D192 follow-up: standalone QK accumulator debug store validates

Updated:

- `tk_fa4/cute_dsl_mxfp4_qk_mma_tile_smoke.py`

What changed:

- added optional `store_accumulator=True` mode to the standalone QK MMA smoke
- added a non-TMA debug epilogue:
  - TMEM accumulator load through `sm100_utils.get_tmem_load_op(...)`
  - register-to-global SIMT store through `cute.nvgpu.CopyUniversalOp()`
- expanded the standalone smoke CTA grid over the flat head/batch `L` mode:
  - `grid = (1, 1, heads_q * batch)`
- kept the issue-only mode available:
  - `store_accumulator=False`
- added raw deterministic validation modes:
  - `zero_inputs=True`
  - `constant_ones=True`
- raw constant-one encoding used for the validation:
  - FP4 E2M1 one packed as byte `0x22`
  - FP8 E8M0 scale one as byte `0x7f`

Validation on `cuda:1`:

- issue-only regression:
  - `run_mxfp4_qk_mma_tile_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1")`
  - returned `status: ok`, `store_accumulator: False`
- zero-input accumulator store:
  - `run_mxfp4_qk_mma_tile_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1", store_accumulator=True, zero_inputs=True)`
  - returned `scores_max_abs: 0.0`
  - returned `scores_nan_count: 0`
- constant-one accumulator store:
  - `run_mxfp4_qk_mma_tile_smoke(batch_size=1, seqlen_q=128, seqlen_k=128, heads_q=12, heads_k=12, device="cuda:1", store_accumulator=True, constant_ones=True)`
  - returned `scores_min: 256.0`, `scores_max: 256.0`
  - returned `expected_score: 256.0`
  - returned `scores_expected_max_abs_diff: 0.0`
  - returned `scores_nan_count: 0`

Important interpretation:

- the standalone QK MXFP4 MMA accumulator is now observable and numerically sane for simple
  deterministic inputs
- the expected constant-one score is `256`, not `192`, because the current QK issue path still
  multiplies across the padded D192-to-D256 lanes
- this debug store is standalone-only; the fused forward path remains issue-only

Next clean step:

- either add padded-lane masking/zero-fill so constant-one validation expects `192`
- or move directly to an online-softmax debug path that handles the padded lanes before PV

### 2026-04-26 CuTe DSL D192 follow-up: first real setup-only test run

I moved `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py` from symbolic setup only to a real
CuTe compile/launch smoke path.

What changed:

- fixed the `D=192 / V=128` output-layout assumptions to use `divby=128`, not `256`
- added real payload TMA construction in `__call__` for:
  - `Q`
  - `K`
  - `V`
- added a minimal `@cute.kernel` smoke kernel that only:
  - prefetches the payload TMA descriptors
  - launches with the real CTA/block/cluster geometry
  - returns
- added `run_mxfp4_fmha_d192_setup_smoke(...)`
  - allocates raw CUDA buffers
  - uses `cute.runtime.make_ptr(...)` for FP4 / FP8 / BF16 pointers
  - compiles and launches the new setup-only path

Important implementation notes:

- host-side `cute.make_layout(...)` is still not safe in this runtime for this path
  - the smoke helper uses analytic buffer sizing instead
- Torch DLPack export still does not support `float4`
  - the helper bypasses DLPack and uses raw `make_ptr(...)`
- `problem_shape` is dynamic at JIT time
  - the earlier `const_expr(...)` guards on `d_qk` / `d_v` had to be removed

Validated on `cuda:1` in the local CuTe venv:

- `problem_size = (1, 128, 128, 12, 12, 192, 128)` -> `status: ok`
- `problem_size = (1, 4096, 4096, 12, 12, 192, 128)` -> `status: ok`

Current blocker:

- scale TMA is not wired yet
- the first direct attempt to build `q_scale` TMA failed during compile with:
  - top-level shape equivalence mismatch between the scale SMEM layout and CTA V-map
- so the current real smoke path is payload-only:
  - payload TMA launch works
  - scale-TMA shape compatibility is the next kernel seam

Practical conclusion:

- we now have a genuine CuTe DSL compile/launch test run for the D192 MXFP4 forward port
- the next edit should stay in `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`
- focus next on:
  - `q_scale` / `k_scale` / `v_scale` TMA shape/layout compatibility
  - then move from setup-only smoke launch to the first real FP4 mainloop stage

### 2026-04-26 CuTe DSL D192 follow-up: payload + K/V scale smoke launch works

I continued the setup-only CuTe DSL bring-up in:

- `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`

What changed:

- kept the real payload-TMA smoke launch
- switched `k_scale` / `v_scale` off the generic blockscaled GEMM TMA path
- instead used the mixed-input FMHA scale helper path:
  - `prefill_utils.get_scale_smem_layout(...)`
  - `cute.nvgpu.cpasync.make_tiled_tma_atom(...)`
- left `q_scale` out of the live smoke launch for now

Key debug conclusion:

- the generic blockscaled `make_smem_layout_sfa/sfb + make_tiled_tma_atom_A/B` path is not directly compatible with the FMHA D192 setup for scale tensors
- the mixed-input helper path *is* compatible for:
  - `k_scale`
  - `v_scale`

Validated in the local CuTe venv on `cuda:1`:

- payload + `k_scale` + `v_scale` setup-only launch:
  - `problem_size = (1, 128, 128, 12, 12, 192, 128)` -> `status: ok`
  - `problem_size = (1, 4096, 4096, 12, 12, 192, 128)` -> `status: ok`

Current isolated blocker:

- `q_scale` TMA still fails shape-compatibility checks
- `q_scale` needs its own A-side scale view / TMA construction, not the generic blockscaled GEMM path

Practical state now:

- real CuTe compile/launch works for:
  - `Q`
  - `K`
  - `V`
  - `K_scale`
  - `V_scale`
- next seam is specifically:
  - `Q_scale`
- after that, the next meaningful step is the first nontrivial mainloop stage rather than more launch plumbing

### 2026-04-27 CuTe DSL D192 follow-up: Q-scale descriptor now works too

I closed the last descriptor-only setup seam in:

- `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`

What changed:

- `Q_scale` no longer uses the generic blockscaled GEMM `SFA` descriptor path
- instead it uses a manual flattened FMHA-specific layout:
  - shape `(s_q * q_scale_d_r, ((h_r, h_k), b))`
  - where `q_scale_d_r = qk_cta_tiler[2] // scale_granularity_qk`
- `Q_scale` TMA now uses:
  - `cute.nvgpu.cpasync.make_tiled_tma_atom(...)`
  - a small 1D swizzled SMEM TMA view
  - a per-CTA tiler of `q_scale_size_m * q_scale_d_r`

This is intentionally a setup-path solution, not yet the final Q-scale consumption layout for the real mainloop.

Validated in the local CuTe venv on `cuda:1`:

- full setup-only launch including:
  - `Q`
  - `K`
  - `V`
  - `Q_scale`
  - `K_scale`
  - `V_scale`

works for:

- `problem_size = (1, 128, 128, 12, 12, 192, 128)` -> `status: ok`
- `problem_size = (1, 4096, 4096, 12, 12, 192, 128)` -> `status: ok`
- `problem_size = (1, 8192, 8192, 12, 12, 192, 128)` -> `status: ok`

Current state:

- all six GMEM descriptor paths required by the MXFP4 D192 FMHA port now compile and launch in CuTe DSL
- the setup-only smoke path is no longer blocked on descriptor/layout construction

Next seam:

- stop working on descriptor-only launch plumbing
- start the first real data movement stage:
  - a minimal `cute.copy(...)` load stage for payload + scales
  - then the first nontrivial FP4 mainloop fragment

### 2026-04-24 CuTe DSL MXFP4 forward scaffolding

Context:

- pivoted away from further NVFP4 micro-tuning for the moment
- current experiments path is still `fp4_pv_experiments.py` `live_mxfp4_pv`
- next credible speed path is a fused CuTe DSL / CUTLASS-style forward kernel

Runtime setup result:

- created local venv:
  - `/workspace/codebases/fp4_matmul/.venv-cute`
- installed closest published runtime matching the repo state:
  - `nvidia-cutlass-dsl[cu13]==4.5.0.dev0`
- linked the venv to the system Torch install via:
  - `.venv-cute/lib/python3.12/site-packages/system_torch.pth`
- validated inside the venv:
  - `import cutlass`
  - `import cutlass.cute`
  - `import torch`

Reference import result:

- with:
  - `PYTHONPATH=/workspace/codebases/fp4_matmul/SageAttention/sageattention3_blackwell/csrc/cutlass/examples/python/CuTeDSL`
- both in-tree reference modules load cleanly:
  - `blackwell/fmha.py`
  - `blackwell/dense_blockscaled_gemm_persistent.py`

Source keep:

- updated:
  - `tk_fa4/cute_dsl_mxfp4_forward_scaffold.py`
- scaffold now:
  - knows the examples root
  - can probe CuTe DSL + Torch availability
  - can validate FMHA reference import
  - can validate blockscaled GEMM reference import
  - records the concrete reference entrypoints:
    - `BlackwellFusedMultiHeadAttentionForward`
    - `Sm100BlockScaledPersistentDenseGemmKernel`
  - records the actual MXFP4 port targets:
    - dense FMHA scheduler/mainloop from `fmha.py`
    - blockscaled FP4 MMA/layout machinery from `dense_blockscaled_gemm_persistent.py`

Current conclusion:

- CuTe DSL runtime is no longer the blocker
- the correct port shape is now explicit:
  - reuse FMHA persistent / online-softmax structure
  - swap dense QK/PV MMA to blockscaled FP4 MMA
  - thread prequantized `V` payload + scales through the mainloop
  - quantize `P` inside the softmax loop and feed PV directly

### 2026-04-24 CuTe DSL prototype bundle and the `D=192` constraint

Source keep:

- added:
  - `tk_fa4/cute_dsl_mxfp4_forward_prototype.py`

What it does:

- instantiates the in-tree CuTe DSL reference kernels as concrete Python objects:
  - `BlackwellFusedMultiHeadAttentionForward`
  - `Sm100BlockScaledPersistentDenseGemmKernel`
- exposes:
  - `prototype_gap_report(...)`
  - `build_reference_kernel_bundle(...)`
  - `summarize_reference_kernel_bundle(...)`
- validates the first real port constraint instead of leaving it implicit

Validated result:

- reference-friendly config works:
  - `qk_head_dim=128`, `v_head_dim=128`
  - FMHA and blockscaled GEMM references instantiate cleanly in the local CuTe DSL venv
- actual tk_fa4 target config is not directly instantiable from `blackwell/fmha.py`:
  - `qk_head_dim=192`, `v_head_dim=128`
  - fails the explicit compatibility gate because the reference FMHA only supports:
    - `32`, `64`, `128`

Important conclusion:

- the right large-D in-tree reference is not plain `blackwell/fmha.py`
- for the `D=192` port, the closest scheduler/mainloop reference is:
  - `blackwell/mixed_input_fmha/mixed_input_fmha_prefill_d256.py`
- so the next serious CuTe DSL step is:
  - keep blockscaled FP4 MMA/layout machinery from `dense_blockscaled_gemm_persistent.py`
  - use the large-D mixed-input FMHA path as the scheduler/mainloop reference
  - not the smaller dense FMHA path alone

### 2026-04-25 CuTe DSL FMHA family selection rule

Source keep:

- updated:
  - `tk_fa4/cute_dsl_mxfp4_forward_scaffold.py`
  - `tk_fa4/cute_dsl_mxfp4_forward_prototype.py`

What changed:

- scaffold runtime probe now also validates:
  - `blackwell/mixed_input_fmha/mixed_input_fmha_prefill_d256.py`
- prototype layer now has an explicit FMHA family chooser:
  - dense FMHA reference for `qk_head_dim in {32, 64, 128}`
  - mixed-input FMHA D256 reference for `qk_head_dim == 256`
  - structured gap for `qk_head_dim == 192`

Validated in the local CuTe DSL venv:

- `qk_head_dim=128`, `v_head_dim=128`
  - instantiates:
    - `BlackwellFusedMultiHeadAttentionForward`
    - `Sm100BlockScaledPersistentDenseGemmKernel`
- `qk_head_dim=256`, `v_head_dim=256`
  - instantiates:
    - `MixedInputFusedMultiHeadAttentionPrefillD256`
    - `Sm100BlockScaledPersistentDenseGemmKernel`
- `qk_head_dim=192`, `v_head_dim=128`
  - no direct reference instantiation:
    - dense reference support set: `32, 64, 128`
    - mixed-input reference support set: `256`

Current conclusion:

- `D=192` is now clearly an adaptation problem between two working reference families
- next real CuTe DSL kernel work should start from:
  - mixed-input D256 FMHA structure for large-D scheduling/mainloop ideas
  - blockscaled GEMM for FP4 MMA/layout
- not from trying to coerce `blackwell/fmha.py` directly into `D=192`

### 2026-04-26 first concrete `D=192` CuTe DSL port geometry

Source keep:

- added:
  - `tk_fa4/cute_dsl_mxfp4_forward_d192_port.py`

What it does:

- records the first explicit `D=192` MXFP4 CuTe DSL port candidate
- validates simple divisibility / tiling invariants locally
- points at the exact reference lineage and method-level patch seams

Current candidate geometry:

- `qk_head_dim = 192`
- `v_head_dim = 128`
- `seq_tile_n = 128`
- `qk_cta_tiler = (128, 128, 192)`
- `qk_mma_tiler = (256, 128, 64)`
- `pv_cta_tiler = (128, 128, 128)`
- `pv_mma_tiler = (256, 128, 128)`
- `qk_scale_granularity = 64`
- `v_scale_granularity = 128`
- `qk_sf_vec_size = 32`
- `pv_sf_vec_size = 32`
- `cluster_shape_mn = (2, 1)`

Validated local invariants:

- `192 % 64 == 0` for the proposed QK MMA K-step
- `128 == pv_mma_tiler[1]` for the proposed V head dim
- `seq_tile_n == pv_mma_tiler[2] == 128`
- scale granularities divide the relevant MMA dimensions
- `all_pass = true` in the local checker

Current conclusion:

- this is the first concrete `D=192` candidate worth coding against
- next kernel edit should use:
  - mixed-input D256 FMHA structure as the scheduler/mainloop base
  - blockscaled GEMM as the FP4 MMA/layout base
- the above `D=192` / `V=128` split geometry as the starting port target

### 2026-04-26 concrete `D=192` CuTe DSL kernel stub

Source keep:

- added:
  - `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`

What it is:

- a real CuTe DSL kernel class stub:
  - `Mxfp4FusedMultiHeadAttentionD192`
- built by subclassing the mixed-input D256 FMHA reference family at runtime
- intentionally not runnable yet
- holds the actual target field layout and stage geometry for the `D=192 / V=128` port

Validated in the local CuTe DSL venv:

- class builds cleanly:
  - `type(instance).__name__ == Mxfp4FusedMultiHeadAttentionD192`
- stage geometry initializes cleanly:
  - `q_stage = 4`
  - `kv_stage = 4`
  - `qk_cta_tiler = (128, 128, 256)`
  - `qk_mma_tiler = (256, 128, 64)`
  - `pv_cta_tiler = (128, 128, 128)`
  - `pv_mma_tiler = (256, 128, 128)`
  - `scale_granularity_qk = 64`
  - `scale_granularity_v = 128`

Important refinement from helper source:

- `make_blockscaled_trivial_tiled_mma(..., sf_vec_size=32)` uses blockscaled FP4 MMA with:
  - MMA op `K = 64`
- blockscaled GEMM then derives CTA reduction depth as:
  - `4 * 64 = 256`
- therefore the current best `D=192` port hypothesis is:
  - pad Q/K head dim to `256`
  - keep FP4 QK MMA step at `64`
  - zero or mask the padded `64` channels

Current conclusion:

- the active edit surface is no longer just a plan file
- next pass should modify:
  - `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`
- first real code merge should be:
  - replace inherited dense/int8 MMA setup with blockscaled FP4 MMA/layout setup
  - keep the mixed-input scheduler/mainloop skeleton intact initially

### 2026-04-26 partial CuTe DSL `__call__` setup path for `D=192`

Source keep:

- updated:
  - `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`

What is now real in the kernel file:

- the `Mxfp4FusedMultiHeadAttentionD192` class no longer has a no-op `__call__`
- `__call__` is now a partial CuTe DSL setup path that:
  - unpacks `(b, s_q, s_k, h_q, h_k, d_qk, d_v)`
  - builds padded FP4 Q/K tensor layouts for `D=192 -> 256`
  - builds FP4 V and BF16 output layouts for `V=128`
  - builds explicit scale tensor layouts for:
    - Q
    - K
    - V
  - constructs blockscaled QK MMA via:
    - `make_blockscaled_trivial_tiled_mma(...)`
  - constructs blockscaled PV MMA with operand-A source in TMEM
  - constructs staged SMEM layouts for:
    - Q payload
    - K payload
    - V payload
    - Q scales
    - K scales
    - V scales
  - constructs staged TMEM layout for P payload

Validated:

- file compiles with `python3 -m py_compile`
- class still instantiates in the local CuTe DSL venv
- stage counts are still consistent:
  - `q_stage = 4`
  - `kv_stage = 4`
  - `scale_q_stage = 4`
  - `scale_k_stage = 4`
  - `scale_v_stage = 4`

Still missing:

- TMA atom wiring for the new scale/payload mix
- shared storage / pipeline barrier rewrite for the FP4 path
- replacement of inherited dense/int8 transform and MMA mainloop code
- kernel launch path

Current conclusion:

- this is the first real CuTe DSL kernel file that contains MXFP4-specific layout/MMA setup
- the next edit should stay in `cute_dsl_mxfp4_forward_kernel_d192.py`
- the highest-value next seam is:
  - replace the inherited dense scale/dequant/TMA path with the FP4 payload+scale TMA path

### 2026-04-24 follow-up: live MXFP4 per-head CUDA clear removed

I traced the new `live_mxfp4_pv` slowdown after the chunked prequantized-QK keep and found that
the main cost was no longer the math kernels. The path was still paying the legacy
`_clear_cuda_after_combo_head(...)` policy for `seqlen >= 4096`, even though `live_mxfp4_pv`
no longer materializes a full logits tensor.

Change kept in:

- `tk_fa4/fp4_pv_experiments.py`

Change:

- in `_run_forward_precision_combo_only(...)`
- keep the legacy per-head CUDA clear for the old paths
- skip it for `p_mode == "live_mxfp4_pv"`

Validation before editing, using an in-process monkeypatch:

- `4096`
  - `clear_true_ms = 318.314765`
  - `clear_false_ms = 70.392925`
- `8192`
  - `clear_true_ms = 423.089141`
  - `clear_false_ms = 66.412138`

Real subprocess validation after the keep on `cuda:1`, `QK=localcta`, `PV=mxfp4_v3`,
`B=1`, `H=12`, `input_mode=random_live_fp4`, `warmup=1`, `iters=2`:

- `S=2048`
  - `timing_ms = 42.854581`
  - `mean_abs_diff_vs_bf16 = 0.00181941`
  - `lse_max_abs_diff_vs_bf16 = 0.0220826`
- `S=4096`
  - `timing_ms = 42.395090`
  - `mean_abs_diff_vs_bf16 = 0.00125518`
  - `lse_max_abs_diff_vs_bf16 = 0.0183486`
- `S=8192`
  - `timing_ms = 112.502478`
  - `mean_abs_diff_vs_bf16 = 8.7629e-4`
  - `lse_max_abs_diff_vs_bf16 = 0.0177755`
- `S=16384`
  - `timing_ms = 184.714197`
  - `mean_abs_diff_vs_bf16 = 6.0456e-4`
  - `lse_max_abs_diff_vs_bf16 = 0.0193442`

Interpretation:

- this is the first real MXFP4 keep that attacks the actual bottleneck instead of the inner math
- the current live MXFP4 path is now limited by the per-head quantize/QK/softmax/PV work itself,
  not by a benchmark-side cache flush

Related reverted probe on the same date:

- I tried a short-sequence streaming-QK bridge using `forward_streaming_live_qk_only(...)`
  plus a factored localCTA packer
- it was not viable: the `2048` validation alone sat at `99%` CPU for minutes
- reverted:
  - `_pack_live_fp4_qk_localcta_factored_from_bf16(...)`
  - `_mxfp4_pv_from_normalized_p_2d(...)`
  - `_live_mxfp4_pv_forward_from_streaming_qk_2d(...)`
  - the short-sequence branch in `_live_mxfp4_pv_forward_from_qkv_2d(...)`

Backend sweep on the kept branch:

- `S=2048`
  - `localcta = 42.854581 ms`
  - `v5 = 76.845566 ms`
  - `mxfp4_v3 = 73.860568 ms`
- `S=4096`
  - `localcta = 52.798566 ms`
  - `v5 = 47.476684 ms`
  - `mxfp4_v3 = 47.330602 ms`
- `S=8192`
  - `localcta = 112.301518 ms`
  - `v5 = 117.282565 ms`
  - `mxfp4_v3 = 118.584077 ms`
- `S=16384`
  - `localcta = 184.227379 ms`
  - `v5 = 186.592116 ms`
  - `mxfp4_v3 = 189.917798 ms`

Conclusion:

- keep `localcta` as the default QK backend for `live_mxfp4_pv`
- `v5` / `mxfp4_v3` only win narrowly at `4096`, and lose at `2048`, `8192`, and `16384`
- the next credible optimization is no longer another benchmark tweak; it is moving more of the
  live MXFP4 path into the extension side so we stop launching per-head quantize/QK/PV work from
  Python

### 2026-04-24 follow-up: stale `cuda:1` process invalidated one MXFP4 timing pass

After the clear-policy keep, I saw an apparent regression to roughly:

- `S=2048`: `~151 ms`
- `S=4096`: `~156 ms`
- `S=8192`: `~4.49 s`
- `S=16384`: `~4.55 s`

That turned out not to be a source regression. `cuda:1` was contaminated by a stale orphaned
benchmark shell:

- PID `3197931`
- command: `python3 -`
- cwd: `/workspace/codebases`
- GPU state: `99%` CPU, `~1360 MiB` on `cuda:1`

Killed that stale process and reran the full `live_mxfp4_pv` surface on a clean `cuda:1`.

Clean revalidation on `cuda:1`, `PV=mxfp4_v3`, `B=1`, `H=12`, `input_mode=random_live_fp4`,
`warmup=1`, `iters=2`:

- `QK=localcta`
  - `S=2048`: `49.944219 ms`
  - `S=4096`: `45.807143 ms`
  - `S=8192`: `133.137697 ms`
  - `S=16384`: `177.122911 ms`

All four points stayed in the same accuracy band as before:

- `S=2048`: `mean_abs_diff = 0.00181941`, `lse_max_abs_diff = 0.0220826`
- `S=4096`: `0.00125518`, `0.0183486`
- `S=8192`: `8.7629e-4`, `0.0177755`
- `S=16384`: `6.0456e-4`, `0.0193442`

I also retried the two batched-quantization ideas after the clean recheck:

- batched `Q/K` quantization hoist across heads
- flattened batch/head `V` MXFP4 quantization

Both were reverted. They were not viable speed directions:

- `Q/K` hoist regressed `2048/4096` and blew up `8192/16384` into multi-second runs
- flattened `V` quantization also regressed heavily and was reverted

Current clean backend sweep on `cuda:1` for `live_mxfp4_pv`:

- `S=2048`
  - `localcta = 59.001900 ms`
  - `v5 = 58.271562 ms`
  - `mxfp4_v3 = 44.611395 ms`
- `S=4096`
  - `localcta = 42.223767 ms`
  - `v5 = 53.712371 ms`
  - `mxfp4_v3 = 48.030227 ms`
- `S=8192`
  - `localcta = 127.349352 ms`
  - `v5 = 136.878654 ms`
  - `mxfp4_v3 = 145.585507 ms`
- `S=16384`
  - `localcta = 178.161362 ms`
  - `v5 = 229.570125 ms`
  - `mxfp4_v3 = 197.815400 ms`

Conclusion:

- keep `localcta` as the overall default QK backend for `live_mxfp4_pv`
- `mxfp4_v3` is only the fastest choice at `S=2048`, and it is less accurate there
- no new source keep from the batched-quantization attempts
- the main remaining opportunity is still extension-side MXFP4 live execution, not more Python-side
  quantization reshaping

### 2026-04-24 follow-up: standardize MXFP4 live QK on `v5` and add CuTe DSL scaffold

Per the current direction change, I removed `localcta` from the active MXFP4 live QK control path
and forced `v5` inside:

- `tk_fa4/fp4_pv_experiments.py`
  - `_live_mxfp4_pv_forward_from_qkv_2d(...)`

This is a policy keep, not a speed keep. It simplifies the MXFP4 branch and avoids
localCTA-specific behavior while we work toward a fused implementation.

Current `v5`-forced `live_mxfp4_pv` surface on clean `cuda:1`, `PV=mxfp4_v3`, `B=1`, `H=12`,
`input_mode=random_live_fp4`, `warmup=1`, `iters=2`:

- `S=2048`: `55.574774 ms`, `mean_abs_diff = 0.00183351`, `lse_max_abs_diff = 0.0243379`
- `S=4096`: `89.860838 ms`, `mean_abs_diff = 0.00126030`, `lse_max_abs_diff = 0.0274510`
- `S=8192`: `113.458685 ms`, `mean_abs_diff = 8.7904e-4`, `lse_max_abs_diff = 0.0314781`
- `S=16384`: `201.371514 ms`, `mean_abs_diff = 6.0562e-4`, `lse_max_abs_diff = 0.0258664`

I also added a separate CuTe DSL scaffold:

- `tk_fa4/cute_dsl_mxfp4_forward_scaffold.py`

What it does:

- points directly at the in-tree Blackwell CuTe DSL FMHA reference:
  - `SageAttention/sageattention3_blackwell/csrc/cutlass/examples/python/CuTeDSL/blackwell/fmha.py`
- probes whether the CuTe DSL runtime is importable
- records the exact MXFP4-specific extensions still needed for a real port:
  - explicit Q/K/V scale tensors
  - blockscaled FP4 MMA path
  - online `P` quantization inside softmax
  - prequantized MXFP4 `V` payload + scale handling
  - a `tk_fa4`-style launcher contract

Current runtime status from the scaffold:

- `cutlass_importable = false`
- reason: `ModuleNotFoundError("No module named 'cutlass'")`
- expected setup script:
  - `SageAttention/sageattention3_blackwell/csrc/cutlass/python/CuTeDSL/setup.sh`

Conclusion:

- the CuTe DSL path is feasible in this tree
- but it is not runnable in the current shell until the local CUTLASS CuTe DSL runtime is set up
- the right next step is a real CuTe DSL port against that Blackwell FMHA example, not more Python
  orchestration work

### 2026-04-23 follow-up: live MXFP4 now runs from chunked prequantized QK

I moved the experiments-only `live_mxfp4_pv` path in:

- `tk_fa4/fp4_pv_experiments.py`

off the materialized full `logits_2d` path.

Structural change:

- `Q` and `K` are now quantized once with the selected `qk_backend`
- the live loop slices prequantized `K` per tile
- each tile issues `_backend_gemm_quantized(q_quant, k_quant_tile, backend=qk_backend)`
- the resulting logits tile goes directly into the existing online softmax + MXFP4-`P` + prequantized-`V` path

This is still experiments-side Python orchestration, but it removes the worst architectural mismatch with the SageAttention3-style flow:

- no full `S x S` logits tensor for `live_mxfp4_pv`
- QK is now tile-driven inside the live path instead of “build all logits first, then consume them”

Support helpers added:

- `_contiguous_fp4_row_slice(...)`
- `_slice_quantized_rows_2d_forward_backend(...)`
- `_live_mxfp4_pv_forward_from_qkv_2d(...)`

Validation:

- `python3 -m py_compile tk_fa4/fp4_pv_experiments.py` passed

First spot check on `cuda:1`, `QK=localcta`, `PV=mxfp4_v3`, `B=1`, `H=12`, `random_live_fp4`, `warmup=1`, `iters=2`:

- `S=2048`: `45.090 ms`, `mean_abs_diff = 0.001819`
- `S=4096`: `344.865 ms`, `mean_abs_diff = 0.001255`
- `S=8192`: `434.562 ms`, `mean_abs_diff = 8.763e-4`
- `S=16384`: `544.804 ms`, `mean_abs_diff = 6.046e-4`

Interpretation:

- the path is structurally closer to the real fused design
- accuracy stayed unchanged versus the prior live MXFP4 branch
- runtime is still noisy on this host, so treat these as spot checks rather than final canonical numbers

Also tested and reverted:

- zero-copy full-range `K` slice reuse inside `_slice_quantized_rows_2d_forward_backend(...)`
- that micro-optimization was not stable enough to keep on the same noisy surface

---

MXFP4 PV pivot:

- Added an experiments-only forward `P_mode` in [fp4_pv_experiments.py](./fp4_pv_experiments.py):
  - `live_mxfp4_pv`
  - alias: `live_mxfp4_softmax`
- This is the first clean SageAttention3-style MXFP4 PV path in-tree:
  - prequantize full `V` once with `_mxfp4_quantize_v_for_gemm(...)`
  - quantize each softmax `P` tile on the fly with the real MXFP4 quantizer
  - issue PV with `_mxfp4_pv_from_p(...)` against the prequantized `V` tile slice
- The existing `live_sa3_baseline` path only quantized `P` online; it still requantized `V` through the generic backend wrapper.
- Base kernel is unchanged. This is a Python-orchestrated experiments path only.

Validation:

- `python3 -m py_compile fp4_pv_experiments.py` passed
- Runtime validation is still pending on a host with visible CUDA devices

Follow-up:

- CUDA visibility on this host was fine. The failing helper path was a subprocess-device bug:
  - `_run_json_subprocess(...)` was reading `device.index` directly
  - direct callers passing `device='cuda:1'` therefore produced a bad `CUDA_VISIBLE_DEVICES`
  - fixed by normalizing through `torch.device(...)` first

First runtime check of the new `live_mxfp4_pv` path on `cuda:1`, `S=2048`, `B=1`, `H=12`, `input_mode=random_live_fp4`, `QK=localcta`, `PV=mxfp4_v3`, `warmup=1`, `iters=2`:

- `live_mxfp4_pv`: `370.536 ms`, `mean_abs_diff = 0.010397`
- `live_sa3_baseline`: `685.774 ms`, `mean_abs_diff = 0.001869`
- `stored_p`: `59.025 ms`, `mean_abs_diff = 0.001958`

Interpretation:

- the new prequantized-`V` MXFP4 live path is real and running
- it is materially faster than the old `live_sa3_baseline` path
- but it is still far slower and less accurate than `stored_p / mxfp4_v3`

Second pass on `live_mxfp4_pv`:

- Root cause of the large accuracy gap: the first real MXFP4 live path was issuing `K=128` tile GEMMs directly.
- The established `mxfp4_v3` backend path always pads `K` to at least `256`, so the live path was outside the contract the old path had been tuned around.

Keeps now active in [fp4_pv_experiments.py](./fp4_pv_experiments.py):

- `live_mxfp4_pv` pads each live MXFP4 `P` tile to the chosen chunk width before quantization
- prebuilds a contiguous prequantized-`V` tile cache once per head
- uses dynamic chunking with cap `_MXFP4_LIVE_TILE_COLS = 4096`
  - `S=2048` => one `2048`-wide chunk
  - `S=4096` => one `4096`-wide chunk
  - `S=8192` => two `4096`-wide chunks

Measured on `cuda:1`, `QK=localcta`, `PV=mxfp4_v3`, `input_mode=random_live_fp4`, `B=1`, `H=12`, `warmup=1`, `iters=2`:

- `S=2048`
  - `live_mxfp4_pv`: `47.760 ms`, `mean_abs_diff = 0.001819`
  - `stored_p`: `41.696 ms`, `mean_abs_diff = 0.001958`
  - `live_sa3_baseline`: `685.774 ms`, `mean_abs_diff = 0.001869`
- `S=4096`
  - `live_mxfp4_pv`: `303.878 ms`, `mean_abs_diff = 0.001255`
  - `stored_p`: `319.566 ms`, `mean_abs_diff = 0.001389`
  - `live_sa3_baseline`: `1873.342 ms`, `mean_abs_diff = 0.001292`
- `S=8192`
  - `live_mxfp4_pv`: `439.631 ms`, `mean_abs_diff = 8.763e-4`
  - `stored_p`: `401.440 ms`, `mean_abs_diff = 0.001001`
  - `live_sa3_baseline`: `3607.759 ms`, `mean_abs_diff = 9.017e-4`

Rejected A/B:

- `_MXFP4_LIVE_TILE_COLS = 8192`
  - `S=4096`: regressed to `419.262 ms`
  - `S=8192`: regressed to `462.863 ms`
  - reverted

Interpretation:

- `live_mxfp4_pv` is now a real viable experiments path
- it is no longer numerically worse than the older SA3-style baseline
- it is much faster than `live_sa3_baseline`
- it is already slightly faster than `stored_p / mxfp4_v3` at `S=4096`
- it is still slightly slower than `stored_p / mxfp4_v3` at `S=2048` and `S=8192`

Long-sequence chunk-policy follow-up:

- Added an adaptive live MXFP4 chunk policy:
  - default cap stays `4096`
  - use `8192` only when `seqlen >= 16384`
- This only changes the `live_mxfp4_pv` experiments path.

Validation:

- `S=16384`, `warmup=1`, `iters=2`:
  - `live_mxfp4_pv`: `531.172 ms`, later reruns `583.567 ms`
  - prior `4096`-cap branch: `711.490 ms`
- higher-signal `S=16384`, `iters=3`:
  - `live_mxfp4_pv`: `565.286 ms`
  - `stored_p`: `487.091 ms`

Interpretation:

- adaptive `8192` chunking for `S>=16384` is a real keep
- it materially improves the 16K live path
- `stored_p / mxfp4_v3` still wins at 16K, but the gap is much smaller

### 2026-04-21 follow-up: lane-local TT load/store skip reverted

I tested one experiments-only consumer-rescale A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- starting from the validated per-lane selective consumer rescale keep
- when a warp entered the consumer rescale path:
  - lanes with `lane_needs_rescale == false` skipped the entire TT-output load/store loop
  - only lanes with `lane_needs_rescale == true` executed the TT-output load, multiply, and store

Build/codegen result:

- experiments rebuild passed
- live kernel codegen stayed flat:
  - `consumer_mode=-1`: `16/32`
  - `consumer_mode=0/2/3/4/5`: `16/20`

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- A/B branch:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.191360 ms`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.193248 ms`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.193216 ms`
- restored validated branch:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.190880 ms`, `mean_abs_diff = 8.0526e-5`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.194496 ms`, `mean_abs_diff = 7.0541e-5`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.192992 ms`, `mean_abs_diff = 1.4508e-4`

Interpretation:

- the A/B was not clearly better
- `mm2` regressed slightly, which is enough to reject it because `mm2` is the current default
- skipping the TT-output load/store entirely for non-rescaling lanes is not a keep

Conclusion:

- reverted and rebuilt back to the validated experiments branch
- current experiments keeps remain:
  - producer fast-rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4`: `10.f`
    - `STATIC_CONSUMER_MODE == 0 || 2 || 5`: `8.f`
  - consumer rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4 || 5`: `0.999f`
  - per-lane selective consumer rescale:
    - `applied_correction = lane_needs_rescale ? correction : 1.0f`
    - all lanes still participate in TT-output load/store once the warp enters rescale
    - only lanes below threshold execute the multiply loop

### 2026-04-21 follow-up: `mm2` producer threshold `10.5f` reverted

I tested one experiments-only producer fast-rescale A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the validated branch intact except for the `mm2` producer fast-rescale threshold
- `STATIC_CONSUMER_MODE == 3`:
  - `softmax_rescale_factor_fast(..., 10.f)` -> `softmax_rescale_factor_fast(..., 10.5f)`
- `STATIC_CONSUMER_MODE == 4` stayed at `10.f`

Build/codegen result:

- experiments rebuild passed
- live kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- A/B branch:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.191264 ms`, `mean_abs_diff = 5.7712e-5`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.195232 ms`, `mean_abs_diff = 7.7851e-5`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.193152 ms`, `mean_abs_diff = 1.6215e-4`
- validated branch before this probe:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.190880 ms`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.194496 ms`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.192992 ms`

Interpretation:

- `mm2` regressed slightly
- `auto` and `mm2_synced` also did not improve enough to justify a branch change
- `10.5f` is not a keep for the `mm2` producer fast-rescale path

Conclusion:

- reverted and rebuilt back to the validated experiments branch

### 2026-04-21 follow-up: consumer `corr_2` live-range cut kept

I tested one experiments-only consumer-rescale micro-edit in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the validated TT-output load/store pattern unchanged once a warp enters consumer rescale
- but stop materializing `applied_correction` / `corr_2` for lanes that do not rescale
- `corr_2 = {correction, correction}` is now created only inside:
  - `if (lane_needs_rescale) { ... }`

Build/codegen result:

- experiments rebuild passed
- live kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- pass 1:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.190784 ms`, `mean_abs_diff = 8.3678e-5`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.193056 ms`, `mean_abs_diff = 9.2062e-5`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.192640 ms`, `mean_abs_diff = 1.4250e-4`
- confirmation:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.191104 ms`, `mean_abs_diff = 8.4098e-5`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.193408 ms`, `mean_abs_diff = 8.4003e-5`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.193408 ms`, `mean_abs_diff = 1.5401e-4`

Interpretation:

- `mm2` stayed effectively flat relative to the validated branch
- `auto` improved modestly
- this is a safe keep because it reduces consumer-rescale live range without hurting the default path

Conclusion:

- kept on the experiments branch
- current recommendation remains:
  - default: `streaming_live_localcta_prod_tcgen_mm2`
  - fallback: `streaming_live_localcta_prod_tcgen_auto`

### 2026-04-21 follow-up: offline `row_sum += tile_sum_scalar` fast-path kept

I tested one experiments-only producer-side fast-path in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- in the offline producer path, after `wait(rescale_finished[0], ...)`
- replace:
  - `row_sum = row_sum * acc_scale + tile_sum_scalar`
- with:
  - `row_sum += tile_sum_scalar` when `acc_scale == 1.0f`
  - keep the original multiply-add only when `acc_scale != 1.0f`

Build/codegen result:

- experiments rebuild passed
- live kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- pass 1:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.190624 ms`, `mean_abs_diff = 8.9338e-5`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.193280 ms`, `mean_abs_diff = 7.2814e-5`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.192704 ms`, `mean_abs_diff = 1.3996e-4`
- confirmation:
  - `streaming_live_localcta_prod_tcgen_mm2`: `0.190560 ms`, `mean_abs_diff = 8.6002e-5`
  - `streaming_live_localcta_prod_tcgen_auto`: `0.193376 ms`, `mean_abs_diff = 8.3410e-5`
  - `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.192928 ms`, `mean_abs_diff = 1.3688e-4`

Interpretation:

- `mm2` improved modestly but repeatably
- `auto` and `mm2_synced` stayed in-family
- this is a safe keep because it only removes a multiply in the dominant `acc_scale == 1.0f` case

Conclusion:

- kept on the experiments branch

### 2026-04-21 follow-up: no-rescale `warpgroup::sync` skip reverted

I tested one experiments-only synchronization A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- in the consumer rescale loop
- move `warpgroup::sync(warpgroup::groupid() + 1)` inside `if (needs_rescale)`
- so warps with no TT-output rescale would skip that sync before `tt_output_reusable` / `rescale_finished`

Build/codegen result:

- experiments rebuild passed
- live kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `streaming_live_localcta_prod_tcgen_mm2`: `0.191232 ms`
- `streaming_live_localcta_prod_tcgen_auto`: `0.193568 ms`
- `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.192672 ms`

Interpretation:

- this was slower than the validated branch with the `row_sum` fast-path keep
- skipping the no-op sync is not a keep

Conclusion:

- reverted and rebuilt back to the validated experiments branch

### 2026-04-21 follow-up: hoisted `corr_2` reverted

I tested one experiments-only consumer-rescale A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the validated lane-local selective rescale logic
- but hoist `float2 corr_2 = {correction, correction}` out of the inner `col` loop for rescaling lanes
- non-rescaling lanes still did not materialize `corr_2`

Build/codegen result:

- experiments rebuild passed
- live kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `streaming_live_localcta_prod_tcgen_mm2`: `0.247456 ms`
- `streaming_live_localcta_prod_tcgen_auto`: `0.244736 ms`
- `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.241280 ms`

Interpretation:

- this was a hard runtime regression
- hoisting `corr_2` widened pressure in a way not reflected by the unchanged spill counts
- the current validated shape, with `corr_2` materialized inside the inner loop only for rescaling lanes, remains the right form

Conclusion:

- reverted and rebuilt back to the validated experiments branch

### 2026-04-22 follow-up: online `acc_scale == 1.0f` fast-path reverted

I tested one experiments-only online producer fast-path in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the validated offline `row_sum += tile_sum_scalar` fast-path
- additionally, in the `ONLINE` branch:
  - if `acc_scale == 1.0f`, use:
    - `row_sum += tile_sum_scalar`
    - `prev_contrib = row_sum_old`
  - otherwise keep the original multiply-add and `prev_contrib = row_sum_old * acc_scale`

Build/codegen result:

- experiments rebuild passed
- live kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `streaming_live_localcta_prod_tcgen_mm2`: `0.234976 ms`
- `streaming_live_localcta_prod_tcgen_auto`: `0.232928 ms`
- `streaming_live_localcta_prod_tcgen_mm2_synced`: `0.234720 ms`

Interpretation:

- this was a hard regression on all three live candidates
- the online correction path does not tolerate the same shortcut that helped offline

Conclusion:

- reverted and rebuilt back to the validated experiments branch

### 2026-04-20 follow-up: live `mm2` producer fast-rescale `11.f` reverted

I tested one bounded experiments-only A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- raise the offline producer fast-rescale threshold only for `STATIC_CONSUMER_MODE == 3`
  (`streaming_live_localcta_prod_tcgen_mm2`) from `10.f` to `11.f`
- leave `STATIC_CONSUMER_MODE == 4` at `10.f`
- leave the consumer-side per-lane selective rescale keep unchanged

Build/codegen result:

- experiments rebuild passed
- live-kernel codegen stayed flat:
  - `consumer_mode = 5/4/3/2/0`: `16/20`
  - `consumer_mode = 1`: `0/0`
  - `consumer_mode = -1`: `16/32`

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- first pass:
  - `mm2`: `0.233760 ms`, `mean_abs_diff = 6.9520e-5`
  - `auto`: `0.234560 ms`, `mean_abs_diff = 6.4650e-5`
  - `mm2_synced`: `0.232416 ms`, `mean_abs_diff = 1.4921e-4`
- confirmation pass:
  - `mm2`: `0.230944 ms`, `mean_abs_diff = 8.9809e-5`
  - `auto`: `0.229600 ms`, `mean_abs_diff = 8.7817e-5`
  - `mm2_synced`: `0.229408 ms`, `mean_abs_diff = 1.5353e-4`

Interpretation:

- the first pass looked mildly positive for `mm2`
- the confirmation pass did not clearly preserve that advantage
- this is not strong enough to keep a mode-specific threshold split

Conclusion:

- reverted and rebuilt back to the validated experiments branch:
  - producer fast-rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4`: `10.f`
    - `STATIC_CONSUMER_MODE == 0 || 2 || 5`: `8.f`
  - consumer rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4 || 5`: `0.999f`
  - consumer rescale applies per lane:
    - `applied_correction = lane_needs_rescale ? correction : 1.0f`

Current live recommendation on this branch:

- default: `streaming_live_localcta_prod_tcgen_mm2`
- accuracy-leaning fallback: `streaming_live_localcta_prod_tcgen_auto`

### 2026-04-20 follow-up: live `auto=11.f` and `mm2_synced=9.f` producer thresholds reverted

I tested two more experiments-only producer fast-rescale threshold A/Bs in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Changes:

- `STATIC_CONSUMER_MODE == 4` (`auto`): `10.f -> 11.f`
- `STATIC_CONSUMER_MODE == 5` (`mm2_synced`): `8.f -> 9.f`

Build/codegen result:

- experiments rebuilds passed
- live-kernel codegen stayed flat on both probes

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `auto=11.f` branch:
  - `mm2`: `0.260608 ms`
  - `auto`: `0.259296 ms`
  - `mm2_synced`: `0.261024 ms`
- `mm2_synced=9.f` branch:
  - `mm2`: `0.239456 ms`
  - `auto`: `0.236992 ms`
  - `mm2_synced`: `0.238016 ms`

Interpretation:

- `auto=11.f` was a clear regression
- `mm2_synced=9.f` did not produce a clean synced-only win and also disturbed the ranking

Conclusion:

- reverted both and rebuilt back to the validated experiments branch:
  - producer fast-rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4`: `10.f`
    - `STATIC_CONSUMER_MODE == 0 || 2 || 5`: `8.f`
  - consumer rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4 || 5`: `0.999f`
  - per-lane selective consumer rescale keep remains active

### 2026-04-21 follow-up: warp-uniform consumer rescale split reverted

I tested one more experiments-only consumer-rescale A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the current per-lane selective consumer rescale keep
- add a warp-uniform split:
  - if `__all_sync(..., lane_needs_rescale)` is true, run the multiply loop unconditionally
  - else keep the current lane-local multiply skip

Build/codegen result:

- experiments rebuild passed
- live-kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- first pass:
  - `mm2`: `0.191584 ms`
  - `auto`: `0.195136 ms`
  - `mm2_synced`: `0.193152 ms`
- confirmation pass:
  - `mm2`: `0.191872 ms`
  - `auto`: `0.195808 ms`
  - `mm2_synced`: `0.193152 ms`

Interpretation:

- the branch was effectively flat to slightly worse than the existing keep
- not enough to justify more control flow in the hot consumer loop

Conclusion:

- reverted and rebuilt back to the current validated experiments branch

### 2026-04-21 follow-up: relaxed `auto` non-first mm2 guard reverted

I retested one older experiments-only A/B on the current branch in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- for `STATIC_CONSUMER_MODE == 4` (`auto`) only, relax the producer-side non-first `mm2` guard
  from:
  - `acc_scale < 1.0f`
  to:
  - `acc_scale < 0.999f`

Reason for revisit:

- the earlier rejection was before the newer consumer-side keeps landed
- with the current live branch, it was worth rechecking whether `auto` could benefit from more
  permissive non-first `mm2` reuse

Build/codegen result:

- experiments rebuild passed
- live-kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- first pass:
  - `mm2`: `0.191424 ms`, `mean_abs_diff = 6.8628e-5`
  - `auto`: `0.195168 ms`, `mean_abs_diff = 6.0395e-5`
  - `mm2_synced`: `0.193312 ms`, `mean_abs_diff = 1.4889e-4`
- confirmation pass:
  - `mm2`: `0.191200 ms`, `mean_abs_diff = 7.2535e-5`
  - `auto`: `0.195296 ms`, `mean_abs_diff = 7.4515e-5`
  - `mm2_synced`: `0.193184 ms`, `mean_abs_diff = 1.4815e-4`

Interpretation:

- `mm2` stayed effectively unchanged
- `auto` did not improve enough to justify the semantic change
- the first-pass accuracy edge for `auto` did not hold on confirmation

Conclusion:

- reverted and rebuilt back to the validated experiments branch

### 2026-04-21 follow-up: `auto` first-issue `mma2` reverted

I tested one more experiments-only `auto` A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- for `STATIC_CONSUMER_MODE == 4` (`auto`) only
- switch the first PV issue from:
  - `mm2_ABt(...)`
  to:
  - `mma2_ABt(...)`
- non-first behavior was left unchanged

Build/codegen result:

- experiments rebuild passed
- live-kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- first pass:
  - `mm2`: `0.191584 ms`, `mean_abs_diff = 7.1449e-5`
  - `auto`: `0.194976 ms`, `mean_abs_diff = 6.8963e-5`
  - `mm2_synced`: `0.192768 ms`, `mean_abs_diff = 1.4175e-4`
- confirmation pass:
  - `mm2`: `0.190656 ms`, `mean_abs_diff = 8.5441e-5`
  - `auto`: `0.195072 ms`, `mean_abs_diff = 6.6972e-5`
  - `mm2_synced`: `0.192960 ms`, `mean_abs_diff = 1.3776e-4`

Interpretation:

- `auto` did not improve enough to justify the semantic change
- any small `mm2` movement was within the normal noise band

Conclusion:

- reverted and rebuilt back to the validated experiments branch

### 2026-04-20 follow-up: base per-lane consumer rescale reverted, experiments lane-local multiply kept

I tested one base-kernel A/B in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- port the experiments-style per-lane selective consumer rescale to the two older base-kernel
  consumer-rescale loops
- instead of multiplying every lane by raw `correction` once the warp enters rescale, use:
  - `lane_needs_rescale = correction < 1.0f`
  - `applied_correction = lane_needs_rescale ? correction : 1.0f`

Build/codegen result:

- base rebuild passed
- codegen stayed flat:
  - `kernel_fp4pv = 16/20`
  - `1328 B` smem

Validation result:

- the narrow production-only matrix check for `qk_pv_nvfp4_production_fullgrid` did not return a
  usable row cleanly on this branch
- that matches the failure mode of earlier unstable base A/Bs

Conclusion:

- reverted and rebuilt base back to the clean production branch

I then tested one experiments-only structural A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the per-lane selective consumer rescale threshold logic
- but only lanes with `lane_needs_rescale` perform the inner `o_reg[ii] = __fmul2_rn(...)`
  multiply
- lanes whose `applied_correction` is `1.0f` now skip that multiply loop entirely

Build/codegen result:

- experiments rebuild passed
- live-kernel codegen stayed flat

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- first pass:
  - `mm2`: `0.191488 ms`, `mean_abs_diff = 8.0063e-5`
  - `auto`: `0.195456 ms`, `mean_abs_diff = 7.8678e-5`
  - `mm2_synced`: `0.193664 ms`, `mean_abs_diff = 1.4448e-4`
- confirmation pass:
  - `mm2`: `0.190976 ms`, `mean_abs_diff = 9.0231e-5`
  - `auto`: `0.194720 ms`, `mean_abs_diff = 6.8204e-5`
  - `mm2_synced`: `0.193056 ms`, `mean_abs_diff = 1.3419e-4`

Interpretation:

- this is a real live-side win
- `mm2` remains clearly best on speed/accuracy balance
- `auto` stays slower than `mm2`, despite slightly lower mean error

Current validated experiments branch:

- producer fast-rescale threshold:
  - `STATIC_CONSUMER_MODE == 3 || 4`: `10.f`
  - `STATIC_CONSUMER_MODE == 0 || 2 || 5`: `8.f`
- consumer rescale threshold:
  - `STATIC_CONSUMER_MODE == 3 || 4 || 5`: `0.999f`
- per-lane selective consumer rescale keep remains active
- lanes with `applied_correction == 1.0f` now skip the inner multiply loop

Current live recommendation on this branch:

- default: `streaming_live_localcta_prod_tcgen_mm2`
- fallback: `streaming_live_localcta_prod_tcgen_auto`

### 2026-04-20 follow-up: `auto=9.f` producer threshold and `auto=0.9985f` consumer threshold reverted

I tested two more experiments-only `auto`-focused A/Bs in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Changes:

- producer fast-rescale threshold for `STATIC_CONSUMER_MODE == 4` (`auto`):
  - `10.f -> 9.f`
- consumer rescale threshold for `STATIC_CONSUMER_MODE == 4` (`auto`):
  - `0.999f -> 0.9985f`

Build/codegen result:

- experiments rebuilds passed
- live-kernel codegen stayed flat on both probes

Warmed canonical live validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `auto=9.f` branch:
  - `mm2`: `0.236608 ms`
  - `auto`: `0.240288 ms`
  - `mm2_synced`: `0.236800 ms`
- `auto=0.9985f` branch:
  - `mm2`: `0.246496 ms`
  - `auto`: `0.245632 ms`
  - `mm2_synced`: `0.238624 ms`
  - `auto` accuracy worsened to `mean_abs_diff = 9.3975e-5`

Interpretation:

- `auto=9.f` was slower than the validated branch
- `auto=0.9985f` did not improve the branch as a whole and degraded `auto` accuracy

Conclusion:

- reverted both and rebuilt back to the validated experiments branch:
  - producer fast-rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4`: `10.f`
    - `STATIC_CONSUMER_MODE == 0 || 2 || 5`: `8.f`
  - consumer rescale threshold:
    - `STATIC_CONSUMER_MODE == 3 || 4 || 5`: `0.999f`
  - per-lane selective consumer rescale keep remains active

### 2026-04-19 follow-up: higher offline fast-rescale threshold kept for `mm2` and `auto`

I tested one bounded experiments-only A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the current offline fast-rescale helper:
  - `softmax_rescale_factor_fast(...)`
- but raise the skip threshold only for:
  - `STATIC_CONSUMER_MODE == 3`
  - `STATIC_CONSUMER_MODE == 4`
- concretely:
  - `mm2` and `auto` now call `softmax_rescale_factor_fast(..., 10.f)`
  - `consumer_mode == 0, 2, 5` stay on the existing `8.f` threshold
  - `consumer_mode == 1` still stays on the exact helper path

Build/codegen result:

- experiments rebuild passed cleanly
- live kernel codegen stayed flat on the kept branch:
  - `consumer_mode = 5/4/3/2/0`: `16/20`
  - `consumer_mode = 1`: `0/0`
  - `consumer_mode = -1`: `16/32`
- base kernel is unchanged and still at:
  - `kernel_fp4pv = 16/20`
  - `1328 B` smem

Warmed fullgrid live recheck on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `streaming_live_localcta_prod_tcgen_mm2`:
  - `0.236736 ms`
  - `mean_abs_diff = 8.3527e-5`
  - `max_abs_diff = 5.0537e-2`
- `streaming_live_localcta_prod_tcgen_auto`:
  - `0.237056 ms`
  - `mean_abs_diff = 6.7633e-5`
  - `max_abs_diff = 4.2480e-2`
- `streaming_live_localcta_prod_tcgen_mm2_synced`:
  - `0.236960 ms`
  - `mean_abs_diff = 1.4359e-4`
  - `max_abs_diff = 4.4678e-2`

Short confirmation rerun:

- `mm2` and `auto` stayed in the same narrow `~0.238-0.243 ms` band
- neither variant separated enough to justify another default flip on this host

Interpretation:

- this is a small but real experiments-only speed keep
- the gain is concentrated on the two live candidates that matter
- error stayed in-family
- the result is not strong enough to change the live recommendation again

Conclusion:

- kept
- current live recommendation remains:
  - default: `streaming_live_localcta_prod_tcgen_auto`
  - near-tie fallback: `streaming_live_localcta_prod_tcgen_mm2`

### 2026-04-19 follow-up: `auto=12.f` and `mm2_synced=10.f` threshold probes reverted

I tested two more bounded experiments-only threshold A/Bs in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change 1:

- raise the offline fast-rescale threshold only for:
  - `STATIC_CONSUMER_MODE == 4`
- from `10.f` to `12.f`

Validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `mm2`: `0.239776 ms`, `mean_abs_diff = 1.0739e-4`
- `auto`: `0.238496 ms`, `mean_abs_diff = 6.4504e-5`

Interpretation:

- `auto` did not improve versus the working branch
- `mm2` got worse

Change 2:

- revert `auto` back to `10.f`
- then widen the `10.f` threshold set to:
  - `STATIC_CONSUMER_MODE == 3 || 4 || 5`

Validation on the same surface:

- `mm2`: `0.252512 ms`, `mean_abs_diff = 7.4792e-5`
- `auto`: `0.251808 ms`, `mean_abs_diff = 6.9241e-5`
- `mm2_synced`: `0.248544 ms`, `mean_abs_diff = 1.3082e-4`

Interpretation:

- all three live variants regressed materially
- there was no accuracy benefit large enough to justify keeping it

Conclusion:

- reverted both probes
- restored and rebuilt the working experiments branch:
  - `STATIC_CONSUMER_MODE == 3 || 4` use `10.f`
  - `STATIC_CONSUMER_MODE == 0 || 2 || 5` use `8.f`

### 2026-04-19 follow-up: offline consumer rescale skip threshold kept for `mm2` and `auto`

I tested one new experiments-only A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the current producer-side offline fast-rescale settings unchanged
- in the offline consumer rescale loop for the shared live path:
  - change `needs_rescale` from `correction < 1.0f`
  - to `correction < 0.999f`
- apply that only for:
  - `STATIC_CONSUMER_MODE == 3`
  - `STATIC_CONSUMER_MODE == 4`
- leave every other mode unchanged

Build/codegen result:

- experiments rebuild passed cleanly
- live kernel codegen stayed flat:
  - `consumer_mode = 5/4/3/2/0`: `16/20`
  - `consumer_mode = 1`: `0/0`
  - `consumer_mode = -1`: `16/32`
- base kernel is unchanged and still at:
  - `kernel_fp4pv = 16/20`
  - `1328 B` smem

Warmed fullgrid live check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- short run:
  - `mm2`: `0.229856 ms`, `mean_abs_diff = 8.5831e-5`, `max_abs_diff = 4.0527e-2`
  - `auto`: `0.230784 ms`, `mean_abs_diff = 9.0057e-5`, `max_abs_diff = 4.2480e-2`
- confirmation run:
  - `mm2`: `0.228032 ms`, `mean_abs_diff = 1.1006e-4`, `max_abs_diff = 4.4922e-2`
  - `auto`: `0.230272 ms`, `mean_abs_diff = 7.5424e-5`, `max_abs_diff = 1.9897e-2`
  - `mm2_synced`: `0.230400 ms`, `mean_abs_diff = 1.6198e-4`, `max_abs_diff = 4.4678e-2`

Interpretation:

- this is a real live-path speed keep
- the gain comes from skipping near-no-op TT-output rescale work in the consumer loop
- `mm2` is now the clear fastest live candidate again on the warmed canonical surface
- `auto` keeps the better error profile, but the speed gap is now large enough to flip the default back

Conclusion:

- kept
- current live recommendation is now:
  - default: `streaming_live_localcta_prod_tcgen_mm2`
  - accuracy-leaning near-tie fallback: `streaming_live_localcta_prod_tcgen_auto`

### 2026-04-19 follow-up: more aggressive consumer-rescale skip probes reverted

I tested two more experiments-only A/Bs in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change 1:

- keep the offline consumer rescale skip threshold at `0.999f` for `auto`
- make `mm2` more aggressive:
  - `STATIC_CONSUMER_MODE == 3` uses `0.998f`
  - `STATIC_CONSUMER_MODE == 4` stays at `0.999f`

Validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `mm2`: `0.237376 ms`, `mean_abs_diff = 7.4894e-5`
- `auto`: `0.237632 ms`, `mean_abs_diff = 9.5475e-5`

Interpretation:

- both variants regressed versus the working branch
- the more aggressive `mm2` threshold did not produce a speed gain worth keeping

Change 2:

- keep the current `0.999f` threshold on the first consumer rescale
- only from the third tile onward (`idx > 1`), use `0.998f`
- apply that only to:
  - `STATIC_CONSUMER_MODE == 3`
  - `STATIC_CONSUMER_MODE == 4`

Validation on the same surface:

- `mm2`: `0.257504 ms`, `mean_abs_diff = 7.9327e-5`
- `auto`: `0.262976 ms`, `mean_abs_diff = 6.5742e-5`

Interpretation:

- this was a hard speed regression
- tile-position gating does not help the offline consumer rescale path

Conclusion:

- reverted both probes
- restored and rebuilt the working experiments branch:
  - `STATIC_CONSUMER_MODE == 3 || 4` use consumer rescale skip threshold `0.999f`
  - other modes are unchanged

### 2026-04-19 follow-up: `auto` non-first `mm2` guard relaxation reverted

I tested one experiments-only structural A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- in the `NEED_NONFIRST_MM2_GUARD` path for `STATIC_CONSUMER_MODE == 4`
- relax the per-warp `active_rescale` test from:
  - `acc_scale < 1.0f`
- to:
  - `acc_scale < 0.999f`
- this lets `auto` choose `mm2_ABt(...)` on more non-first tiles, aligned with the current
  consumer-side skip threshold

Build/codegen result:

- experiments rebuild passed cleanly
- live kernel codegen stayed flat

Validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `mm2`: `0.242816 ms`, `mean_abs_diff = 8.1761e-5`
- `auto`: `0.239072 ms`, `mean_abs_diff = 6.5256e-5`

Interpretation:

- the branch did not outperform the working state
- `auto` stayed slower than the current best band, and there was no strong accuracy upside to
  justify changing the `p_nonfirst_mm2_ok` contract

Conclusion:

- reverted and rebuilt back to the working experiments branch

### 2026-04-20 follow-up: current experiments branch revalidated, handoff corrected

I rechecked the live three-way surface after noticing the source and the note had diverged.

Actual current experiments source state:

- in the offline consumer rescale loop:
  - `STATIC_CONSUMER_MODE == 3 || 4 || 5` use `correction_threshold = 0.999f`

Revalidation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- short warmed run:
  - `mm2`: `0.192128 ms`, `mean_abs_diff = 7.0072e-5`
  - `auto`: `0.195744 ms`, `mean_abs_diff = 8.7615e-5`
  - `mm2_synced`: `0.193856 ms`, `mean_abs_diff = 1.6222e-4`
- longer confirmation:
  - `mm2`: `0.229376 ms`, `mean_abs_diff = 6.7593e-5`
  - `auto`: `0.232224 ms`, `mean_abs_diff = 5.1034e-5`
  - `mm2_synced`: `0.232352 ms`, `mean_abs_diff = 1.5418e-4`

Interpretation:

- the current source state is healthy
- the previous handoff text claiming the working branch was `3 || 4` only was stale

Conclusion:

- corrected the recorded working experiments state:
  - `STATIC_CONSUMER_MODE == 3 || 4 || 5` use consumer rescale skip threshold `0.999f`

### 2026-04-20 follow-up: per-lane consumer rescale application kept in experiments

I tested one experiments-only A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- keep the existing offline consumer rescale skip threshold:
  - `STATIC_CONSUMER_MODE == 3 || 4 || 5` use `correction_threshold = 0.999f`
- but when a warp enters the consumer rescale path:
  - compute `lane_needs_rescale = correction < correction_threshold`
  - use `applied_correction = lane_needs_rescale ? correction : 1.0f`
- instead of multiplying every lane by its raw `correction` whenever any lane in the warp
  forces `needs_rescale`

Build/codegen result:

- experiments rebuild passed cleanly
- live kernel codegen stayed flat

Warmed fullgrid live check on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- short run:
  - `mm2`: `0.191104 ms`, `mean_abs_diff = 7.9427e-5`, `max_abs_diff = 1.7212e-2`
  - `auto`: `0.195072 ms`, `mean_abs_diff = 8.9907e-5`, `max_abs_diff = 4.6143e-2`
  - `mm2_synced`: `0.193504 ms`, `mean_abs_diff = 1.5567e-4`, `max_abs_diff = 4.9072e-2`
- longer confirmation:
  - `mm2`: `0.191008 ms`, `mean_abs_diff = 7.5829e-5`, `max_abs_diff = 3.8330e-2`
  - `auto`: `0.193312 ms`, `mean_abs_diff = 8.8798e-5`, `max_abs_diff = 5.0537e-2`
  - `mm2_synced`: `0.192768 ms`, `mean_abs_diff = 1.7663e-4`, `max_abs_diff = 4.4678e-2`

Interpretation:

- this is a real live-path win
- the consumer loop was doing unnecessary rescale multiplies on lanes that were already above the
  skip threshold
- after removing that extra work, `mm2` is now clearly best on both speed and error on this host

Conclusion:

- kept
- current live recommendation is now decisively:
  - default: `streaming_live_localcta_prod_tcgen_mm2`
  - fallback only if needed for other reasons: `streaming_live_localcta_prod_tcgen_auto`

### 2026-04-19 follow-up: extending consumer rescale skip to `mm2_synced` reverted

I tested one bounded experiments-only A/B in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change:

- extend the current offline consumer rescale skip threshold from:
  - `STATIC_CONSUMER_MODE == 3 || 4`
- to:
  - `STATIC_CONSUMER_MODE == 3 || 4 || 5`
- so `mm2_synced` also skips near-no-op consumer rescale when `correction >= 0.999f`

Build/codegen result:

- experiments rebuild passed cleanly
- live kernel codegen stayed flat

Validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4`:

- `mm2`: `0.252512 ms`, `mean_abs_diff = 7.4792e-5`
- `auto`: `0.251808 ms`, `mean_abs_diff = 6.9241e-5`
- `mm2_synced`: `0.248544 ms`, `mean_abs_diff = 1.3082e-4`

Interpretation:

- all three live variants regressed materially
- extending the consumer skip to `mm2_synced` is not a viable direction

Conclusion:

- reverted and rebuilt back to the working experiments branch

### 2026-04-20 follow-up: base production consumer-rescale skip reverted

I tested one bounded base-kernel A/B in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- in the base production consumer correction loop
- when `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE` is true
- treat `correction >= 0.999f` as a no-op and skip the TT-output rescale

Build/codegen result:

- base rebuild passed cleanly
- hot-kernel codegen stayed flat:
  - `kernel_fp4pv = 16/20`
  - `1328 B` smem

Validation result:

- the narrow production-only validation path did not return a usable row cleanly on this branch
- that is enough to treat it as not production-healthy

Conclusion:

- reverted and rebuilt back to the clean base branch

### 2026-04-19 follow-up: score-only `ex2.approx` kept in base kernel

I tested one bounded hot-path change in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- add `fp4pv_exp2_approx(float)`
- in the direct-row-update post-QK loop, replace only the per-score:
  - `scores_reg[si].x = exp2f(scores_reg[si].x)`
  - `scores_reg[si].y = exp2f(scores_reg[si].y)`
- with:
  - `scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x)`
  - `scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y)`
- keep the correction-path / `acc_scale` exponent exact

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `16/20` spills and `1328 B` smem

Narrow production-only validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- run 1 warm timings: `0.256289`, `0.242337` ms
- run 2 warm timings: `0.252673`, `0.229410` ms
- `mean_abs_diff`: `8.61e-4` to `8.65e-4`
- `max_abs_diff`: `5.30e-2` to `5.93e-2`
- `lse_max_abs_diff = 1.6070e-2`

Baseline on the restored pre-change branch on the same narrow surface was:

- warm timings: `0.281986`, `0.250049` ms
- `mean_abs_diff = 8.7592e-4`
- `max_abs_diff = 5.9326e-2`
- `lse_max_abs_diff = 1.6070e-2`

Interpretation:

- accuracy stayed in-family
- both reruns came back faster than the current baseline on both warm samples
- this is a keep for the base production kernel

Current kept state:

- float-source `block_amax` helper keep
- score-only `ex2.approx` keep in the base direct-row-update path
- base `kernel_fp4pv = 16/20`
- `1328 B` smem

### 2026-04-19 follow-up: experiments-only `acc_scale` approximation kept

I propagated the kept score-only `ex2.approx` change into:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

and then tested one experiments-only follow-up:

- replace the live-kernel `acc_scale = exp2f(acc_scale_log2)` sites with
  `fp4pv_exp2_approx(acc_scale_log2)`
- also route `softmax_rescale_factor_strict(...)` through the same approximate exponent
- base production kernel was left exact during this experiment

Build/codegen result:

- experiments live kernels stayed compile-flat
- live fullgrid kernels remained at:
  - `consumer_mode=5/4/3/2/0`: `16/20`
  - `consumer_mode=1`: `0/0`
  - `consumer_mode=-1`: `16/32`

Warmed fullgrid live rechecks on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- run A medians:
  - `prod_tcgen_mm2 = 0.251104 ms`, `mean_abs_diff = 9.41e-5`
  - `prod_tcgen_auto = 0.262016 ms`, `mean_abs_diff = 7.55e-5`
  - `prod_tcgen_mm2_synced = 0.249536 ms`, `mean_abs_diff = 1.14e-4`
- run B medians:
  - `prod_tcgen_mm2 = 0.243072 ms`, `mean_abs_diff = 7.29e-5`
  - `prod_tcgen_auto = 0.249440 ms`, `mean_abs_diff = 5.56e-5`
  - `prod_tcgen_mm2_synced = 0.241184 ms`, `mean_abs_diff = 1.17e-4`

Interpretation:

- this is a real live-path speed improvement
- `mm2` and `mm2_synced` are now both ahead of the earlier `~0.259-0.263 ms` band
- `mm2_synced` was fastest on the two longer reruns, but at a modest accuracy cost
- `mm2` still has the cleaner speed/accuracy balance, so it remains the default live candidate for now

Conclusion:

- kept in the experiments kernel

### 2026-04-19 follow-up: base `acc_scale` approximation reverted

I tested the analogous change in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- in the base production direct-row-update path only
- replace `acc_scale = exp2f(acc_scale_log2)` with `fp4pv_exp2_approx(acc_scale_log2)`
- keep the already-kept score-only `ex2.approx` path unchanged

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `16/20` spills and `1328 B` smem

Narrow production-only validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- first rerun returned cleanly:
  - timings: `0.247361`, `0.230753` ms
  - `mean_abs_diff = 8.6643e-4`
  - `max_abs_diff = 5.9326e-2`
  - `lse_max_abs_diff = 1.6070e-2`
- second rerun under the same `timeout 45s` surface stalled and timed out

Interpretation:

- the first run looked promising
- but the branch is not stable enough to keep in the base production kernel

Conclusion:

- reverted and rebuilt base back to the stable kept branch:
  - float-source `block_amax` helper keep
  - score-only `ex2.approx` keep in base
  - base `kernel_fp4pv = 16/20`
  - `1328 B` smem

### 2026-04-19 follow-up: targeted offline fast-rescale keep for `mm2_synced`

I narrowed the noisy all-offline fast-rescale branch down to the only variant that could justify it:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`
- add `softmax_rescale_factor_fast(...)`
- in the live hot site:
  - keep `ONLINE` on `softmax_rescale_factor_strict(...)`
  - keep offline `consumer_mode != 5` on the original exact `softmax_rescale_factor(...)`
  - route only offline `STATIC_CONSUMER_MODE == 5` (`streaming_live_localcta_prod_tcgen_mm2_synced`) through `softmax_rescale_factor_fast(...)`

Build/codegen result:

- compile-flat on the live kernels
- live fullgrid kernels remained at:
  - `consumer_mode=5/4/3/2/0`: `16/20`
  - `consumer_mode=1`: `0/0`
  - `consumer_mode=-1`: `16/32`

Warmed fullgrid live recheck on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- 5-sample medians:
  - `prod_tcgen_mm2 = 0.248128 ms`, `mean_abs_diff = 8.45e-5`
  - `prod_tcgen_auto = 0.246496 ms`, `mean_abs_diff = 6.07e-5`
  - `prod_tcgen_mm2_synced = 0.238656 ms`, `mean_abs_diff = 1.64e-4`
- shorter confirmation:
  - `prod_tcgen_mm2 = 0.272512 ms`
  - `prod_tcgen_mm2_synced = 0.271808 ms`

Interpretation:

- the targeted branch gives `mm2_synced` a real speed advantage on the longer warmed run
- the shorter confirmation still keeps `mm2_synced` slightly ahead, but only narrowly
- the accuracy cost on `mm2_synced` is still higher than plain `mm2`

Conclusion:

- kept as a targeted optimization for `streaming_live_localcta_prod_tcgen_mm2_synced`
- default live recommendation stays `streaming_live_localcta_prod_tcgen_mm2`
  because its speed/accuracy balance is still cleaner

### 2026-04-19 follow-up: fast offline rescale keep widened to `prod_tcgen` / `mm2` / `auto` / `mm2_synced`

I widened the experiments-only offline fast-rescale branch in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Current branch:

- `softmax_rescale_factor_fast(...)` remains experiments-only
- in the live hot site:
  - `ONLINE` still uses `softmax_rescale_factor_strict(...)`
  - offline `STATIC_CONSUMER_MODE == 2 || 3 || 4 || 5` now use `softmax_rescale_factor_fast(...)`
  - offline `STATIC_CONSUMER_MODE == 0 || 1 || -1` keep the original exact helper path

Build/codegen result:

- compile-flat on the live kernels
- live fullgrid kernels remained at:
  - `consumer_mode=5/4/3/2/0`: `16/20`
  - `consumer_mode=1`: `0/0`
  - `consumer_mode=-1`: `16/32`

Warmed fullgrid live recheck on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- 5-sample medians:
  - `prod_tcgen = 0.241376 ms`, `mean_abs_diff = 3.63e-4`
  - `prod_tcgen_mm2 = 0.236992 ms`, `mean_abs_diff = 7.57e-5`
  - `prod_tcgen_auto = 0.238496 ms`, `mean_abs_diff = 6.33e-5`
  - `prod_tcgen_mm2_synced = 0.245632 ms`, `mean_abs_diff = 1.50e-4`

Short confirmation on the current branch:

- `prod_tcgen_mm2 = 0.237440 ms`, `mean_abs_diff = 7.51e-5`
- `prod_tcgen_auto = 0.238400 ms`, `mean_abs_diff = 5.62e-5`

Interpretation:

- widening the fast offline rescale path materially improved plain `mm2`
- `auto` also remains competitive and slightly more accurate
- `prod_tcgen` remains too inaccurate to promote despite being fairly fast
- `mm2_synced` no longer wins on this wider branch

Conclusion:

- kept
- default live recommendation stays `streaming_live_localcta_prod_tcgen_mm2`
- `streaming_live_localcta_prod_tcgen_auto` is now the closest alternate if accuracy is weighted slightly more heavily than raw speed

### 2026-04-19 follow-up: fast offline rescale keep widened again, `auto` now edges out `mm2`

I widened the experiments-only offline fast-rescale branch one more step in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Current branch:

- offline `STATIC_CONSUMER_MODE == 0 || 2 || 3 || 4 || 5` use `softmax_rescale_factor_fast(...)`
- offline `STATIC_CONSUMER_MODE == 1 || -1` still keep the exact helper path
- `ONLINE` still uses the strict helper path

Build/codegen result:

- compile-flat on the live kernels
- live fullgrid kernels remain:
  - `consumer_mode=5/4/3/2/0`: `16/20`
  - `consumer_mode=1`: `0/0`
  - `consumer_mode=-1`: `16/32`

Warmed fullgrid recheck on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- 5-sample medians:
  - `direct_tcgenaccum = 0.236544 ms`, `mean_abs_diff = 3.60e-4`
  - `prod_tcgen_mm2 = 0.236992 ms`, `mean_abs_diff = 7.76e-5`
  - `prod_tcgen_auto = 0.235744 ms`, `mean_abs_diff = 6.05e-5`
- shorter confirmation:
  - `prod_tcgen_mm2 = 0.239200 ms`, `mean_abs_diff = 8.27e-5`
  - `prod_tcgen_auto = 0.238976 ms`, `mean_abs_diff = 6.86e-5`

Interpretation:

- widening to `consumer_mode == 0` does improve `direct_tcgenaccum`, but not enough to overcome its accuracy gap
- the main useful change is that `prod_tcgen_auto` now edges out `prod_tcgen_mm2` on both speed and accuracy on the current branch
- the gap is still small, so this is a branch-local recommendation rather than a major semantic change

Conclusion:

- kept
- current best live default on this branch is `streaming_live_localcta_prod_tcgen_auto`
- `streaming_live_localcta_prod_tcgen_mm2` remains a near-tie fallback

### 2026-04-19 follow-up: `consumer_mode == 1` widening reverted

I tested one more experiments-only widening step in:

- `b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu`

Change tested:

- extend the offline fast-rescale helper from
  - `STATIC_CONSUMER_MODE == 0 || 2 || 3 || 4 || 5`
- to also include:
  - `STATIC_CONSUMER_MODE == 1`

Build/codegen result:

- compile-flat on the live kernels
- live fullgrid kernels remained:
  - `consumer_mode=5/4/3/2/0`: `16/20`
  - `consumer_mode=1`: `0/0`
  - `consumer_mode=-1`: `16/32`

Warmed fullgrid recheck on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- `direct = 0.383968 ms`, `mean_abs_diff = 2.96e-4`
- `prod_tcgen_mm2 = 0.237248 ms`, `mean_abs_diff = 8.48e-5`
- `prod_tcgen_auto = 0.236512 ms`, `mean_abs_diff = 5.25e-5`

Interpretation:

- widening into `consumer_mode == 1` did not make the accurate direct live reference meaningfully more useful
- it did not improve the main candidate ranking either
- there was no reason to carry the extra branch complexity

Conclusion:

- reverted and rebuilt back to the prior keep branch:
  - offline fast-rescale applies to `consumer_mode == 0 || 2 || 3 || 4 || 5`
  - `consumer_mode == 1` is back on the exact helper path

### 2026-04-19 follow-up: direct float pack primitive reverted

I tested one larger pack-primitive swap in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- in `fp4pv_quantize_scores_group_from_float2(...)`
- keep the current float-source `block_amax` scan
- remove the temporary `bf16_2 scores_group_bf[8]` backing
- replace the TE BF16-backed payload pack
  - `mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(...)`
- with direct scaled-float payload pack through:
  - `fp4pv_packed_float_to_e2m1(...)`

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `16/20` spills and `1328 B` smem

Narrow production-only validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- warm timings: `0.310785`, `0.261698` ms
- `mean_abs_diff = 8.5435e-4`
- `max_abs_diff = 5.0293e-2`
- `lse_max_abs_diff = 1.6070e-2`

Interpretation:

- accuracy stayed in-family
- runtime was slower than the current working branch on both warm samples
- there was no codegen upside to offset the regression

Conclusion:

- reverted and rebuilt back to the working branch:
  - base `kernel_fp4pv = 16/20`
  - `1328 B` smem

### 2026-04-19 follow-up: packed-`u64` helper shape reverted

I tested one more bounded helper-shape A/B on the current working branch in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- keep the float-source `block_amax` keep
- but store the converted BF16 pairs inside `fp4pv_quantize_scores_group_from_float2(...)` as
  four packed `uint64_t` groups instead of eight `bf16_2` elements
- fill those four `u64` groups two `float2` pairs at a time and feed them directly into the
  existing `mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(...)` calls

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `16/20` spills and `1328 B` smem

Validation result:

- a narrow production-only runtime/accuracy check under a hard `45s` shell timeout did not return
  cleanly on this branch
- that is not enough signal to keep a helper-shape swap with zero codegen upside

Conclusion:

- reverted and rebuilt back to the working branch:
  - base `kernel_fp4pv = 16/20`
  - `1328 B` smem

### 2026-04-19 follow-up: quarter-shared scale primitive reverted

I tested one larger semantic A/B in:

- `b300_causal/bf16_b300_mha_causal_fp4.cu`

Change:

- in `fp4pv_pack_scores_to_stage_and_scales(...)`
- replace the two independent 8-score group quantizations per quarter with one quarter-level helper
  that:
  - scans `quarter_amax` across all `16 x float2` values in the quarter
  - computes one shared FP4 decoding scale for the whole quarter
  - packs the two 8-score payload groups with that same `coeff`
  - duplicates the prepared scale into both scale slots for the quarter

Build/codegen result:

- base `kernel_fp4pv` stayed flat at `16/20` spills and `1328 B` smem

Narrow production-only validation on `cuda:1`, `S=4096`, `B=1`, `H=12`, `random_live_fp4` vs stored-`P`:

- warm timings: `6.948355`, `6.941251` ms
- warm median: `6.948355 ms`
- `mean_abs_diff = 8.5066e-4`
- `max_abs_diff = 5.1514e-2`
- `lse_max_abs_diff = 1.6070e-2`

Interpretation:

- accuracy stayed roughly in-family
- runtime was catastrophically worse than the working branch on the same narrow surface
- quarter-shared scale is not a viable speed direction

Conclusion:

- reverted and rebuilt back to the working branch:
  - base `kernel_fp4pv = 16/20`
  - `1328 B` smem

### 2026-04-30 follow-up: B300 backward 2048 hot DQ opt-in improved, stock routes rejected

Scope:

- file set: `b300_bwd_cute16_candidate.cuh`, `b300_bwd_cute16_kernel_candidate.cuh`, `b300_bwd_hot.cuh`
- device: `CUDA_VISIBLE_DEVICES=2` exposing one `NVIDIA GB200`
- main probe:
  - `direct_bwd_probe.py candidate 2048 0 cute 1 ref --dq-debug-groups`
  - timing probes used `TK_FA4_SPLIT_TIMING=1`

Validated dead ends:

- `candidate2` / public `hot` wrapper still runs near `258 us`, but is invalid:
  - `dq_refdiff.count=2048`
  - `dv_refdiff.count=2048`
  - `dk_finite=False`
- `legacy` hot wrapper is invalid on the same input:
  - hits CUDA illegal memory access
- `kUseSharedDsMonolithic2048=true` was exact but slower:
  - `time_us ~= 946`
  - ptxas reported `2004 B` spill stores and `6004 B` spill loads
- `kUseFullStockCuTe2048=true` was worse and invalid for DK/DV:
  - `time_us ~= 1370`
  - `dq` exact-level, but `dk_refdiff.count=2048`, `dv_refdiff.count=2048`
- `cudaClusterSchedulingPolicySpread` on the clustered DQ launch regressed warm split totals to about `727-732 us`

Current best opt-in path:

- keep public/default gate off:
  - `kUseHotClusteredDqOnly2048 = false`
- use runtime opt-in:
  - `TK_FA4_USE_HOT_CLUSTERED_DQ=1`
- q-major clustered DQ is the correct bulk direction:
  - `hot_compute_dq_qmajor_loop(...)`
  - DQ stays under probe threshold with `dq_refdiff.count=0`
  - representative max DQ diff is about `1.77e-4`
- explicit clustered-DQ launch policy:
  - `cudaClusterSchedulingPolicyLoadBalancing`
- deferred clustered-DQ TMA-store wait:
  - wait before reusing `dq_smem`, not immediately after each store
  - final wait before kernel exit

Representative warm timings after load-balancing plus deferred TMA-store wait:

- opt-in hot clustered DQ, no ref, `time_iters=5`:
  - split totals: `658.46`, `662.27`, `654.59`, `655.49`, `645.41`, `651.52`, `665.98 us`
  - final `time_us ~= 703.41`
- default exact split, same rebuilt extension, no ref, `time_iters=5`:
  - split totals: `689.66`, `701.28`, `687.81`, `689.76`, `693.50`, `695.97`, `697.31 us`
  - final `time_us ~= 739.42`

Conclusion:

- the current opt-in q-major clustered DQ path is now a measured improvement over the default exact split, but it is still not the old near-`200 us` recipe
- the remaining bottleneck is still resource contention between exact DK/DV and clustered DQ:
  - default exact DK/DV is about `376-385 us` when paired with the default DQ path
  - opt-in hot clustered DQ still stretches DK/DV to about `617-632 us`
- do not flip this on as the safe default without an explicit tolerance decision, because DQ diff is higher than the exact split even though it is under the probe threshold

Additional rejected follow-ups from the same pass:

- direct final DQ accumulation from the clustered main compiled and removed the reduce kernel, but was invalid:
  - DQ became wrong across all rows
  - representative `dq_refdiff.max_absdiff ~= 2.86`, `dq_refdiff.count=2048`
  - DK/DV remained clean, so the failure was isolated to clustered DQ ownership/layout semantics
  - reverted to scratch-backed rank accumulation plus reduce
- fused patch+reduce compiled and was correct, but was slower:
  - ptxas reported `255` registers, `416 B` spill stores, `800 B` spill loads
  - fused patch+reduce stage took about `45-52 us`
  - split patch plus reduce remained about `42 us` combined
  - left disabled with `kUseFusedPatchReduceClusteredDq = false`

Additional stream scheduling A/B:

- normal-priority DQ stream plus normal-priority DK/DV stream is the current best scheduling variant:
  - no-ref rerun warm split totals were `642.27`, `647.10`, `628.77`, `634.62`, `628.83`, `636.61`, `650.34 us`
  - final no-ref `time_us ~= 681.77`
  - ref check stayed clean with `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - representative ref warm split totals were `654.66`, `687.94`, `676.83 us`
- high-priority DQ with normal-priority DK/DV was worse:
  - DQ main dropped to about `438-452 us`
  - DK/DV stretched to about `631-644 us`
  - total regressed to about `662-679 us`, final `time_us ~= 715.88`
- DK/DV on the current stream with DQ on the normal pool was also unstable:
  - late warm totals were around `840-849 us`, final `time_us ~= 774.44`
- current scheduling state:
  - DQ normal pool stream
  - DK/DV normal pool stream

Additional rejected micro-optimization:

- split masked vs dense calls inside `hot_compute_dq_qmajor_loop(...)`:
  - idea was to apply the causal mask only for the first q block after each k pair and use a dense instantiation for all later q blocks
  - build stayed at `168` registers and no spills for the clustered DQ main kernel
  - runtime did not improve: DQ main stayed around `525-539 us`, total stayed around `638-656 us`
  - reverted to the single masked q-major call to avoid carrying branch/code-size risk
- `kUseSeq2048ExactOwnership=true` was also rejected:
  - ptxas for `main_kernel_causal_seq2048_exact_dkdv_only` reported `255` registers, `908 B` spill stores, `1552 B` spill loads
  - DK/DV improved to about `437-442 us`
  - DQ regressed to about `704-712 us`
  - total regressed to about `717-726 us`, final `time_us ~= 774.79`
  - reverted `kUseSeq2048ExactOwnership = false`
- `kUseCute16HotExact2048=true` was rejected:
  - ptxas for the hot-BSHD DK/DV kernel used `168` registers but still spilled (`904 B` stores, `816 B` loads for causal)
  - mixed sequential path timed around `862-901 us`
  - DQ stayed clean, but DK/DV were invalid against ref:
    - `dk_refdiff.max_absdiff ~= 3.02`, `dk_refdiff.count=2045`
    - `dv_refdiff.max_absdiff ~= 4.20`, `dv_refdiff.count=2041`
  - reverted `kUseCute16HotExact2048 = false`

Host/runtime note:

- late timing reruns on physical GPU 2 became invalid because `nvidia-smi` showed GPU 2 at `100%` utilization
- the same restored build on physical GPU 1 returned to the expected band:
  - warm split totals mostly `638-655 us`
  - representative final `time_us ~= 698.27`
- treat multi-millisecond split timings on GPU 2 from this pass as environmental noise, not a source regression

2026-04-30 continuation:

- Nsight Compute on the hot clustered DQ main showed the real limiter is latency/synchronization, not DRAM bandwidth:
  - DQ main duration under NCU was about `524-534 us`
  - DRAM throughput was only about `2.2%`
  - compute throughput was only about `12.7%`
  - scheduler had `~78.8%` no-eligible cycles
  - warp-state guidance pointed at CTA barrier stalls as the largest issue (`~50%` of warp cycles per issued instruction)
- The old FA4/CuTe-style recipe differs structurally:
  - it uses producer/consumer staging and ping-pong Q/DO/vector buffers
  - this D192 exact clustered path is still single-buffered because duplicating full Q/DO D192 tiles would exceed the shared-memory budget
  - this is why small barrier tweaks do not recover the old near-`200 us` behavior
- Rejected barrier experiment:
  - replacing the midpoint full-block sync with only the compute-group sync kept correctness but regressed clean no-debug timing to about `789 us`
  - restored the full `__syncthreads()` before consumer `0` reads both `ds_warp_smem` producers
- Retained reduce-tail optimization:
  - added a chunked clustered DQ reduce kernel using three parallel D64 warps
  - ptxas for the new reduce kernel reports `72` registers, no spills, no barriers
  - the old full-D192 reduce kernel had been `212` registers
  - internal reduce timing improved modestly to about `16.7-17.1 us`
- Rejected patch chunking:
  - chunking the first-block patch stores was correct but did not reduce ptxas resources (`255` registers and the same tiny spill)
  - clean GPU3 timing regressed to about `660 us`
  - reverted to the full-D patch kernel
- Promoted the validated hot clustered DQ path to the default 2048 causal non-deterministic candidate path:
  - set `kUseHotClusteredDqOnly2048 = true`
  - `TK_FA4_USE_HOT_CLUSTERED_DQ=1` remains usable, but is no longer needed for the default candidate 2048 path
- Clean validation on physical GPU 3:
  - default candidate, no env, no ref, `time_iters=20`: `time_us ~= 591.71` and `592.37`
  - env-forced hot clustered DQ, no ref, `time_iters=20`: `time_us ~= 599.00` and `601.03`
  - default candidate ref check stayed clean: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - representative default ref run: `time_us ~= 621.95`, `dq_refdiff.max_absdiff ~= 1.7738e-4`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
- Host/runtime update:
  - physical GPU 1 also became slow/contaminated during this pass (`~789-803 us` for both hot and default paths)
  - physical GPU 3 restored the expected timing band and should be preferred for the next timing pass
  - `TK_FA4_SPLIT_TIMING=1` currently perturbs full-run timings into multi-ms territory on GPU3; use no-debug timings for end-to-end and treat only `clustered_dq_timing_us` internals as directional

2026-04-30 continuation, later pass:

- Runtime harness note:
  - direct probes needed the CuTe DSL venv plus the older quack install on `PYTHONPATH`:
    - `PYTHONPATH=/workspace/codebases/fp4_matmul/.venv/lib/python3.12/site-packages`
    - `/workspace/codebases/fp4_matmul/.venv-cute/bin/python`
  - without that, the probe failed before launching kernels on `cutlass.cute` / `quack` imports
- Rejected two-stage Q/DO/stat staging for the clustered DQ main:
  - implemented the FA4-like ping-pong shape with `q_smem[2]`, `do_smem[2]`, `lse_log2_smem[2][...]`, `dpsum_smem[2][...]`, and two tensor-memory score/dp slots per consumer
  - build succeeded, but the DQ main used `231540 B` shared memory, very close to the GB200 per-block ceiling
  - correctness was clean: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - no-ref timing was slower than the accepted single-buffer path: about `600-602 us` versus the clean `~591-594 us` band
  - reverted; the simple ping-pong preload does not recover the old FA4/CuTe producer/consumer recipe because the main loop still reconverges through full-block barriers and the larger smem footprint leaves no practical headroom
- Re-tested midpoint full-block sync removal on a cleaner physical GPU 1:
  - correctness stayed clean
  - internal `clustered_dq_timing_us` sometimes improved a few microseconds (`main` roughly `529-550 us` after warmup)
  - end-to-end no-ref timing did not improve (`591.68`, `592.30 us`)
  - reverted to the full `__syncthreads()` before consumer `0` reads both `ds_warp_smem` producers, because there was no end-to-end win and the full sync is the safer memory-ordering point
- Rejected `DonorBulkOnly` clustered mode:
  - correctness stayed clean and DQ max diff was tighter (`~2.42e-5`)
  - ptxas reported spills for the donor-bulk DQ main (`412 B` stores, `436 B` loads)
  - no-ref timing regressed to about `709 us`
  - reverted to `kDqOnlyClusteredMode = LegacyPatched`
- Current validated state after reverts:
  - `kUseHotClusteredDqOnly2048 = true`
  - `kDqOnlyClusteredMode = LegacyPatched`
  - `kUseChunkedClusteredDqReduce = true`
  - full midpoint `__syncthreads()` restored
  - DQ main ptxas remains `168` registers, no spills, `189524 B` smem
  - chunked reduce remains `72` registers, no spills, no barriers
- Current clean validation:
  - physical GPU 1 no-ref, `time_iters=20`: `time_us ~= 593.57`
  - physical GPU 1 ref check: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - representative ref values: `time_us ~= 623.20`, `dq_refdiff.max_absdiff ~= 1.7738e-4`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
  - physical GPU 0 was unavailable, physical GPU 2 had resident memory, and physical GPU 3 produced a noisy slow sample in this pass; prefer whichever of GPU 1 or GPU 3 is clean at the start of the next timing run

2026-04-30 continuation, patch-tail pass:

- Accepted first-block patch causal-tail skip:
  - `dq_only_clustered_first_block_patch_kernel(...)` now skips future K subtiles with `if (kv_offset > warp) continue`
  - this avoids the guaranteed-zero causal work in the first block patch without changing the main clustered DQ kernel
  - the kept version uses the normal masked `repair_dq_step<true, C>(..., false)` for all remaining K subtiles; a dense-past split was correct but carried more ptxas damage and no end-to-end win
- Accepted direct stat loads in the same patch kernel:
  - removed the patch-only shared `lse_log2_smem` / `dpsum_smem` staging and the extra block sync
  - ptxas for the kept patch kernel:
    - `255` registers
    - `232 B` stack
    - `456 B` spill stores, `376 B` spill loads
    - `90136 B` smem, down from `91160 B`
  - DQ main remains unchanged at `168` registers, no spills, `189524 B` smem
  - chunked reduce remains `72` registers, no spills, no barriers
- Rejected follow-ups:
  - dense-past patch call (`kv_offset < warp` as `dense_unmasked`) was correct but ptxas worsened to about `480 B` stores / `408 B` loads and did not improve the no-ref path
  - static dense/masked helper specialization was worse again (`672 B` stores / `564 B` loads), reverted
  - reduce-before-patch with direct final `dq` add compiled much lighter:
    - `28 B` spill stores, `32 B` spill loads
    - `24 B` stack
    - `40984 B` smem
    - but it moved the patch stage to about `24-25 us` and no-ref samples were `597.02`, `592.86`, `594.70 us`; reverted because runtime did not beat the scratch-add patch
- Current clean validation for the kept state:
  - physical GPU 3 ref check: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - representative ref values: `time_us ~= 617.73`, `dq_refdiff.max_absdiff ~= 1.7738e-4`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
  - physical GPU 3 no-ref, `time_iters=20`: `589.78`, `590.95`, and final rebuilt `591.03 us`
  - directional clustered-DQ internals on the kept path are still roughly `main ~= 530-550 us`, `patch ~= 21.8-23.9 us`, `reduce ~= 16.4-18.4 us`
- Host/runtime note:
  - physical GPU 1 repeatedly accumulated stale `python3 -` / `python -` probe children and produced invalid `~1360 us` no-ref samples
  - those stale helpers were killed during the pass; prefer physical GPU 3 for the next timing run unless `nvidia-smi pmon -c 1` shows GPU 1 clean

2026-05-01 continuation, clustered-DQ overlap pass:

- Accepted single-buffer next-Q/DO/stat pre-issue in the clustered DQ main:
  - the load warp now issues the first Q/DO/stat tile before the loop, then pre-issues the next Q block after the post-hot-compute sync and before consumer `0` performs current-block DQ scratch accumulation
  - this overlaps next-iteration Q/DO/stat TMA issue with current DQ accumulation without adding a second full D192 Q/DO shared-memory buffer
  - restored the deferred DQ TMA store wait discipline around scratch stores; correctness stayed clean
- Current ptxas/resource profile after reverting the failed split-consumer experiment:
  - clustered DQ main: `167` registers, no spills, `16 B` stack, `189524 B` smem
  - first-block patch kernel remains `255` registers, `232 B` stack, `456 B` spill stores, `376 B` spill loads, `90136 B` smem
  - chunked reduce remains `72` registers, no spills, no barriers
- Clean validation on physical GPU 3 with the CuTe DSL venv and quack `PYTHONPATH`:
  - ref check after final rebuild: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - representative ref values: `time_us ~= 530.46`, `dq_refdiff.max_absdiff ~= 1.7738e-4`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
  - no-ref `time_iters=20`: earlier samples `502.31`, `504.10`, `502.99 us`; final rebuilt sample `503.44 us`
  - this is the current best validated state, down from the previous patch-tail baseline around `589-591 us`
- Directional timing:
  - `TK_FA4_CLUSTERED_DQ_TIMING=1` warm internals showed DQ main around `446-462 us`, patch around `21-24 us`, reduce around `16-18 us`, total around `486-499 us`
  - `TK_FA4_SPLIT_TIMING=1` showed DK/DV around `384-405 us` and DQ around `499-523 us`; split timing still perturbs final end-to-end time, so use it only directionally
- Rejected split-consumer DQ accumulation:
  - attempted to split DQ scratch accumulation across consumer `0` and consumer `1`, with local and peer `ds*k` written through separate scratch add stores
  - correctness stayed clean, but ptxas regressed to `168` registers with `96 B` spill stores, `96 B` spill loads, and `64 B` stack
  - ref run slowed to about `696.64 us`; reverted to the consumer-`0` local-plus-peer accumulation path
- Host/runtime note:
  - physical GPU 1 had unrelated `/usr/bin/python -u train.py ...` processes from `/workspace/ue5m3_project`; left them alone
  - physical GPU 3 was clean for the accepted validation above and should remain the preferred probe target when `nvidia-smi pmon -c 1` shows it idle

2026-05-01 continuation, startup-overlap pass:

- Accepted two additional clustered-DQ scheduling tweaks:
  - the loop-tail `__syncthreads()` is now skipped on the final Q block, where no later iteration needs the preissued Q/DO/stat data
  - initial Q/DO/stat TMA/stat staging now starts immediately after semaphore initialization and overlaps with K/V staging by the compute warps
  - this removes the separate startup staging sync and better matches the useful part of the FA4/CuTe producer recipe without adding a second full Q/DO buffer
- Accepted follow-up cleanup:
  - the `kv_b` semaphore was removed entirely; the full-block sync after K/V staging already orders K/V before the Q loop
  - ptxas stayed at `168` registers with no spills, and clustered-DQ smem dropped slightly from `189524 B` to `189508 B`
- Current ptxas/resource profile:
  - clustered DQ main: `168` registers, no spills, `16 B` stack, `189508 B` smem
  - first-block patch kernel unchanged: `255` registers, `232 B` stack, `456 B` spill stores, `376 B` spill loads, `90136 B` smem
  - chunked reduce unchanged: `72` registers, no spills, no barriers
  - the extra register does not affect occupancy here because shared memory is the limiting resource
- Clean validation on physical GPU 3:
  - ref check: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - representative ref values: `time_us ~= 527.94`; final current-source ref check after removing `kv_b`: `time_us ~= 529.31`
  - ref diffs stayed clean: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`, with `dq_refdiff.max_absdiff ~= 1.7738e-4`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
  - no-ref `time_iters=20`: `499.20`, `498.38`, `498.66 us`; final current-source after removing `kv_b`: `498.09 us`
  - this supersedes the earlier same-day `~502-504 us` overlap baseline and is the current best validated 2048 candidate path
- Directional clustered-DQ timing with `TK_FA4_CLUSTERED_DQ_TIMING=1`:
  - warm DQ main around `447.1-457.0 us` after removing `kv_b`
  - patch around `20.8-21.9 us`
  - reduce usually around `16.4-18.4 us`
  - total around `484.8-496.0 us`
- Rejected same-day cleanup:
  - removing the `include_peer` runtime branch stayed correct but changed ptxas to `168` registers and did not improve no-ref timing (`~504.57 us`); reverted

2026-05-01 continuation, clustered-DQ cleanup after startup overlap:

- Accepted stat-staging register-lifetime cleanup:
  - the load warp now reuses one `stats_stage_vec` for the LSE and dPsum staging loops instead of keeping separate LSE/dPsum vectors live
  - this restored the clustered-DQ main ptxas profile from `168` registers back to `167` registers, with no spills
- Accepted DQ scratch accumulation lifetime cleanup:
  - consumer `0` now reuses one `ds_reg` for local and peer DQ scratch accumulation
  - ptxas stayed unchanged, correctness stayed clean, and the cleanup was kept because it reduces live state without changing scheduling
- Current ptxas/resource profile:
  - clustered DQ main: `167` registers, no spills, `16 B` stack, `189508 B` smem
  - first-block patch kernel unchanged: `255` registers, `232 B` stack, `456 B` spill stores, `376 B` spill loads, `90136 B` smem
  - chunked reduce unchanged: `72` registers, no spills, no barriers
- Clean validation on physical GPU 3:
  - ref check: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - final ref sample: `time_us ~= 523.20`, with `dq_refdiff.max_absdiff ~= 1.7738e-4`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
  - final no-ref `time_iters=20`: `498.74 us`
  - directional `TK_FA4_CLUSTERED_DQ_TIMING=1` after stat-vector reuse showed warm DQ main around `441.15-453.34 us`, patch around `20.86-21.92 us`, reduce around `16.35-18.43 us`, and total around `479.42-492.64 us`
- Rejected same-day retest:
  - removing the `include_peer` runtime branch was retested after stat-vector reuse, but ptxas again rose to `168` registers; the branch was restored

2026-05-01 continuation, clustered-DQ tail/main micro-screen:

- Final source was restored to the last clearly accepted clustered-DQ shape:
  - clustered DQ main: `167` registers, no spills, `16 B` stack, `189508 B` smem
  - first-block patch kernel: `255` registers, `232 B` stack, `456 B` spill stores, `376 B` spill loads, `90136 B` smem
  - chunked reduce: `72` registers, no spills, no barriers
- Clean validation on physical GPU 3 after the restore:
  - ref check: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - representative final ref sample: `time_us ~= 534.21`, with `dq_refdiff.max_absdiff ~= 1.7738e-4`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
  - comparable no-ref samples remained in the same `~498-500 us` band; one clean sample after restore was `499.62 us`
- Rejected micro-cleanups:
  - removing the startup `q_start_block < q_blocks` guard raised clustered-DQ main ptxas from `167` to `168` registers; reverted
  - enabling fused patch+reduce was correct but slower (`~507.34 us`) and introduced fused-tail spills (`416 B` stores / `800 B` loads); reverted
  - switching from the 3-warp chunked reduce to the 1-warp reduce was slower (`~499.35 us`); reverted
  - removing redundant launched-tail warp guards was exact but not a clear win (`~498.36 us`); reverted
  - removing the DQ-subtile guard and replacing `warpgroup::warpid()` with `warp` were exact and ptxas-neutral, but did not produce a stable timing separation from the accepted baseline; reverted
  - rewriting the `include_peer` branch as `if constexpr` was ptxas-neutral and not a clear win; reverted
- Timing hygiene note:
  - a `~991-993 us` no-ref signature appeared when physical GPU 3 was occupied by an unrelated `tests/bench_ffn.py --m-values 65536` process; treat that signature as contamination unless `nvidia-smi pmon -c 1` proves the device is idle

2026-05-03 continuation, clustered-DQ dense-main and patch live-range pass:

- Accepted first-block patch accumulator live-range cleanup:
  - added a candidate-local `repair_dq_step_accumulate(...)` helper that accumulates the final `ds_bf * k_col` MMA directly into the caller's `dq_partial`
  - `dq_only_clustered_first_block_patch_kernel(...)` now removes the separate `dq_contrib` tile and the explicit `warp::add(...)`
  - ptxas for the kept patch kernel improved from `255` registers with `232 B` stack, `456 B` spill stores, and `376 B` spill loads to `255` registers, `0 B` stack, and no spills
  - warm internal patch timing improved from roughly `20.6-21.9 us` to roughly `18.0-20.1 us`
- Accepted clustered-DQ main dense q-major split:
  - `LegacyPatched` main now calls `hot_compute_dq_qmajor_loop<false, C>(...)` when `q_block_idx > kv_tile_base + consumer_idx`
  - only the remaining diagonal consumer path uses the masked `hot_compute_dq_qmajor_loop<true, C>(...)`
  - ptxas for clustered DQ main is now `168` registers, no spills, `16 B` stack, `189508 B` smem; this is kept because shared memory remains the occupancy limiter and runtime improves
  - directional `TK_FA4_CLUSTERED_DQ_TIMING=1` warm best after this pass: `main ~= 435.23 us`, `patch ~= 18.02 us`, `reduce ~= 16.38 us`, `total ~= 469.63 us`
- Clean final validation on physical GPU 3:
  - ref check: `dq_refdiff.count=0`, `dk_refdiff.count=0`, `dv_refdiff.count=0`
  - final ref sample: `time_us ~= 524.54`, `dq_refdiff.max_absdiff ~= 8.392e-5`, `dk_refdiff.max_absdiff ~= 3.099e-5`, `dv_refdiff.max_absdiff = 0`
  - no-ref `time_iters=20`: representative samples `487.46 us` and final rebuilt `488.30 us`
- Rejected screens from this pass:
  - patch kernel `__launch_bounds__(..., 2)` kept ptxas unchanged but worsened patch timing to about `22.8 us`; reverted
  - `DonorBulkOnly` clustered mode regressed ptxas for clustered DQ main to `168` registers with local-memory traffic (`428 B` stores / `492 B` loads) and `208 B` stack; reverted
  - single `score_tt` / `dp_tt` descriptor variables raised clustered DQ main from `167` to `168` registers; reverted on ptxas risk
  - dense-past first-block patch calls (`kv_offset < warp`) were correct but reintroduced patch spills (`384 B` stores / `612 B` loads); reverted
  - patching both boundary q-blocks and starting main at `kv_tile_base + 2` lowered main to `~425-430 us` but patch cost jumped to `~55-57 us`, so total regressed to `~498-503 us`; reverted
  - a single runtime maybe-mask q-major helper had no ptxas benefit and timed slower (`~474-483 us` directional total) than the kept dual-instantiation split; reverted
  - a narrower second-block diagonal patch plus `q_start = kv_tile_base + 2` kept ptxas nearly clean and lowered main to `~422-437 us`, but patch cost doubled to `~39-40 us` and totals stayed around `~479-493 us`; reverted
  - overlapping odd-q-block reduce on an auxiliary DQ stream was exact and reduced the measured reduce tail to `~12.6 us`, but patch stretched to `~24-26 us` and no-ref regressed to `~493 us`; reverted
  - forcing the hot-clustered DQ opt-in scheduling path (`TK_FA4_USE_HOT_CLUSTERED_DQ=1`) did not beat the current scheduling; warm split totals remained around `~499-511 us`
  - removing the exact-grid `kv_block_idx` guard / unused `cluster_idx` was ptxas-neutral (`168` registers, no spills, `189508 B` smem) and was reverted as cosmetic
  - final restored validation after these rejected screens: ref remained clean with `dq/dk/dv refdiff.count=0`; final no-ref sample was `time_us ~= 487.26`

2026-05-03 continuation, CuTe DSL backward comparison:

- Refreshed the current accepted candidate timing on physical GPU 3 using the CuTe forward path and the required quack/CuTe venv `PYTHONPATH`:
  - no-ref candidate, `time_iters=50`: `time_us ~= 488.01`
  - with `TK_FA4_CLUSTERED_DQ_TIMING=1` / `TK_FA4_SPLIT_TIMING=1`, warm internals were roughly `DQ main ~= 436-446 us`, `patch ~= 18.5-20.2 us`, `reduce ~= 16.5-18.4 us`, split `DK/DV ~= 505-523 us`, split `DQ ~= 519-535 us`; debug event syncs inflate the headline timing, so use no-debug for end-to-end
- Timed the vendored public CuTe DSL SM100 backward on the same shape `(B=1, S=2048, H=16, Dqk=192, Dv=128, causal)`:
  - `_flash_attn_bwd(...)`, `time_iters=50`: `time_us ~= 120.88`
  - outputs are BF16 (`dq/dk/dv`), while the current TK exact probe returns FP32 gradients, so this is not an output-contract-identical comparison; it is still the relevant architectural target for the `~200 us` question
- Code-level delta against `/workspace/codebases/fp4_matmul/flash-attention/flash_attn/cute/flash_bwd_sm100.py`:
  - CuTe DSL uses mandatory 2-CTA D192 (`cluster_size=2`, `use_2cta_instrs=True`) with a single 16-warp main kernel: reduce warps `0-3`, compute warps `4-11`, MMA warp `12`, load warp `13`, relay warp `14`, empty warp `15`
  - its main loop computes `S`, `dP`, `dK`, `dV`, and `dQ` in one TMEM/SMEM pipeline; `dS` is produced once, exchanged across the 2-CTA cluster, consumed by both `dK` and `dQ`, and reduced to global `dq_accum` via `cpasync_reduce_bulk_add_f32`
  - the D192 dQ shape reduces across the cluster-wide K tile (`tile_n * cta_group_size = 256`) and uses `dQ_reduce_ncol = 32`, `dQaccum_reduce_stage = 6`
  - the current TK candidate is a split-overlap design: preprocess, one `dkdv_only` kernel, one `dq_only_clustered` kernel, then first-block patch plus chunked DQ reduce; this means the DQ path recomputes score/probability/dS work instead of consuming the DK/DV path's already-computed `dS`
  - the current TK patch/reduce tail is only about `35-39 us`, so removing it cannot close the `~120 us` vs `~488 us` gap; the dominant missing recipe is the CuTe DSL single-kernel TMEM pipeline, not another small patch-stage cleanup
- Re-screened `kDqOnlyClusteredMode = DonorBulkOnly` after the current cleanup:
  - correctness stayed clean against `ref` (`dq/dk/dv refdiff.count=0`)
  - no-ref `time_iters=50`: `time_us ~= 489.97`, slightly worse than the accepted `LegacyPatched` baseline
  - reverted to `kDqOnlyClusteredMode = LegacyPatched`
- Re-screened the existing `kUseDsScratchDqOnly2048 = true` path as a cheap proxy for CuTe DSL's "compute dS once" structure:
  - correctness stayed clean against `ref` (`dq/dk/dv refdiff.count=0`)
  - no-ref `time_iters=50`: `time_us ~= 490.37`, slightly worse than the accepted recompute-DQ baseline
  - reverted to `kUseDsScratchDqOnly2048 = false`
  - conclusion: writing `dS` to global scratch is not the recipe; CuTe DSL's win is keeping `dS` in the same TMEM/cluster pipeline and consuming it before it leaves the kernel
- Next credible optimization direction:
  - stop spending much time on standalone patch/reduce micro-edits unless they are essentially free
  - to mimic CuTe DSL, build a fused D192 backward kernel that shares the same in-kernel `S/P/dP/dS` dataflow across `dK/dV/dQ`, uses dedicated reduce warps, and writes `dq_accum` through in-kernel TMA reduce instead of a separate global scratch + reduce tail

2026-05-04 continuation, dormant fused-route screens:

- Re-screened the existing `kUseSharedDsMonolithic2048 = true` route as the cheapest in-tree "single-kernel" proxy:
  - cluster-1 monolithic was exact against `ref` (`dq/dk/dv refdiff.count=0`)
  - no-ref samples were `time_us ~= 487.56` and `486.73`, very close to the accepted split baseline sample `487.26`
  - not kept because the delta is sub-microsecond/noise-scale and the kernel still uses per-warp SIMT fragments plus global DQ add stores, not the CuTe DSL TMEM/warp-specialized recipe
- Re-screened the same monolithic route with `SharedDsMonolithicConfig = config<..., 2>` to test the 2-CTA ownership shape:
  - correctness stayed clean against `ref`
  - no-ref sample was `time_us ~= 486.97`, again noise-scale versus the split baseline
  - reverted to the accepted split constants
- Re-screened `kUseCuTe16NativeExact2048 = true`:
  - correctness stayed clean against `ref`
  - no-ref sample regressed to `time_us ~= 501.28`
  - reverted to `kUseCuTe16NativeExact2048 = false`
- Conclusion:
  - simply switching to the dormant monolithic/full-stock wrappers does not recover the CuTe DSL `~120 us` public backward timing
  - the missing recipe remains the SM100 CuTe DSL structure: one 2-CTA, 16-warp, TMEM-resident mainloop with dedicated reduce warps and in-kernel `dQaccum` TMA-reduce, not a global scratch/add-store monolith

2026-05-04 continuation, direct-final clustered DQ screen:

- Important build hygiene:
  - header-only edits in this tree do not necessarily rebuild the extension; use `touch tk_fa4.cu && make` before trusting timings
  - several early micro-screens in this pass were invalid because the loaded `_C` binary was stale, so only the rebuilt results below should be used
- Accepted direct-final clustered DQ path:
  - added a full-width `dq_full` TMA descriptor and direct `store_add_async` from the clustered DQ main kernel into the final `dq`
  - added a direct first-block causal patch kernel that also TMA-adds full-width `dq` directly
  - wrapper zeroing now clears final `dq` on the DQ stream for this mode instead of clearing `dqacc`
  - this removes the chunked global scratch plus separate DQ reduce tail from the hot 2048 causal candidate path
- Validation after rebuild:
  - strict `candidate 2048 0 cute 20 ref` on physical GPU 3 stayed exact: `dq/dk/dv refdiff.count=0`
  - strict timing samples: `time_us ~= 467.00` and `467.11`
  - no-ref samples: `time_us ~= 468.31` and a noisy `479.89`
  - component timing with `TK_FA4_SPLIT_TIMING=1 TK_FA4_CLUSTERED_DQ_TIMING=1`: clustered DQ main about `430-439 us`, direct patch about `18.8-20.9 us`, reduce `0.00 us`
  - the overlapping split is now limited by the exact DK/DV kernel in instrumented runs, roughly `480-495 us`
- Rejected after rebuild:
  - DK chunk-store epilogue was exact but ptxas did not improve the DK/DV kernel (`255` registers, `1596 B` spill stores, `3988 B` spill loads) and strict timing regressed to `time_us ~= 470.30`; reverted
  - moving DK/DV back to its own normal-pool stream was exact but slower in no-ref screens (`time_us ~= 471.57` and `473.39`); reverted to caller-stream DK/DV plus pooled DQ
  - reducing DK/DV Q/DO staging from `8` buffered tiles to `4` cut smem from `83992 B` to `42008 B`, but ptxas worsened to `1704 B` spill stores / `3692 B` spill loads and no-ref timing regressed to `time_us ~= 485.22`; reverted
  - increasing DK/DV Q/DO staging to `16` buffered tiles exploded ptxas to `4204 B` spill stores / `12416 B` spill loads; reverted without timing
  - forcing DK/DV `__launch_bounds__(..., 2)` reduced register count to `128`, but ptxas exploded to `7276 B` spill stores / `9332 B` spill loads; reverted without timing
  - substituting the in-tree TMEM-style hot BSHD DK/DV route confirmed why it is not the current recipe:
    - stock full-tile SMEM/TMA epilogue produced nonfinite DK/DV (`dk_nonfinite.count=644`) even though DQ stayed exact
    - per-warp chunk stores removed nonfinites, but DK/DV remained wrong (`dk_refdiff.count=2045`, `dv_refdiff.count=2041`) and strict timing regressed to `time_us ~= 627.64`
    - reverted to the exact SIMT DK/DV kernel despite its spills
- Small safety cleanup kept:
  - the env-forced hot clustered DQ probe path now also zeros final `dq` when `kUseDirectFinalClusteredDq=true`, rather than zeroing unused `dqacc`
- Final validation after reverting rejected DK/DV screens:
  - strict `candidate 2048 0 cute 20 ref`: `time_us ~= 467.08`, `dq/dk/dv refdiff.count=0`
  - no-ref `time_iters=80`: `time_us ~= 467.92`
- Current implication:
  - the earlier direct-final failure was not inherent; the necessary pieces are final-`dq` zeroing, direct first-block patching, and a full-width DQ TMA descriptor
  - more DQ scratch/reduce work is now low leverage
  - the next credible target is the spilled exact DK/DV SIMT path, or replacing the split design with the CuTe DSL-style single-kernel TMEM pipeline

2026-05-05 continuation, direct-final DQ micro-screens:

- Kept one DQ-only cleanup:
  - removed the same-warp `__syncwarp()` immediately before direct-final DQ TMA add stores in the clustered DQ main kernel and direct first-block patch kernel
  - ptxas resources stayed unchanged:
    - clustered DQ main: `168` registers, no spills, `189508 B` smem
    - direct first-block patch: `255` registers, `4 B` spill stores / `8 B` spill loads, `90136 B` smem
    - DK/DV unchanged at `255` registers, `1596 B` spill stores / `3988 B` spill loads
  - strict GPU3 gate stayed exact: `candidate 2048 0 cute 20 ref`, `dq/dk/dv refdiff.count=0`, `time_us ~= 467.56`
  - clean GPU3 no-ref samples after the edit were `467.77`, `465.96`, and `466.17 us`
  - DQ debug timing showed the expected direct-final band, with warm main DQ around `431-435 us` and patch around `19-21 us`
- Rejected/reverted screens:
  - reusing the `dp` FP32 tile as `dS` in the DQ/DKDV repair helpers was exact but ptxas stayed unchanged and same-GPU2 no-ref A/B was worse (`~793.76 us` rewrite vs `~788.22/790.77 us` reverted)
  - changing the diagonal DK/DV subtile loop from full unroll to `#pragma unroll 1` did not change ptxas; timing became contaminated by a training job, so the unproven edit was reverted
  - switching the 2048 DQ path to `kUseDedicatedLoadDqOnly2048=true` was exact but much slower:
    - dedicated DQ kernel introduced `300 B` spill stores / `1464 B` spill loads
    - strict GPU2 timing regressed to `time_us ~= 1205.07`
    - reverted to hot clustered direct-final DQ
- Runtime caveat from this pass:
 - physical GPU3 became occupied by a training process mid-screen and GPU2 was clean but low-clocked, so use the GPU3 samples above for comparison with the prior `~467 us` baseline
  - final rebuilt current path stayed exact on GPU2: `candidate 2048 0 cute 20 ref`, `dq/dk/dv refdiff.count=0`

2026-05-05 continuation, clustered DQ first-block-in-main screen:

- Accepted one DQ ownership cleanup:
  - `kUseMainFirstBlockClusteredDq=true` lets the clustered DQ main kernel process `q_start_block = kv_tile_base`
    for the legacy patched direct-final mode instead of launching the separate direct first-block patch kernel
  - the direct patch launch is now skipped in the `kUseDirectFinalClusteredDq` path
  - ptxas for the active clustered DQ main stayed unchanged at `168` registers, no spills, `16 B` stack, `189508 B` smem
  - DK/DV ptxas stayed unchanged at `255` registers, `552 B` stack, `1596 B` spill stores, `3988 B` spill loads
- Validation on physical GPU3 after rebuild:
  - strict `candidate 2048 0 cute 20 ref`: exact, `dq/dk/dv refdiff.count=0`, samples around `459.85-460.51 us`
  - no-ref `candidate 2048 0 cute 80`: `458.43`, then final rebuild samples `459.56` and `459.82 us`
  - DQ component timing changed from main `~431-438 us` plus direct patch `~19-21 us` to main `~441-449 us` plus no real patch work (`~2.7 us` event gap), so the net win is from removing the separate patch kernel despite a slightly longer main loop
- Rejected/reverted screens:
  - split direct-final DQ writeback into D64 chunks: exact and reduced clustered DQ main ptxas from `168` to `128` registers, but tripled TMA add traffic and strict timing regressed to `~991 us`
  - skipping the masked peer warpgrouper for the first causal block deadlocked, likely because the hot loop expects both consumer groups to participate in its semaphore/barrier protocol
  - changing clustered launch scheduling from `cudaClusterSchedulingPolicyLoadBalancing` to `cudaClusterSchedulingPolicyDefault` stayed exact but regressed strict timing to `~488 us`
  - changing active DQ main Q/dO TMA loads from `cache_policy::NORMAL` to `EVICT_LAST` stayed exact but did not improve no-ref timing (`~458.88 us` vs the kept `~458.43 us` best)
- Current implication:
  - DQ remains the critical path: split timing showed DQ about `457-465 us`, DK/DV about `377-385 us`, and total about `468-476 us`
  - simple chunking/cache/scheduling changes are not enough; further progress likely needs reducing final-DQ TMA add count, cluster-level Q/dO reuse, or the CuTe DSL-style fused TMEM pipeline rather than more register-only reshaping

2026-05-05 continuation, direct-final DQ cluster-size screen:

- Accepted a small direct-final DQ launch-shape cleanup:
  - switched the active `HotClusteredDqConfig` wrapper alias from `dq_only_clustered_config<..., ClusterSize=2>` to `dq_only_clustered_cluster1_config<..., ClusterSize=1>`
  - the direct-final DQ kernel does not use cluster DSM or cross-CTA shared state, so the 2-CTA cluster launch was unnecessary for the current exact path
  - ptxas for the active clustered DQ main stayed unchanged at `168` registers, no spills, `16 B` stack, `189508 B` smem
  - DK/DV ptxas stayed unchanged at `255` registers, `552 B` stack, `1596 B` spill stores, `3988 B` spill loads
- Validation on physical GPU3 after rebuild:
  - baseline before the edit: strict `candidate 2048 0 cute 20 ref` exact at `459.01 us`
  - strict checks after the edit stayed exact with `dq/dk/dv refdiff.count=0`: samples included `457.00`, `453.42`, and final rebuilt `453.98 us`
  - no-ref `candidate 2048 0 cute 80` samples included `453.13`, `453.07`, `453.70`, and `454.19 us`, with one noisy outlier at `487.07 us`
  - instrumentation remains perturbing, but the DQ main timing stayed in the same broad `~436-447 us` band; the win appears to be launch/scheduling overhead rather than lower ptxas resource use
- Rejected/reverted scheduling follow-ups:
  - omitting cluster launch attributes for `ClusterSize=1` stayed exact but did not improve timing (`~453.48 us` no-ref, `~455.43 us` strict); reverted to explicit cluster attributes
  - changing the cluster scheduling policy from `cudaClusterSchedulingPolicyLoadBalancing` to `cudaClusterSchedulingPolicyDefault` stayed exact but did not beat the kept variant (`~455.51 us` no-ref, `~454.68 us` strict); reverted to load-balancing
- Current implication:
  - this is a real but modest cleanup, not the CuTe DSL recipe
  - the remaining gap still requires reducing direct-final DQ TMA-add traffic or moving to a CuTe-style fused TMEM/reduce pipeline

2026-05-06 continuation, PTX-guided DQ writeback screens:

- Compared the current TK DQ writeback against the CuTe DSL dump:
  - CuTe uses three raw `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` operations (`UBLKRED.G.S.ADD.F32.RN`) after staging contiguous DQ chunks in shared memory
  - the current TK direct-final DQ path uses tensor-map reductions through `warp::tma::store_add_async` (`UTMAREDG.5D.ADD`) because the public candidate output is BSHD, so each fixed-head sequence tile is strided by `H * D`
  - CuTe also uses warpgroup `setmaxnreg` budgeting; this was not ported because the current split DQ kernel has eight compute warps plus a single load warp, not a clean all-warpgroup protocol
- Rejected/reverted screens on physical GPU3:
  - a barrier-preserving zero-`ds` participant for the first fully future-masked peer consumer still timed out in the strict gate; the q/dO plus score/dp semaphore cadence cannot be shortened that way
  - a raw full-tile bulk DQ reduction lowered active DQ ptxas from `168` to `159` registers, but was incorrect for BSHD output (`dq_refdiff.count=2048`) because raw bulk cannot express the strided fixed-head rows; this would require a BHSD DQ scratch plus final transpose/copy, not a direct swap
  - switching the inter-store wait from full `store_async_wait()` to `store_async_read_wait()` stayed exact but did not beat the kept timing (`453.34`, `454.94`, `454.14 us` no-ref samples)
  - changing direct-final DQ TMA-add cache policy from `NORMAL` to `EVICT_FIRST` stayed exact but regressed badly (`~981 us` strict sample)
- Current implication:
  - the direct CuTe `UBLKRED` recipe is blocked by output layout, not just missing inline PTX
  - the next meaningful writeback experiment would need either a BHSD-output scratch/copy tradeoff or a different DQ ownership/layout, while smaller wait/cache-policy tweaks have now been screened

2026-05-08 continuation, CuTe PTX-guided DQ screens:

- Rechecked the current local CuTe reference point on physical GPU3:
  - local CuTe DSL autograd backward at the same `B=1,S=2048,H=16,Dqk=192,Dv=128` shape timed at `~264.14 us`
  - that path returns BF16 gradients, while the current TK candidate returns float gradients and is gated against the float `ref` backend, so the numbers are not directly interchangeable
  - preallocated TK outputs did not matter: allocated `candidate_internal` was `~453.01 us`, and `candidate_out_internal` was `~453.03 us`
- Rejected/reverted DQ writeback screens:
  - splitting the active direct-final full-width DQ TMA add into three 64-wide tensor-map reductions stayed exact but regressed (`~470.44 us` strict, `~469.46 us` no-ref)
  - partial DQ source double-buffering for three of four subtiles compiled under the SMEM limit (`226372 B`, same `168` registers, no spills) and stayed exact, but regressed (`~467.31 us` strict, `~467.92 us` no-ref)
  - full DQ source double-buffering remains blocked by SMEM (`238660 B` required vs `0x38c00` max); the partial-buffer result makes that direction low confidence even if enough SMEM is freed
  - switching clustered DQ to raw scratch accumulation plus a copy/layout kernel was not usable in this form; both full-width raw scratch and chunked raw scratch variants hit illegal memory access in the probe
  - full-stock in-repo CuTe-style routing was not a keeper: it timed at `~1315.54 us` and DK/DV were wrong against `ref` (`dk_refdiff.count=2048`, `dv_refdiff.count=2048`)
- Rejected/reverted DK/DV screens after comparing the CuTe PTX/resource dump:
  - CuTe DSL backward main resource file reports `REG:128 STACK:0 LOCAL:0` and the PTX uses explicit `setmaxnreg` regions plus raw `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`; the active TK split still has DK/DV at `255` registers with `1596/3988` spill store/load bytes
  - `seq2048_exact` DK/DV compiled with lower spills (`908/1552` bytes) and stayed exact, but its existing DQ pairing regressed to `~710.30 us`; pairing it with the current hot clustered DQ was also exact but still slower (`~478.85 us` strict, `~479.84 us` no-ref)
  - routing split DK/DV through the low-register hot BSHD DK/DV kernel compiled at `168` registers and looked fast (`~448-453 us`), but DK/DV were non-finite and failed `ref`; removing its explicit tensor-memory zero-store did not fix it
  - a compact `WarpTiles=4` DK/DV-only specialization reduced spills (`988/2100` bytes) and stayed exact, but doubled DK/DV CTAs and regressed (`~475.52 us` strict, `~476.60 us` no-ref)
- Rejected/reverted descriptor and layout screens from the same pass:
  - `dq_full` descriptor prefetch stayed exact but did not improve no-ref timing (`~456.17 us`; one strict sample was a noisy `~639.96 us`)
  - removing both Q and dO descriptor prefetches regressed badly (`~586.28 us`); removing only dO was `~455.22 us`, removing only Q was `~454.12 us`; keep both prefetches
  - a BHSD-storage/view experiment compiled but was wrong (`dq_refdiff.count=2048`) and slow (`~1085.66 us` no-ref); the naive coordinate swap is not the CuTe scratch recipe
- Restored baseline after the rejected screens:
  - build succeeds with active clustered DQ ptxas unchanged: `168` registers, no spills, `189508 B` shared memory
  - strict `candidate 2048 0 cute 20 ref` on GPU3 stayed exact with `dq/dk/dv refdiff.count=0`, timing `~453.28 us`
  - no-ref `candidate 2048 0 cute 80` after restore was `~453.44 us`
  - after the DK/DV screens above, the restored build again passed strict `candidate 2048 0 cute 20 ref` on GPU3 with `dq/dk/dv refdiff.count=0`, timing `~454.54 us`, and no-ref `candidate 2048 0 cute 80` was `~454.81 us`
- Follow-up rejected/reverted screens from the same CuTe-PTX pass:
  - a cluster1 scratch-copy variant for the non-direct DQ path built but hit illegal memory access in the strict probe; the attempted copy kernel was removed and `kUseDirectFinalClusteredDq=true` restored
  - routing the diagonal/first-block DQ work through the existing direct patch kernel stayed exact but regressed badly (`~564.10 us` strict); the extra launch plus `255`-register patch kernel outweighed the removed main-kernel work, so `kUseMainFirstBlockClusteredDq=true` remains the keeper
  - shrinking DK/DV Q/dO staging from 8 tiles to 4 tiles while keeping `WarpTiles=8` cut DK/DV shared memory (`~83,992 B` to `~42,008 B`) but kept `255` registers, shifted spills to `1704/3692`, and regressed strict timing to `~474.17 us`; restored the 8-tile staging
  - replacing the direct-final DQ memset with "first contribution normal store, later contributions TMA add" is not safe inside one unordered grid: later CTAs can add before the first CTA stores, and the normal store can overwrite those adds. Avoid this unless it is split into an ordered init kernel or redesigned through scratch.
- Final restored validation after these follow-ups:
  - rebuilt keeper ptxas was back to active DQ `168` registers/no spills/`189508 B` smem and DK/DV `255` registers with `1596/3988` spill store/load bytes
  - strict `candidate 2048 0 cute 20 ref` rerun was exact at `~455.16 us`
  - no-ref `candidate 2048 0 cute 80` was `~453.70 us`
  - a single strict `~987.32 us` sample immediately before the rerun was treated as a transient outlier because the next strict and no-ref samples returned to the keeper band
- Follow-up instrumentation and stream-priority screen:
  - added debug-only `TK_FA4_SPLIT_TIMING` event breakdown for DQ zero, DQ wait, and DQ kernel window in the split candidate launcher
  - with `TK_FA4_SPLIT_TIMING=1 TK_FA4_CLUSTERED_DQ_TIMING=1`, steady samples showed DQ zero only about `13.47-16.77 us`, DQ wait about `3.01-5.41 us`, clustered DQ main about `433.98-440.48 us`, direct patch about `2.94-3.36 us`, and DK/DV about `475.52-506.85 us`
  - implication: the direct-final DQ memset is measurable but not the main blocker; the real remaining costs are the clustered DQ main window and the high-register/spilling DK/DV path
  - moving `dq_stream` to a high-priority CUDA stream stayed exact but regressed strict timing to `~591.87 us`; reverted to the normal-priority stream
  - after the stream-priority revert, the rebuilt keeper again passed strict `candidate 2048 0 cute 20 ref` on GPU3 with `dq/dk/dv refdiff.count=0` at `~454.97 us`, and no-ref `candidate 2048 0 cute 80` was `~454.52 us`
- Follow-up CuTe contract check and DK/DV launch-bound screen:
  - direct CuTe autograd probe confirmed BF16 inputs produce BF16 `dq/dk/dv`; the current TK candidate still exposes float32 gradients, so the output contract is a real recipe difference
  - CuTe's PTX still uses `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` for DQ accumulation, with a separate postprocess PTX doing `cvt.rn.bf16x2.f32` and vector BF16 stores; simply changing TK output tensors to BF16 would not mimic CuTe
  - forcing the active `config` `MinBlocksPerSm` from `1` to `2` capped DK/DV at `128` registers but exploded DK/DV spills from `1596/3988` to `7276/9332` bytes and regressed strict timing to `~1119.94 us`; reverted
  - after the launch-bound revert, the rebuilt keeper again passed strict `candidate 2048 0 cute 20 ref` on GPU3 with `dq/dk/dv refdiff.count=0` at `~453.70 us`, and no-ref `candidate 2048 0 cute 80` was `~454.97 us`
- Added an opt-in CuTe-contract DK/DV output variant:
  - `candidate_bf16_dkdv` keeps default `candidate` on float `dq/dk/dv`, but instantiates the DK/DV-only kernel with BF16 `dk/dv` output globals and returns BF16 `dk/dv` tensors
  - ptxas for the BF16-DK/DV kernel is unchanged versus float DK/DV: `255` registers, `1596/3988` spill store/load bytes, `83992 B` shared memory
  - default `candidate` remained exact after the split instantiation: strict `candidate 2048 0 cute 20 ref` at `~454.00 us`, no-ref `candidate 2048 0 cute 80` at `~453.24 us`
  - `candidate_bf16_dkdv 2048 0 cute 20 ref` was finite and faster at `~451.87 us`; as expected for BF16 outputs, it no longer satisfies the old float strict threshold for DV (`dv_refdiff.max_absdiff ~= 0.0141`, `dv_refdiff.count=2` at threshold `1e-2`)
  - no-ref `candidate_bf16_dkdv 2048 0 cute 80` was `~451.45 us`, a small but real `~1.8 us` win versus the rebuilt float keeper while matching CuTe's BF16 gradient contract for DK/DV
- Current implication:
  - the remaining CuTe gap is not allocation overhead or a trivial tensor-map chunking issue
  - the real recipe difference is CuTe's contiguous `dQaccum`/postprocess design, BF16-gradient contract, and warp-specialized low-register main kernel; matching it in TK likely requires a deliberate BF16-gradient/contiguous-scratch route, not more direct-final BSHD TMA tweaks or narrower DK/DV CTA splitting

2026-05-09 continuation, BF16-gradient cleanup and cluster-1 launch rescreen:

- Cleaned up the failed direct BF16-DQ experiment:
  - removed the temporary `candidate_bf16_grads` Python/backend binding after compile proved the route invalid
  - the failure was structural: TK's direct-final DQ tensor-map reduction descriptor is built from `st<float,...>` tiles, so binding a BF16 `dq` global pointer to that descriptor fails at compile time
  - this matches the CuTe PTX read: CuTe still accumulates/reduces DQ as FP32 and then uses a separate postprocess kernel with BF16 vector stores
  - implication: BF16 DQ needs a CuTe-like FP32 `dQaccum` plus postprocess path, not a direct BF16 output pointer on the existing TMA-reduce store
- Re-screened the dormant native CuTe16 exact wrapper:
  - temporarily routed `candidate` through `kUseCuTe16NativeExact2048=true`
  - strict `candidate 2048 0 cute 20 ref` timed out under the 90s gate
  - reverted; the older in-tree CuTe16 wrapper is not the public CuTe DSL recipe and does not recover the `~264 us` local CuTe DSL timing
- Accepted a tiny cluster-1 DQ launch cleanup:
  - `launch_backward_dq_only_clustered` now uses a plain CUDA kernel launch when `C::ClusterSize == 1`
  - real clustered launches with attributes are still used for `ClusterSize > 1`
  - rebuild succeeded; active DQ ptxas remained `168` registers, no spills, `189508 B` smem, so this only changes launch/scheduling path, not kernel resources
- Validation on physical GPU3:
  - default strict `candidate 2048 0 cute 20 ref`: exact with `dq/dk/dv refdiff.count=0`, `~454.04 us`
  - default no-ref `candidate 2048 0 cute 80`: `~453.76 us`, then `~453.05 us`
  - opt-in `candidate_bf16_dkdv 2048 0 cute 20 ref`: finite, `~452.42 us`, with DQ still exact-ish and expected BF16-output DK/DV drift (`dv_refdiff.max_absdiff ~= 0.0141`, `dv_refdiff.count=2`)
  - opt-in no-ref `candidate_bf16_dkdv 2048 0 cute 80`: `~450.99 us`
- Current implication:
  - the keeper exact default is back in the `~453-454 us` band, and the CuTe-contract BF16 DK/DV opt-in remains the fastest available variant at `~451 us`
  - the remaining gap is still DQ architecture: CuTe DSL has a low-register warp-specialized main kernel plus FP32 `dQaccum`/BF16 postprocess, while TK is still direct-final BSHD TMA-reducing from a `~190 KB` shared-memory DQ kernel

2026-05-01 continuation, CuTe DSL MXFP4 QK forward bring-up:

- Added a standalone CuTe DSL blockscaled MXFP4 QK MMA issue smoke:
  - file: `tk_fa4/cute_dsl_mxfp4_qk_mma_tile_smoke.py`
  - QK MMA tile: `(128, 128, 256)` with `CtaGroup.ONE`
  - scale layout: generic blockscaled SFA/SFB layout from the D192 geometry
  - launch grid now covers the real flattened `(batch, head)` L dimension
- Added an issue-only fused D192 forward smoke:
  - file: `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`
  - current mode: `qk_mma_issue`
  - this launches QK issue over the real Q/K tile grid but still does not run online softmax or PV
- Validated issue-only runtime on `cuda:1`:
  - standalone QK issue smoke passes for `S=128` and `S=4096`
  - fused issue-only forward returns `qk_issue_grid=(1, 1, 12)` at `S=128`
  - fused issue-only forward returns `qk_issue_grid=(32, 32, 12)` at `S=4096`
  - representative fused issue timing: about `61.98 us` at `S=128`, about `151.80 us` at `S=4096`
- Added a debug accumulator store path to the standalone QK smoke:
  - uses a tmem-to-register epilogue copy and SIMT global store
  - epilogue threads use an epilogue-local thread id; using the full CTA thread id left the first output lanes unwritten
- Deterministic accumulator validation on `cuda:1`, shape `(B=1, Sq=128, Sk=128, H=12)`:
  - zero raw Q/K/scales: scores are exactly `0.0`, no NaNs
  - padded all-ones raw Q/K with scale byte `0x7f`: scores are exactly `256.0`, no NaNs
  - packed-row unpadded all-ones D192: scores are exactly `192.0`, no NaNs
- Rejected diagnostic:
  - zeroing a `256`-byte suffix per row produced scores from `128.0` to `256.0`
  - this was a host-layout mistake: a logical D256 FP4 row is physically `128` bytes, so the correct D192 padding mask zeros bytes `[96:128]` in each packed row
- Current forward path status:
  - QK MXFP4 issue and accumulator visibility are now validated for simple deterministic inputs
  - the fused CuTe DSL forward path is still issue-only
  - next step is to consume the QK accumulator in an online-softmax skeleton, then add MXFP4 `P` quantization and prequantized `V`/PV

2026-05-01 continuation, fused CuTe DSL QK accumulator visibility:

- Wired the same gated QK accumulator debug store into `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`:
  - `run_mxfp4_fmha_d192_setup_smoke(..., store_qk_accumulator=True)`
  - deterministic raw input modes now match the standalone smoke for zero, full padded ones, and packed-row D192 ones
  - the fixed accumulator consumer is the softmax warp group, which matches the next planned online-softmax handoff
- Fused D192 numeric validation on `cuda:1`, shape `(B=1, Sq=128, Sk=128, H=12)`:
  - zero raw Q/K/scales: scores are exactly `0.0`, no NaNs
  - padded all-ones raw Q/K with scale byte `0x7f`: scores are exactly `256.0`, no NaNs
  - packed-row unpadded all-ones D192: scores are exactly `192.0`, no NaNs
- Rechecked issue-only timing after switching the accumulator consumer to the softmax warp group:
  - `S=128`, warmup `2`, iterations `10`: about `58.30 us`, grid `(1, 1, 12)`
  - `S=4096`, warmup `2`, iterations `10`: about `151.25 us`, grid `(32, 32, 12)`
  - this is not a regression versus the previous issue-only timings

2026-05-01 continuation, fused single-tile softmax/P debug path:

- Added a gated fused softmax/P consumer in `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`:
  - `run_mxfp4_fmha_d192_setup_smoke(..., store_qk_softmax=True)`
  - consumes the validated QK accumulator with the softmax warp group
  - computes single-tile unnormalized `P = exp2(score * scale - rowmax * scale)`
  - writes `P` back into tensor memory, then reuses the existing debug dump path
- Deterministic fused validation on `cuda:1`, shape `(B=1, Sq=128, Sk=128, H=12)`:
  - zero raw Q/K/scales: dumped `P` is exactly `1.0`, no NaNs
  - padded all-ones raw Q/K with scale byte `0x7f`: dumped `P` is exactly `1.0`, no NaNs
  - packed-row unpadded all-ones D192: dumped `P` is exactly `1.0`, no NaNs
- Added and validated one non-uniform deterministic probe:
  - Q is packed-row D192 ones
  - first half of K rows are packed-row D192 ones
  - second half of K rows are zero
  - accumulator dump range is exactly `0.0 .. 192.0`, no NaNs
  - softmax/P dump range is exactly `0.0 .. 1.0`, no NaNs
- Rechecked issue-only timing after adding the optional softmax/P path:
  - `S=128`, warmup `2`, iterations `10`: about `62.23 us`, grid `(1, 1, 12)`
  - `S=4096`, warmup `2`, iterations `10`: about `151.04 us`, grid `(32, 32, 12)`
- Current forward path status:
  - fused QK issue is live
  - fused accumulator debug dump is live
  - fused single-tile unnormalized softmax/P dump is live
  - still missing: MXFP4 `P` quantization, prequantized `V` staging, and blockscaled FP4 PV

2026-05-03 continuation, fused MXFP4 `P` quantization debug path:

- Added a gated MXFP4 `P` quantization debug mode in `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`:
  - `run_mxfp4_fmha_d192_setup_smoke(..., store_qk_mxfp4_quant=True)`
  - computes the same single-tile unnormalized `P` as the softmax debug path
  - computes per-32 `P` amax groups with `cute.logical_divide(tTMEM_LOADrS, make_layout(size/4))`
  - rounds each group amax to an E8M0-style power-of-two scale
  - the temporary E8M0 helper now covers a wider softmax-tail range via a compile-time threshold loop instead of only the first few powers of two
  - quantizes positive `P * 6 / scale` to positive E2M1 levels and dumps reconstructed `P`
- Added `store_qk_mxfp4_scale_debug=True` to dump the selected per-group scale directly.
- Added `store_qk_mxfp4_payload_debug=True` to dump the E2M1 payload value before applying the
  group scale back to reconstructed `P`.
- Important diagnostic:
  - the first implementation used `tTMEM_LOADrS.load().reduce(...)` over the whole fragment
  - that produced row-fragment scale `1.0 .. 1.0` for a mixed `P = 0.6 .. 1.0` row
  - the reconstructed quantized range was therefore `0.6666667 .. 1.0`
  - after switching to four 32-value fragments, scale debug reports `0.5 .. 1.0` and reconstructed quantized `P` reports `0.5 .. 1.0`
- Validated on `cuda:1`, shape `(B=1, Sq=128, Sk=128, H=12)`:
  - accumulator zero: `0.0 .. 0.0`, no NaNs
  - accumulator padded ones: `256.0 .. 256.0`, no NaNs
  - accumulator packed D192 ones: `192.0 .. 192.0`, no NaNs
  - softmax half-zero probe at `scale=1/192`: `0.5 .. 1.0`, no NaNs
  - MXFP4 quant half-zero probe at `scale=1/192`: `0.5 .. 1.0`, expected diff `0.0`
  - MXFP4 quant probe with true low probability `0.6`: reconstructed range `0.5 .. 1.0`, expected diff `0.0`
  - after widening the E8M0 helper, the `0.6` probe still passes and reports initialized `V` payload/scales
- Added a stronger mixed-within-block deterministic probe:
  - Q is packed-row D192 ones
  - alternating K rows are packed-row D192 ones / zero, so each 32-column MX group contains both high and low probabilities
  - with true low probability `0.6`, validation on `cuda:1` reports:
    - accumulator range `0.0 .. 192.0`, expected diff `0.0`
    - raw softmax/P range `0.6000000238 .. 1.0`
    - per-block scale debug range `1.0 .. 1.0`
    - E2M1 payload debug range `4.0 .. 6.0`
    - reconstructed MXFP4 P range `0.6666666865 .. 1.0`
  - this confirms the implementation is no longer doing per-value rescaling for `P`; mixed values in the same MX block share the group scale, matching the SageAttention-style contract
- Runner update:
  - deterministic raw modes now initialize `V` payload and `V` scales as prequantized MXFP4 too (`0x22` payload and `0x7f` scale for ones)
- PV-shaped proxy check:
  - direct reuse of the QK smoke with a shrunk `K=128` tile failed scale-TMA layout creation
  - keeping the blockscaled MMA at padded `K=256` but marking only active `K=128` passed:
    - `run_mxfp4_qk_mma_tile_smoke(..., geometry=qk_head_dim=128/qk_head_dim_padded=256, constant_ones_unpadded_packed=True)`
    - output range `128.0 .. 128.0`, expected diff `0.0`, no NaNs
- Current next step:
  - wire the Sage-style `P` payload plus per-32 `P` scales into the actual PV tmem-source path
  - do not spend time on V quantization first; the prequantized V representation is already validated enough for the next fused PV bring-up

2026-05-03 continuation, CuTe DSL tmem-source operand-A smoke:

- Extended `tk_fa4/cute_dsl_mxfp4_qk_mma_tile_smoke.py` with an `a_source_tmem=True`
  compile-time variant.
- Purpose:
  - validate the exact CuTe DSL mechanics needed for PV-style operand-A-from-TMEM before wiring the
    fused MXFP4 `P` payload into the real PV MMA
  - keep Q/K on the v5 one-CTA MXFP4 path; this is only a TMEM-source payload staging proxy
- Implementation notes:
  - `OperandSource.TMEM` must be selected as a kernel-object compile option, not passed as a dynamic
    launch argument
  - direct `Cp4x32x128bOp` S2T for the FP4 payload is the wrong primitive; it trips TMEM layout
    verification for this operand payload
  - the working path follows the Blackwell mixed-input pattern: smem -> registers -> TMEM via
    `tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE)`
  - that store pattern requires four participating transform warps; issuing it from the single MMA
    warp compiled and ran, but left part of the TMEM tile zero (`0.0 .. 128.0`)
  - the smoke now reuses the four epilogue warps as Q->TMEM transform warps, synchronizes them with
    the MMA warp through named barriers, fences the async TMEM store, then issues the blockscaled MMA
    from the TMEM A fragment
- Validation on `cuda:1`:
  - `py_compile` passes for `tk_fa4/cute_dsl_mxfp4_qk_mma_tile_smoke.py`
  - default SMEM-A D192 baseline still passes:
    - output range `192.0 .. 192.0`, expected diff `0.0`, no NaNs
  - TMEM-A D192 variant passes:
    - `a_source_tmem=True`
    - output range `192.0 .. 192.0`, expected diff `0.0`, no NaNs
  - TMEM-A PV-shaped active128/padded256 variant passes:
    - `geometry=qk_head_dim=128/qk_head_dim_padded=256`, `a_source_tmem=True`
    - output range `128.0 .. 128.0`, expected diff `0.0`, no NaNs
- Updated next step:
  - use this validated TMEM-source A pattern for the fused PV side
  - replace the proxy Q payload with the fused Sage-style MXFP4 `P` payload plus per-32 `P` scales
  - keep the four-warp TMEM store/fence/barrier discipline when staging payload into TMEM

2026-05-05 continuation, first fused MXFP4 `P -> PV` accumulator seam:

- Added a gated fused PV debug mode in `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`:
  - `run_mxfp4_fmha_d192_setup_smoke(..., store_mxfp4_pv_accumulator=True)`
  - keeps the existing QK producer path and accumulator pipeline
  - softmax warps consume the QK accumulator, compute the single-tile softmax `P`, multiply by `6`,
    convert the vector to `Float4E2M1FN`, and store the payload into TMEM
  - the MMA warp waits on the same four-warp TMEM-store/fence/barrier discipline validated by the
    standalone tmem-source operand-A smoke
  - the MMA warp then runs a one-CTA blockscaled PV MMA with operand A sourced from TMEM and dumps
    the PV accumulator through the existing debug GMEM path
- Scope of this seam:
  - this validates fused `QK -> softmax -> FP4 P payload in TMEM -> PV MMA -> accumulator dump`
  - the payload convention is Sage-style `P * 6`; with unit scales the debug accumulator is therefore
    `6 * sum(P * V)` before any final `1/6`/softmax normalization correction
  - separate `V` scale TMA layout is not wired yet; for this seam the PV SFB pointer reuses the
    already-staged all-ones K-scale TMEM path so the runnable test stays focused on P payload staging
- Validation on `cuda:1`, shape `(B=1, Sq=128, Sk=128, H=12)`:
  - `py_compile` passes for `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py`
  - existing QK accumulator baseline still passes:
    - packed D192 ones output `192.0 .. 192.0`, expected diff `0.0`, no NaNs
  - fused PV accumulator, packed D192 Q/K ones and prequantized V ones:
    - output `768.0 .. 768.0`, expected diff `0.0`, no NaNs
  - fused PV accumulator, half-zero K probe with `scale_softmax_log2=1/192`:
    - output `576.0 .. 576.0`, expected diff `0.0`, no NaNs
  - fused PV accumulator, alternating K probe with `scale_softmax_log2=1/192`:
    - output `576.0 .. 576.0`, expected diff `0.0`, no NaNs
- Short warmed timing screen on `cuda:1`, `warmup=2`, `iterations=10`, `S=128`:
  - QK accumulator debug seam: about `6.71 us`
  - fused PV accumulator debug seam: about `8.74 us`
- Current next step:
  - replace the temporary unit-scale reuse with the real `P` scale / `V` scale TMEM path
  - then add the final output correction (`1/6` payload scale and softmax row-sum normalization)
  - after that, compare against BF16 / FA4 / SageAttention3 on the same warmed benchmark harness

2026-05-05 continuation, rejected separate P-scale TMEM/S2T path:

- Tried the minimal plumbing step of allocating a distinct PV `P` scale TMEM window after the Q/K
  scale windows, copying the existing unit Q-scale SMEM tile into it, and pointing PV SFA at that
  new pointer.
- Correctness stayed fine on the existing deterministic probes:
  - QK packed D192 ones still matched `192.0`
  - fused PV packed D192 ones still matched `768.0`
  - fused PV half-zero and alternating probes at `scale_softmax_log2=1/192` still matched `576.0`
- Rejected the route for performance/codegen:
  - adding the extra S2T scale descriptor/copy path pushed the warmed debug timing screen to roughly
    `63 us` for both QK and PV variants on `cuda:1`
  - moving the copy under the PV branch did not recover the timing because the nested kernel flag is
    dynamic to CuTe DSL codegen, so the extra descriptor path still pollutes the generated kernel
  - patch was reverted; do not reintroduce a separate per-tile P-scale S2T path unless it is split
    into a genuinely separate compile-time kernel variant
- Low-probability correctness target for the next attempt:
  - QK debug MXFP4 quantization already preserves a half-zero low lane with
    `scale_softmax_log2=5/192`: output range `0.03125 .. 1.0`, expected diff `0.0`
  - current fused PV unit-scale payload path drops that low lane: output `384.0`, expected under the
    current unit-scale debug convention `384.0`
  - a real generated-P-scale MXFP4 PV path should preserve the low lane and accumulate `396.0`
    before final `1/6` / row-sum normalization (`64*6 + 64*(6*2^-5)`)
- Updated direction:
  - follow SageAttention3's shape more closely for `P`: compute per-32 `AbsMaxP`, pack payload and
    SFP together/register-local for the PV MMA if possible
  - avoid adding another TMEM scale-factor S2T stream to the hot path
  - keep the existing fused `P` payload TMEM seam as the rollback baseline until a faster generated
    SFP path compiles and passes the `396.0` low-probability PV probe

2026-05-05 continuation, static debug-mode specialization hook:

- Updated `tk_fa4/cute_dsl_mxfp4_forward_kernel_d192.py` so the debug store mode is also stored on
  the kernel object at construction time:
  - `store_qk_accumulator_static`
  - `store_qk_softmax_static`
  - `store_qk_mxfp4_quant_static`
  - `store_qk_mxfp4_scale_debug_static`
  - `store_qk_mxfp4_payload_debug_static`
  - `store_mxfp4_pv_accumulator_static`
- The producer kernel now reads those static object fields before its mode branches, allowing
  `cutlass.const_expr(...)` on the PV/QK debug branch choices. Passing the booleans as nested kernel
  arguments was still dynamic to CuTe DSL codegen and rejected by `const_expr`.
- Validation on `cuda:1` after the refactor:
  - QK packed D192 ones: `192.0 .. 192.0`, expected diff `0.0`
  - fused PV packed D192 ones: `768.0 .. 768.0`, expected diff `0.0`
  - low-probability QK MXFP4 quant probe at `scale_softmax_log2=5/192`: `0.03125 .. 1.0`, expected
    diff `0.0`
  - `py_compile` passes
- Timing note:
  - warmed `iterations=1000` screen still reports about `62-64 us` for the QK/PV debug variants in
    this process, so this refactor is a correctness/codegen hook, not a measured speed win
  - earlier recorded `~6.71 us` / `~8.74 us` short screen should be rechecked in a clean timing
    harness before drawing performance conclusions

2026-05-05 continuation, generated P-scale CuTe DSL attempt rejected:

- Tried to extend `qk_softmax_p_to_tmem_payload(...)` from payload-only `P*6` into true generated
  MXFP4 P quantization:
  - compute four per-row/per-32-column `p_amax` groups from the softmax fragment
  - compute E8M0-style `p_scale`
  - divide the FP4 payload by `p_scale`
  - write the generated scales into SMEM for the PV SFA S2T path
- First blocker:
  - scalar `p_scale.to(Float8E8M0FNU)` is illegal in CuTe DSL, with verifier error
    `nvgpu.cvt_fptrunc operand #0 must be 32-bits aligned floating-point-like 1-d vector`
  - switching to a 4-element vector conversion compiles
- Second blocker:
  - writing into the QK-shaped scale SMEM layout with a naive `row*8 + group` index corrupts PV
    scale interpretation: the `scale_softmax_log2=5/192` half-zero PV probe produced row-dependent
    `24.0 .. 768.0` instead of the target `396.0`
  - using the TK swizzle index `(row % 32) * 16 + (row / 32) * 4 + group` still produced incorrect
    rows (`24.0 .. 768.0` at `5/192`, `384.0 .. 768.0` at `1/192`), confirming that the QK SFA
    layout is the wrong source for PV-generated P scales
- Third blocker:
  - adding a PV-shaped `p_scale_smem_layout_staged` and PV-shaped `tCtScaleP_layout` compiled but
    hit a CUDA illegal instruction at runtime during the first PV smoke
  - this patch was reverted immediately
- Restored validated state after revert:
  - QK packed D192 ones: `192.0 .. 192.0`, expected diff `0.0`
  - fused PV packed D192 ones: `768.0 .. 768.0`, expected diff `0.0`
  - fused PV half-zero at `scale_softmax_log2=1/192`: `576.0 .. 576.0`, expected diff `0.0`
- Updated direction:
  - do not continue trying ad-hoc SMEM writes into CuTe DSL SFA layouts
  - either find a first-class CuTe DSL way to form register-local blockscaled PV operands, or move
    generated P SFP packing into a lower-level C++/CuTe path modeled directly on SageAttention3
  - keep the low-probability target (`396.0` before final normalization) as the acceptance test

2026-05-05 continuation, CuTe DSL backward PTX/SASS dump:

- Dumped the vendored CuTe DSL SM100 backward artifacts for shape
  `(B=1, S=2048, H=16, Dqk=192, Dv=128, causal)` using:
  - `CUTE_DSL_KEEP_PTX=1`
  - `CUTE_DSL_KEEP_CUBIN=1`
  - `CUTE_DSL_ARCH=sm_100a`
  - `CUDA_VISIBLE_DEVICES=3`
  - `.venv-cute/bin/python` with the vendored `flash-attention` path and the older `.venv`
    site-packages on `PYTHONPATH` for `quack`
- Output directory:
  - `tk_fa4/results/cute_dsl_bwd_dump_20260505T162123Z/`
- Main backward artifacts:
  - `*flash_bwd_sm100*BackwardSm100*.sm_100a.cubin`
  - `*flash_bwd_sm100*BackwardSm100*.sm_100a.ptx`
  - `*flash_bwd_sm100*BackwardSm100*.sm_100a.sass`
  - `cute_bwd_main_sass_summary.txt`
- CuTe main backward resource usage:
  - `REG:128`
  - `STACK:0`
  - `LOCAL:0`
  - static `SHARED:1024` plus dynamic launch shared memory
- CuTe main backward SASS signature:
  - 2-CTA UMMA/TMEM path is explicit: `UTCHMMA.2CTA`, `UTMALDG.4D.2CTA`, `UTCBAR.2CTA.MULTICAST`,
    `SYNCS`, `LDTM`, `STTM`
  - TMA/cluster movement is explicit: `UBLKCP.S.G`, `UBLKRED`, `UCGABAR_*`, `ATOMS`
  - top counted instructions include `UTCHMMA 44`, `UTMALDG 26`, `SYNCS 108`, `MUFU 64`,
    `F2FP 144`
- Dumped current TK extension side-by-side:
  - `tk_fa4/results/tk_current_sass_20260505T162225Z/`
  - `resources.txt`
  - `tk_all.sass`
  - `tk_sass_summary.txt`
- Current TK active split route resource signature:
  - DQ clustered main: `REG:168`, `STACK:16`, `SHARED:190532`
  - DK/DV only: `REG:255`, `STACK:552`, `SHARED:85016`
  - first-block patch: `REG:255`, `STACK:0`, `SHARED:91160`
  - chunked DQ reduce: `REG:72`, `STACK:0`, `SHARED:0`
- Reverse-engineering conclusion:
  - the SASS confirms the earlier code-level diagnosis: CuTe is one low-register 2-CTA
    TMEM/UMMA pipeline, while TK is still multiple high-register kernels with SIMT/HMMA-heavy
    recompute and global DQ add/reduce traffic
  - next useful comparison is not the whole SASS blob; diff CuTe skip variants
    (`FLASH_ATTN_FP4_BWD_NATIVE_SKIP_DQ`, `...SKIP_DK`, `...SKIP_DKV_EPILOGUE`,
    `...SKIP_DQ_REDUCE`) and use the SASS deltas to identify the exact DQ-reduce and
    dS-consumption instruction slices to port

2026-05-06 continuation, FP4 PV production timing and fixed P-scale screen:

- Rebuilt and measured the current production/experiments path on `cuda:1` / GB200 (`sm_100`),
  using `S=4096`, `B=1`, `H=12`, `Dqk=192`, `Dv=128`, `qk_quant_backend=v5`,
  `v_quant_backend=localcta`, `warmup=2`, `iters=5`.
- Canonical BF16/FP4 comparison after rebuild:
  - BF16 persistent baseline: about `0.1016 ms`
  - QK-FP4 / V-BF16 persistent: about `0.0852 ms`
  - full FP4 production fullgrid: about `0.2166 ms`
  - conclusion: QK-FP4 is not the current bottleneck; the extra cost is in FP4 PV integration
    (`~0.131 ms` over QK-FP4 / V-BF16), mostly P quantization plus P/V scale handoff.
- Accepted small production change in `b300_causal/bf16_b300_mha_causal_fp4.cu`:
  - added `FP4PV_USE_FIXED_P_TILE_SCALE = true`
  - in the direct online P quantizer, uses fixed P tile amax `1.0f` and skips the CTA-wide P amax
    reduction when `FP4PV_DIAG_USE_DIRECT_ROW_UPDATE` is enabled
  - rationale: online softmax probabilities are in `[0,1]`; the per-group block scale still tracks
    local group amax, while the expensive tile-wide amax is not needed for the normal random/zero
    probes
- Validation with direct production launch-mode benchmark:
  - random live FP4, `S=4096,H=12`: persistent/fullgrid about `0.216 ms`, max abs diff vs stored-P
    oracle about `0.0684`, mean abs diff about `8.8e-4`
  - zero-QK random-V, `S=4096,H=12`: persistent/fullgrid about `0.216 ms`, max abs diff about
    `0.1729`, mean abs diff about `2.9e-3`
  - production and experiments extensions both rebuild with the fixed-scale source
- Rejected branch:
  - changing FP4PV dispatch from `config_fp4pv<...,2>` to `config_fp4pv<...,1>` does not compile
  - ptxas rejects the kernel because the function mixes single-CTA (`.cta_group::1`) and CTA-pair
    (`.cta_group::2`) granularities
  - do not retry cluster-size 1 without first splitting the QK/PV issue helpers so a kernel instance
    uses one CTA-group granularity consistently
- Remaining direction:
  - FP4 PV is still about `2.13x` slower than BF16 full attention at this shape despite QK-FP4 /
    V-BF16 being faster than BF16
  - the next real win needs a SageAttention3-style register-scale PV path or a lower-overhead
    P-scale handoff; additional standalone PV-only tuning is unlikely to close the `~0.13 ms`
    integrated gap by itself

2026-05-06 continuation, fixed P-scale fast path and Sage D192 check:

- SageAttention3 import/API check:
  - `PYTHONPATH=/workspace/codebases/fp4_matmul/SageAttention/sageattention3_blackwell` imports as
    `sageattn3`, not `sageattention`
  - the public `sageattn3_blackwell` path cannot be used as a direct D192 comparison here:
    `per_block_mean=True` fails in Triton because `tl.arange(0, D)` requires power-of-two `D`, and
    `per_block_mean=False` then hits the upstream quantizer guard `Unsupported head dim: 192`
  - the in-repo SageAttention3 quantizer dispatch supports only `head_dim == 64` and `128`, so a
    fair current FA4 comparison would require adding/separately building a D128 FA4 path or changing
    the experiment shape
- Accepted production P-pack fast path in `b300_causal/bf16_b300_mha_causal_fp4.cu`:
  - with `FP4PV_USE_FIXED_P_TILE_SCALE=true`, `fp4pv_pack_scores_to_stage_and_scales(...)` now uses
    a direct per-16 group quantizer:
    - payload coefficient `6 / group_amax`
    - decode scale `group_amax / 6`
  - this removes the generic localCTA two-level scale formula from the hot online `P` pack path for
    the fixed-scale direct-row-update configuration, while keeping the generic path intact behind
    `!FP4PV_USE_FIXED_P_TILE_SCALE`
  - the source also avoids computing `s_enc` / `sg_val` when the fixed-scale path is compiled in
- Build validation:
  - `make -C tk_fa4/b300_causal -j2` succeeds
  - `ptxas` still reports the same FP4PV kernel spill shape as before: `16` spill-store bytes and
    `20` spill-load bytes for `kernel_fp4pv<config_fp4pv<128,128,192,128,2>>`
- Timing/correctness screen on `cuda:1`, GB200, `S=4096,B=1,H=12,Dqk=192,Dv=128`, warmup `2`,
  iters `5`:
  - random live FP4 production launch modes:
    - persistent `0.1976 ms`
    - fullgrid `0.2011 ms`
    - stored-P oracle diff max/mean `0.0549` / `0.000849`
  - zero-QK random-V production launch modes:
    - persistent `0.2006 ms`
    - fullgrid `0.1999 ms`
    - stored-P oracle diff max/mean `0.1729` / `0.002824`
  - canonical comparison in the same rebuilt process had production fullgrid `0.2068 ms` versus
    older fused fullgrid `0.2149 ms`; BF16 was noisy in that run (`0.1467 ms`), so use the
    production launch-mode deltas as the cleaner signal for this micro-change
  - post-QK decomposition in the same rebuilt process had production fullgrid `0.2003 ms` versus
    regular fused fullgrid `0.2026 ms`
- Rejected micro-edit:
  - replacing the fixed-scale helper's BF16 staging plus `mul_cvt_bf16_to_fp4_8x_round_to_nearest`
    with direct `f32 -> e2m1` packing compiled, but did not give a clear timing win
  - screen result after the edit was random live FP4 persistent/fullgrid `0.2095` / `0.2030 ms`,
    zero-QK `0.2018` / `0.2019 ms`, with a slight oracle-diff change on random
  - reverted immediately; keep the direct-scale fast path but continue using the BF16 converter
- Rejected micro-edit:
  - replacing scalar `fabsf/fmaxf` group amax in the fixed-scale helper with the BF16 pairwise
    `abs_max_2x` path compiled, but did not give a clean win
  - screen result was random persistent/fullgrid `0.2023` / `0.1986 ms`, zero-QK `0.2028` /
    `0.2028 ms`; reverted because persistent and zero-QK regressed
- Rejected structural skip:
  - tried bypassing `fp4pv_pack_scores_to_stage_and_scales(...)` for fully future-masked causal
    tiles (`idx > m_tile`) and publishing direct zero payload/scale instead
  - correctness stayed unchanged, but timing regressed badly: random persistent/fullgrid `0.2036` /
    `0.2057 ms`, zero-QK `0.2328` / `0.2241 ms`
  - reverted; the extra branch/control flow is worse than quantize-then-zero in this pipeline
- Accepted production P-pack reciprocal change:
  - in the fixed-scale helper, replaced precise `6.0f / block_amax` with inline
    `rcp.approx.ftz.f32(block_amax) * 6.0f`
  - build still succeeds with the same FP4PV spill footprint (`16` spill-store bytes, `20`
    spill-load bytes)
  - confirmation screen on `cuda:1`, GB200, `S=4096,B=1,H=12`, warmup `3`, iters `7`:
    - random live FP4 persistent/fullgrid `0.1644` / `0.1640 ms`
    - zero-QK random-V persistent/fullgrid `0.1644` / `0.1643 ms`
    - stored-P oracle diffs unchanged: random max/mean `0.0549` / `0.000849`, zero-QK max/mean
      `0.1729` / `0.002824`
  - canonical comparison after this change:
    - BF16 persistent baseline `0.1020 ms`
    - QK-FP4 / V-BF16 persistent `0.0834 ms`
    - production full FP4 fullgrid `0.1654 ms`
    - older fused fullgrid `0.2121 ms`, MM2 fullgrid `0.1919 ms`
    - production full FP4 remains slower than BF16 (`~1.62x`) but is now substantially faster than
      the previous production `~0.20-0.216 ms` range
- Accepted follow-up guard cleanup:
  - removed the `isfinite(block_amax)` and post-reciprocal `coeff` finite checks from the fixed-scale
    helper; `!(block_amax > 0)` still catches zero and NaN groups
  - confirmation screen on `cuda:1`, GB200, `S=4096,B=1,H=12`, warmup `3`, iters `7`:
    - random live FP4 persistent/fullgrid `0.1608` / `0.1607 ms`
    - zero-QK random-V persistent/fullgrid `0.1606` / `0.1608 ms`
    - stored-P oracle diffs unchanged: random max/mean `0.0549` / `0.000849`, zero-QK max/mean
      `0.1729` / `0.002824`
  - canonical comparison after this cleanup:
    - BF16 persistent baseline `0.0992 ms`
    - QK-FP4 / V-BF16 persistent `0.0830 ms`
    - production full FP4 fullgrid `0.1605 ms`
    - production full FP4 is still `~1.62x` BF16 and `~1.93x` QK-FP4/V-BF16, so the remaining gap is
      still PV-side integration rather than QK
- Updated direction:
  - this is a useful local win, but it does not change the architectural bottleneck: the remaining
    gap is still the integrated FP4 PV scale/payload handoff and TCGEN05 blockscale path
  - do not spend more time trying to benchmark upstream SageAttention3 at D192 unless its quantizer
    is extended; use SageAttention3 for code-shape guidance, not as a direct D192 timing baseline
- SageAttention3-style PV conclusion:
  - direct SageAttention3 D192 support is not a quick route; a local proof patch that added D192
    dispatch compiled far enough to instantiate the kernel but failed in its traits/layouts
  - representative failures were `epilogue_tma_ws.h` static assertion
    `NumMmaThreads must be a multiple of kGmemThreadsPerRow` and multiple CuTe layout/copy shape
    divisibility assertions
  - root cause is that SageAttention3 ties Q/K/V/O to one `kHeadDim` trait, while the FA4 target is
    split-dim (`Dqk=192`, `Dv=128`); treating Sage as a whole-kernel D192 baseline is therefore the
    wrong integration layer
  - the initial Sage-style hypothesis was to port only the PV mechanism:
    - keep the existing FA4/TK QK path
    - build FP4 `P` payload and UE4M3 `SFP` scale in registers from the online softmax fragments
    - use the Sage register-scale warp MMA atom (`mma.sync.aligned.kind::mxf4nvf4...`) for PV
      instead of staging generated P scales through shared/TMEM for TCGEN05
    - carry Sage's scaled-unnormalized-exp recurrence and divide by row sum in the final epilogue
  - the instruction-level smoke below rejects this exact register-MMA plan for sm100a; keep only the
    algorithmic pieces as possible follow-up work
- SageAttention3 register-MMA hard constraint on sm100a:
  - tried a one-warp smoke inside `b300_causal/bf16_b300_mha_causal_fp4.cu` using the Sage-style
    warp instruction
    `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X...f32.e2m1.e2m1.f32.ue4m3`
  - nvcc reached ptxas, but ptxas rejected the instruction for the actual target:
    `Instruction 'mma with block scale' not supported on .target 'sm_100a'`
  - removed the smoke so the production extension can build again
  - this means exact SageAttention3 register-scale PV is an sm120-style path, not the GB200/B200
    sm100a path we are compiling here
  - on sm100a, the available MXFP4/NVFP4 blockscaled path is TCGEN05 UMMA; both CUTLASS and
    ThunderKittens encode scale operands as addresses (`[%sfa]`, `[%sfb]`), so generated P scales
    still need a descriptor-visible backing location
  - next optimization should therefore be Sage-inspired at the algorithm level
    (scaled unnormalized exp, row-sum final normalization, leaner generated P scale layout), not a
    direct SageAttention3 register-scale MMA port

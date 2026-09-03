# MXFP4 Forward Overlap Loop Ledger - 2026-06-22

Scope: session 6, MXFP4 forward only. Primary device was physical GPU2 via `CUDA_VISIBLE_DEVICES=2` with PyTorch seeing `cuda:0` (`NVIDIA GB200`). All kernel timing claims below use direct preallocated extension calls, not wrapper allocation timing.

## Code / Config Changes

- Added compile-gated MXFP4 forward timeline instrumentation in `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`.
  - Gate: `TK_FA4_ENABLE_MXFP4_FWD_TIMELINE`, default `0`.
  - Event IDs cover QK wait/issue/commit, P wait/pack ready, V wait/scale ready, PV wait/issue/commit/done, tile wait, and epilogue start/end.
  - Default debug sampling is one CTA and one task; default build expands the record macro to a no-op.
- Added host reset/read helpers in `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.
- Exposed timeline helpers in `tk_fa4/fp4_fa4_fwd/fwd_pybind.inc`.
- Added `MXFP4_FWD_TIMELINE ?= 0` to `tk_fa4/fp4_fa4_fwd/Makefile`.
- Added timeline capture/parser driver:
  `results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_driver.py`.
- Tried changing the Python h4/s2048 auto selector to the best sweep candidate, but alternating timings rejected it; the selector change was reverted.

## Builds

- Timeline build: `make -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=1` succeeded.
- Final production/default build: `make -C tk_fa4/fp4_fa4_fwd forward` succeeded with `TK_FA4_ENABLE_MXFP4_FWD_TIMELINE=0`.

## Baseline Timing

Artifact: `forward_overlap_loop_20260622_baseline_prealloc_gpu2.json`

- Config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`
- h4/s2048 direct preallocated median: `0.061792 ms`, min `0.059584 ms`
- BF16 median: `0.075648 ms`
- Finite: true; max abs diff vs BF16 `0.8125`, LSE max abs diff `0.0294919461`

Final default-build auto check: `forward_overlap_loop_20260622_final_auto_default_gpu2.json`

- Auto still resolves to the same current config.
- h4/s2048 direct preallocated median: `0.060512 ms`, min `0.057376 ms`
- BF16 median: `0.073984 ms`
- Finite: true; max abs diff vs BF16 `0.8125`, LSE max abs diff `0.0294919461`

## Timeline Evidence

Stable subset artifact: `forward_overlap_loop_20260622_timeline_gpu2_lanefiltered_oneiter.json`

- Timeline-enabled debug build, h4/s2048, one iteration.
- Instrumented runtime is intentionally not a performance number: `253.982 ms`.
- Records decoded: `132`.
- Event counts:
  - `qk_wait_begin/done/issue/commit`: `15` each
  - `pv_wait_begin/done`: `32` each
  - `epilogue_start/end`: `2` each
  - `tile_wait_begin/done`: `2` each
- Median phase deltas:
  - QK wait: `958 cycles`
  - QK issue to commit: `879 cycles`
  - Epilogue: `3769 cycles`
- CSV inspection showed QK issue records on scheduler thread `256` and output/PV wait records on threads `0` and `32`.

Rejected timeline hook attempts:

- Full per-warp sampling timed out after the 5 s CUDA event limit and the outer process exited with code `124`.
- Adding the active PV issue lane (`threadIdx.x == 288`, inferred from QK scheduler thread `256`) also timed out after the 5 s CUDA event limit and the outer process exited with code `124`.
- Conclusion: the gated event points exist in code, but recording directly inside the active PV warpgroup issue lane perturbs the TC issue path enough to be unsafe. The stable validated timeline covers QK/output ordering; full P-pack/PV-issue live timestamps remain blocked.

## Candidate Timing

Candidate family tested without adding shared/TMEM pressure:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix`

Artifacts:

- `forward_overlap_loop_20260622_candidate_sweep_gpu2.json`
- `forward_overlap_loop_20260622_candidate_sweep2_gpu2.json`
- `forward_overlap_loop_20260622_confirm_best_gpu2.json`
- `forward_overlap_loop_20260622_auto_select_verify_gpu2.json`
- `forward_overlap_loop_20260622_interleaved_verify_gpu2.json`
- `forward_overlap_loop_20260622_alternating_old_vs_candidate_gpu2.json`

Grouped sweeps looked promising:

- Sweep2 best median: `0.059040 ms`, finite true.
- Paired confirm old first / candidate second: old `0.060352 ms`, candidate `0.058880 ms`, finite true.

But alternating per-sample timing rejected it as a default change:

- Old median: `0.063440 ms`
- Candidate median: `0.063584 ms`
- Median speedup candidate/old: `0.9977x`
- Old mean: `0.064384 ms`
- Candidate mean: `0.064151 ms`
- Mean speedup candidate/old: `1.0036x`
- Both finite with the same BF16 comparison envelope.

Conclusion: apparent grouped speedups were mostly order/clock-state effects. The selector change was reverted.

## Plan Item Status

1. Timeline: partially landed.
   - Compile-gated timeline support, parser, and host API landed with default performance unaffected.
   - Stable QK/output timeline evidence captured.
   - Active PV issue-lane capture was rejected due debug-induced timeout, so full P/PV phase timing remains blocked.

2. Producer/consumer overlap at P->PV handoff: not landed.
   - Existing config already has decoupled PV issue and `P_STAGE_SLOTS=2`.
   - New overlap change was not landed because the required active P/PV lane timeline could not be safely captured, and the nearby overlap/persist-reuse config candidate was not robust under alternating timing.

3. Double-buffer only P payload/scales/descriptors: not landed.
   - No new buffering was added.
   - Existing tested candidates use existing config-level P-stage/reuse mechanisms; no robust direct-preallocated win was proven.

4. Raise UTCOMMA issue density before softmax rework: candidate rejected.
   - Tested fixed existing issue-density/reuse configs without adding TMEM/shared pressure.
   - Best grouped candidate did not survive alternating old-vs-candidate timing, so no default/config switch landed.

## Final Conclusion

Concrete artifact landed: compile-gated MXFP4 forward timeline infrastructure and parser artifacts under `forward_overlap_loop_20260622_*`.

Optimization candidate conclusion: no production config/code performance change landed. The fastest-looking candidate was rejected after alternating direct-preallocated timing showed no robust median improvement. The next useful step is a lower-perturbation PV-lane trace mechanism, likely per-buffer shared-memory stamps written by the owning warp without global atomics near UTCOMMA issue, or SASS/PC-sampling correlation around the active PV scheduler lane.

## Timeline-Only Checkpoint: 2026-06-22

Latest user constraint: stop before overlap/synchronization candidates and validate the timeline-only patch first.

Formatting/brace audit:

- Cleaned indentation drift in `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc` output paths around timeline hooks, output drain branches, final epilogue/LSE store, and persistent-output task loop.
- No overlap, barrier, ownership, or double-buffering logic was changed in this checkpoint.

Default build verification:

- Command: `timeout 900s make -C tk_fa4/fp4_fa4_fwd forward`
- Result: success with `TK_FA4_ENABLE_MXFP4_FWD_TIMELINE=0`.
- Direct preallocated GPU2 artifact: `forward_overlap_loop_20260622_timeline_only_default_gpu2.json`
- h4/s2048 auto effective config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`
- MXFP4 median/min: `0.060000 ms` / `0.058208 ms`
- BF16 median: `0.073824 ms`
- finite: `true`

Timeline build/readout:

- Command: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=1`
- Result: success with `-DTK_FA4_ENABLE_MXFP4_FWD_TIMELINE=1`.
- Trace command: `timeout 120s env CUDA_VISIBLE_DEVICES=2 python3 results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_driver.py --device cuda:0 --warmup 0 --iters 1 --output-prefix results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_only_gpu2`
- Trace artifact: `forward_overlap_loop_20260622_timeline_only_gpu2.json`
- Events CSV: `forward_overlap_loop_20260622_timeline_only_gpu2_events.csv`
- Summary JSON: `forward_overlap_loop_20260622_timeline_only_gpu2_summary.json`
- Decoded records: `132`
- Event counts: QK wait begin/done/issue/commit `15` each; PV wait begin/done `32` each; epilogue start/end `2` each; tile wait begin/done `2` each.
- Median deltas: QK wait `987 cycles`; QK issue-to-commit `745 cycles`; epilogue `3776 cycles`.
- CSV sanity: QK records on thread `256`; PV/output waits on threads `0` and `32`; output finite.

Checkpoint summary artifact:

- `forward_overlap_loop_20260622_timeline_only_checkpoint.md`

Status:

- Item 1 timeline-only: landed and validated for h4/s2048 GPU2.
- Items 2-4: intentionally not attempted in this checkpoint per user instruction.
- Final active shared object restored to default with `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0`; `read_mxfp4_forward_timeline()` returns length `0` in the restored default build.

## Fixed-Slot Timeline Pivot: 2026-06-22

Latest user constraint: replace/gate the atomic device-symbol timeline before overlap work. The validated trace now uses deterministic fixed slots and no `atomicAdd` in the timeline hot path.

Code/config changed:

- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - Removed the timeline head/atomic reservation design.
  - Added a fixed-size timeline array indexed by `(event_id, normalized_idx, lane_slot)`.
  - Default sample gate is `TK_FA4_MXFP4_FWD_TIMELINE_SAMPLE_BLOCKS=1` and `TK_FA4_MXFP4_FWD_TIMELINE_SAMPLE_TASKS=1`, so the h4/s2048 trace samples only `blockIdx.x == 0` and `task_num == 0`.
  - Event stores are plain stores from elected lanes. Event word is written last, and host compaction treats nonzero event word as occupied.
  - Lane ownership used for the successful smoke: QK/P/V producer events on thread `256`; PV issue/commit on thread `288`; PV/output waits and epilogue on thread `0`.
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
  - `reset_mxfp4_forward_timeline()` now clears only the fixed-slot array.
  - `read_mxfp4_forward_timeline()` copies the full fixed-slot array and compacts occupied records for the Python decoder.

Source guard check:

- `grep -R "fp4pv_forward_timeline_head\|atomicAdd(&fp4pv_forward_timeline" tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- Result: no matches.
- Existing unrelated `atomicAdd` users remain in other diagnostics, but not in the forward timeline record path.

Build gates:

- Default build before trace: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0`
  - Result: success.
  - Import/readout probe: `read_mxfp4_forward_timeline()` returned length `0`.
- Timeline build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=1`
  - Result: success.
- Restored default build after trace: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0`
  - Result: success.
  - Import/readout probe: `read_mxfp4_forward_timeline()` returned length `0`.

Default direct-preallocated baseline on physical GPU2:

- Artifact: `forward_overlap_loop_20260622_default_postfixedslot_gpu2.json`
- Command: `timeout 120s env CUDA_VISIBLE_DEVICES=2 python3 results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_driver.py --device cuda:0 --warmup 20 --iters 100 --output-prefix results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_default_postfixedslot_gpu2`
- Device: `NVIDIA GB200`, visible as `cuda:0` from physical GPU2.
- MXFP4 median/min: `0.059552 ms` / `0.057216 ms`.
- Finite: `true`.
- Timeline readout in default build: `0` records.

Fixed-slot timeline smoke on physical GPU2:

- Artifact: `forward_overlap_loop_20260622_timeline_fixedslot_gpu2.json`
- Events CSV: `forward_overlap_loop_20260622_timeline_fixedslot_gpu2_events.csv`
- Summary JSON: `forward_overlap_loop_20260622_timeline_fixedslot_gpu2_summary.json`
- Command: `timeout 120s env CUDA_VISIBLE_DEVICES=2 python3 results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_driver.py --device cuda:0 --warmup 0 --iters 1 --output-prefix results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_fixedslot_gpu2`
- Result: completed without timeout.
- Decoded records: `144`.
- Finite: `true`.
- Instrumented runtime: `238.119 ms`; not a performance number.
- Event counts:
  - `qk_wait_begin/done/issue/commit`: `15` each
  - `pv_wait_begin`: `32`
  - `pv_issue/commit/wait_done`: `16` each
  - `epilogue_start/end`: `1` each
  - `tile_wait_begin/done`: `1` each
- Median phase deltas:
  - QK wait: `631 cycles`
  - QK issue to commit: `295 cycles`
  - PV issue to commit: `193 cycles`
  - PV commit to output wait done: `261.5 cycles`
  - Epilogue: `2108 cycles`
- CSV sanity:
  - All decoded rows have `block_idx=0` and `task_num=0`.
  - QK records come from thread `256`.
  - PV issue/commit records come from thread `288`.
  - PV/output wait and epilogue records come from thread `0`.
  - QK and PV records interleave in timestamp order, so the trace is sufficient to inspect phase ordering/gaps for h4/s2048.

Status after fixed-slot pivot:

- Item 1 timeline-only: landed for this checkpoint as a compile-gated non-atomic fixed-slot trace with h4/s2048 GPU2 evidence.
- Items 2-4: not attempted in this checkpoint per user instruction. No synchronization, overlap, P-buffering, descriptor, or issue-density candidate was started after this pivot.
- Earlier atomic timeline artifacts remain historical only; the fixed-slot artifact supersedes them for future overlap decisions.
- Active shared object at checkpoint end is restored to the default `MXFP4_FWD_TIMELINE=0` build.

## Gap Analysis From Fixed-Slot Trace: 2026-06-22

Artifact:

- `forward_overlap_loop_20260622_gap_analysis.md`

Trace inputs:

- `forward_overlap_loop_20260622_timeline_fixedslot_gpu2_events.csv`
- `forward_overlap_loop_20260622_timeline_fixedslot_gpu2_summary.json`

Findings:

- QK pre-issue waits dominate the fixed-slot trace: long waits at `idx=3,4,6,8,10,12,14` sum to `24,036 cycles`, while the short QK waits sum to `3,509 cycles`.
- QK `issue -> commit` is not the exposed bottleneck in this trace; most QK issues commit in about `279-303 cycles`.
- PV owner `wait_begin -> issue` is short (`104-155 cycles`) and PV `issue -> commit` is also short (`158-209 cycles`).
- Output waits begin early on thread `0` and finish `222-286 cycles` after PV commit; output/slot-release cadence then creates `1.6k-2.4k cycle` gaps before the next output-side PV wait.
- Epilogue is visible but secondary: `2,108 cycles` from epilogue start to end, with a `245-cycle` tile wait inside.

Candidate direction anchored to this trace:

- Candidate A should target reuse/wait ownership around producer/consumer handoff without adding new full-CTA barriers.
- Candidate B should target scheduling/issue cadence or pulling independent QK/PV work forward; optimizing UTCOMMA commit latency itself is not supported by this trace.
- Candidate C remains a fallback issue-density/chunk-scheduling variant if A/B compile and run correctly but do not beat baseline.

## Forward Overlap Candidate Loop Final Checkpoint: 2026-06-22

Final checkpoint artifact:

- `forward_overlap_loop_20260622_final_checkpoint.md`

Final default build/import state:

- Rebuilt with `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0`.
- Repo loader sanity probe:
  - timeline API present: `true`
  - `read_mxfp4_forward_timeline()` length: `0`
  - h4/s2048 auto config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`
- No candidate was installed as the default selector.

Baseline:

- Artifact: `forward_overlap_loop_20260622_default_postfixedslot_gpu2.json`
- Physical GPU: GPU2 via `CUDA_VISIBLE_DEVICES=2`, visible as `cuda:0`.
- Direct-preallocated median/min: `0.059552 ms` / `0.057216 ms`.
- Finite: `true`.
- Timeline records in default build: `0`.
- Recheck artifact after candidate runs: `forward_overlap_loop_20260622_default_recheck_after_candidates_gpu2.json`
  - median/min: `0.061248 ms` / `0.056928 ms`
  - finite: `true`

Candidate A, PV-owned reuse ownership:

- Config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_pvarruse_vtma_vstma_pstage3_q200_p112_o56_qkscfix`
- Artifact: `forward_overlap_loop_20260622_candidateA_pairpsc_pvarruse_gpu2.json`
- Direct-preallocated median/min: `0.063680 ms` / `0.060224 ms`.
- Finite: `true`.
- Decision: rejected, about `6.9%` slower than baseline.
- Reason: the trace-supported reuse-ownership idea required the pstage3 pair-P-scale path in this candidate and increased pressure; PTXAS emitted spills for pstage3 pair-P-scale instantiations, consistent with the slower timing.

Candidate B, pstage2 scheduling/issue cadence:

- Config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix`
- Single-run artifact: `forward_overlap_loop_20260622_candidateB_schedwg4_persistreuse_gpu2.json`
- Single-run direct-preallocated median/min: `0.059520 ms` / `0.056928 ms`.
- Interleaved artifact: `forward_overlap_loop_20260622_interleaved_B_vs_default_gpu2.json`
- Interleaved baseline median: `0.065072 ms`.
- Interleaved candidate median: `0.065376 ms`.
- Interleaved candidate/baseline median ratio: `1.004672`.
- Finite: `true` for both baseline and candidate paths.
- Decision: rejected. The single-run median was noise-scale below the fixed baseline, but the interleaved median did not confirm a robust speedup.

Candidate C, pstage2 issue-density/accumulator scheduling:

- Config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix`
- Artifact: `forward_overlap_loop_20260622_candidateC_issuedonelead_gpu2.json`
- Direct-preallocated median/min: `0.060800 ms` / `0.059136 ms`.
- Finite: `true`.
- Decision: rejected, about `2.1%` slower than baseline.
- Reason: more aggressive issue/reuse scheduling did not overcome the control-flow/int-predicate overhead profile, and it did not reduce the exposed pre-QK issue gap enough to land.

Point status:

- Item 1, fixed-slot timeline: landed and verified.
- Item 2, producer/consumer overlap at P->PV: attempted through bounded ownership/scheduling candidates, rejected for this loop.
- Item 3, double-buffer only P payload/scales/descriptors: no new change landed. The default remains pstage2; pstage3 pressure was rejected.
- Item 4, raise UTCOMMA issue density before softmax rework: attempted through Candidate B/C, not landed.

Final decision:

- No overlap/issue-density candidate landed because none was correct and robustly faster than the `0.059552 ms` fixed-slot baseline.
- Active build remains the default timeline-off MXFP4 forward path.
- Next concrete bottleneck: split the long QK pre-issue wait into K arrival, K-scale arrival, score-slot copy reuse, and P-scale/P-stage reuse waits with deterministic fixed-slot records.

## QK Wait Split Timeline: 2026-06-22

Report:

- `forward_overlap_loop_20260622_qkwaitsplit_report.md`

Trace artifacts:

- `forward_overlap_loop_20260622_qkwaitsplit_gpu2.json`
- `forward_overlap_loop_20260622_qkwaitsplit_gpu2_events.csv`
- `forward_overlap_loop_20260622_qkwaitsplit_gpu2_summary.json`

Instrumentation patch:

- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - Added compile-gated deterministic fixed-slot events 17-26 to split the existing `qk_wait_begin -> qk_wait_done` window.
  - Split QK pre-issue wait into K payload, K-scale, P-scale/alias reuse, score-copy, and score-spare reuse waits.
  - Used thread `256` as the QK elected lane and copy-buffer lane slots for score-copy records.
- `results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_driver.py`
  - Added decoder names and summary medians for the new split events.

Build and trace:

- Default build before trace: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0`
  - Result: success.
  - Import/readout: `read_mxfp4_forward_timeline()` length `0`.
- Timeline build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=1`
  - Result: success.
- GPU2 trace command:
  `timeout 180s env CUDA_VISIBLE_DEVICES=2 python3 results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_timeline_driver.py --device cuda:0 --warmup 0 --iters 1 --output-prefix results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_qkwaitsplit_gpu2`
  - Result: finite `true`.
  - Decoded records: `232`.
  - Device: physical GPU2, visible as `cuda:0`, `NVIDIA GB200`.
  - Instrumented runtime: `283.338 ms`, not a performance number.
- Restored default build after trace: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0`
  - Result: success.
  - Import/readout: `read_mxfp4_forward_timeline()` length `0`.
  - h4/s2048 auto config remains `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`.

Key split result:

- Long-index set from the prior fixed-slot trace remained `idx=3,4,6,8,10,12,14`.
- Long-index QK wait total in the split trace: `19,828 cycles`.
- K payload wait total: `13,239 cycles`, `66.8%` of long QK wait.
- K-scale wait total: `1,478 cycles`, `7.5%`.
- Score-copy wait total: `1,419 cycles`, `7.2%`.
- Unattributed in-window overhead/gaps: `3,692 cycles`, `18.6%`.
- The dominant owner for every long window is `qk_k_wait_begin -> qk_k_wait_done`.

Per-long-idx owner:

- `idx=3`: QK `4043`, K payload `3113` (`77.0%`).
- `idx=4`: QK `2133`, K payload `1180` (`55.3%`).
- `idx=6`: QK `2492`, K payload `1522` (`61.1%`).
- `idx=8`: QK `3006`, K payload `2088` (`69.5%`).
- `idx=10`: QK `2649`, K payload `1735` (`65.5%`).
- `idx=12`: QK `2871`, K payload `1980` (`69.0%`).
- `idx=14`: QK `2634`, K payload `1621` (`61.5%`).

Decision:

- No targeted candidate was implemented in this checkpoint.
- The split identifies a clear bottleneck, but it is K payload arrival from the producer path, not a locally safe QK consumer-side synchronization tweak.
- A correct candidate needs to be producer-side K prefetch/scheduling work around `run_k_payload_scale_stage_for_tile` and the `k_finished`/`k_shared_finished` K buffer lifetime. A small wait-site change would be a guess, so it was not attempted.

## K-Pipeline Producer Scheduling: 2026-06-23

Report:

- `forward_overlap_loop_20260623_kpipe_report.md`

Code/config changed:

- `tk_fa4/fp4_fa4_fwd/Makefile`
  - Added `KPIPE_STAGE ?= 0`.
  - Added `-DTK_FA4_MXFP4_FWD_KPIPE_STAGE=$(KPIPE_STAGE)`.
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - Added compile-time KPIPE controls.
  - Added producer-side K prefetch helpers using existing `C::LOAD_STAGES`, `k_finished`, and `k_shared_finished`.
  - Added no new K shared-memory slot.

Stage 1, `KPIPE_STAGE=1`:

- Build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=1`
  - Result: success.
- Smoke artifact: `forward_overlap_loop_20260623_kpipe_stage1_smoke_gpu2.json`
  - Result: finite `true`.
- Timing artifact: `forward_overlap_loop_20260623_kpipe_stage1_timing_gpu2.json`
  - GPU2 direct-prealloc median: `0.059648 ms`.
  - Min: `0.058016 ms`.
  - Timeline records: `0`.
- Trace artifacts:
  - `forward_overlap_loop_20260623_kpipe_stage1_qkwaitsplit_gpu2.json`
  - `forward_overlap_loop_20260623_kpipe_stage1_qkwaitsplit_gpu2_events.csv`
  - `forward_overlap_loop_20260623_kpipe_stage1_qkwaitsplit_gpu2_summary.json`
- Decision: rejected. Original long-set K wait dropped, but all-index K wait only moved from `14,881` to `14,513 cycles`, shifting long waits to later indices.

Stage 2, `KPIPE_STAGE=2`:

- Build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=2`
  - Result: success.
- Smoke artifact: `forward_overlap_loop_20260623_kpipe_stage2_smoke_gpu2.json`
  - Result: finite `true`.
- Timing artifact 1: `forward_overlap_loop_20260623_kpipe_stage2_timing_gpu2.json`
  - GPU2 direct-prealloc median: `0.057888 ms`.
  - Min: `0.055808 ms`.
  - Timeline records: `0`.
- Trace artifacts:
  - `forward_overlap_loop_20260623_kpipe_stage2_qkwaitsplit_gpu2.json`
  - `forward_overlap_loop_20260623_kpipe_stage2_qkwaitsplit_gpu2_events.csv`
  - `forward_overlap_loop_20260623_kpipe_stage2_qkwaitsplit_gpu2_summary.json`
  - Decoded records: `232`.
- Timing artifact 2: `forward_overlap_loop_20260623_kpipe_stage2_timing2_gpu2.json`
  - GPU2 direct-prealloc median: `0.062624 ms`.
  - Min: `0.058432 ms`.
  - Timeline records: `0`.
- Same-window default artifact: `forward_overlap_loop_20260623_default_final_gpu2.json`
  - GPU2 direct-prealloc median: `0.065344 ms`.
  - Min: `0.062208 ms`.
  - Timeline records: `0`.

Stage 2 trace result:

- All-index K payload wait: `14,881 -> 4,943 cycles`.
- Long-set K payload wait (`idx=3,4,6,8,10,12,14`): `13,239 -> 2,799 cycles`.
- Long-set QK wait: `19,828 -> 12,368 cycles`.
- Score-copy/reuse wait became the moved bottleneck: all-index `2,913 -> 10,150 cycles`.

Stage 2 decision:

- Keep as compile-gated candidate only (`KPIPE_STAGE=2`).
- Do not make default. It has strong timeline evidence and one faster timing pass, but the second timing pass did not beat the robust `0.059552 ms` baseline in absolute terms.

Stage 3:

- Not run. Stage 2 moved K wait substantially, so the Stage 3 condition did not apply.

Stage 4:

- Not implemented. Stage 2 moved K wait without new K buffers, so an extra K slot is not yet justified. Adding a K slot would affect the `C::LOAD_STAGES`-sized K ring and semaphore ownership.

Final state:

- Restored default build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0`
  - Result: success.
- Verification on GPU2:
  - `read_mxfp4_forward_timeline()` returned `0`.
  - h4/s2048 effective/runtime config remains `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`.

Next bottleneck:

- Stage 2 exposes score-copy/reuse wait as the next local target while preserving K lead behavior behind `KPIPE_STAGE=2` for controlled comparisons.

## KPIPE Stability And Score-Copy Follow-Up: 2026-06-23

Report:

- `forward_overlap_loop_20260623_kpipe_stability_scorecopy_report.md`

Code/config changed:

- `tk_fa4/fp4_fa4_fwd/Makefile`
  - Added defaults and NVCC macros for `MXFP4_FWD_TIMELINE`, `KPIPE_STAGE`, `SCORE_REUSE_PIPE_STAGE`, and `KPIPE_SELECTIVE_POLICY`.
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - Added gated score-copy preconsume candidate: `SCORE_REUSE_PIPE_STAGE=1`.
  - Added gated selective K lead policy: `KPIPE_SELECTIVE_POLICY=1`.
  - Preserved default behavior behind `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0`.

Stability result:

- Paired Stage0/Stage2 timing artifacts:
  - `forward_overlap_loop_20260623_kpipe_stability_pair1_stage0_gpu2.json`: p50 `0.060416 ms`, min `0.058432 ms`, p10/p90 `0.059072/0.062205 ms`, finite `true`, timeline records `0`.
  - `forward_overlap_loop_20260623_kpipe_stability_pair1_stage2_gpu2.json`: p50 `0.060112 ms`, min `0.057536 ms`, p10/p90 `0.058742/0.062131 ms`, finite `true`, timeline records `0`.
  - `forward_overlap_loop_20260623_kpipe_stability_pair2_stage0_gpu2.json`: p50 `0.061360 ms`, min `0.059392 ms`, p10/p90 `0.060154/0.063955 ms`, finite `true`, timeline records `0`.
  - `forward_overlap_loop_20260623_kpipe_stability_pair2_stage2_gpu2.json`: p50 `0.060784 ms`, min `0.058880 ms`, p10/p90 `0.059635/0.062368 ms`, finite `true`, timeline records `0`.
- Stage2 won both paired medians by a small margin, but remained non-robust versus saved repeat artifact `forward_overlap_loop_20260623_kpipe_stage2_timing2_gpu2.json` p50 `0.062624 ms` and the older robust baseline `0.059552 ms`.

Candidate results:

- Score-preconsume, `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=1 KPIPE_SELECTIVE_POLICY=0`
  - Smoke: `forward_overlap_loop_20260623_scorepre_stage1_smoke_gpu2.json`, finite `true`, timeline records `0`.
  - Timing: `forward_overlap_loop_20260623_scorepre_stage1_timing_gpu2.json`, p50 `0.060064 ms`, min `0.057984 ms`, p10/p90 `0.058909/0.061846 ms`, finite `true`.
  - Trace: `forward_overlap_loop_20260623_scorepre_stage1_qkwaitsplit_gpu2_summary.json`, decoded `204`, all-index QK/K/score-copy sums `12,548/2,869/0 cycles`.
  - Decision: keep gated as diagnostic only; not a timing win.
- Selective K, `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=1`
  - Smoke: `forward_overlap_loop_20260623_kpipe_selective_smoke_gpu2.json`, finite `true`, timeline records `0`.
  - Timing: `forward_overlap_loop_20260623_kpipe_selective_timing_gpu2.json`, p50 `0.058656 ms`, min `0.056448 ms`, p10/p90 `0.057312/0.060448 ms`, finite `true`.
  - Trace: `forward_overlap_loop_20260623_kpipe_selective_qkwaitsplit_gpu2_summary.json`, decoded `232`, all-index QK/K/score-copy sums `26,486/12,687/3,431 cycles`, long-set QK/K/score-copy sums `8,491/1,376/2,149 cycles`.
  - Decision: best new gated candidate, but not defaulted from one 40-iter timing run.
- Combined, `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=1 KPIPE_SELECTIVE_POLICY=1`
  - Smoke: `forward_overlap_loop_20260623_combined_smoke_gpu2.json`, finite `true`, timeline records `0`.
  - Timing: `forward_overlap_loop_20260623_combined_timing_gpu2.json`, p50 `0.060720 ms`, min `0.058336 ms`, p10/p90 `0.059149/0.062794 ms`, finite `true`.
  - Trace: `forward_overlap_loop_20260623_combined_qkwaitsplit_gpu2_summary.json`, decoded `204`, all-index QK/K/score-copy sums `16,610/6,837/0 cycles`.
  - Decision: rejected as default; worse direct timing than selective K.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_overlap_loop_20260623_final_restore_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0`, config unchanged.
- Direct import verification:
  - `read_mxfp4_forward_timeline()` length `0`.
  - h4/s2048 config remains `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`.

Next bottleneck:

- Selective K reduces the long-index K payload waits without the full Stage2 score-copy blowup, but all-index K payload wait remains `12,687 cycles`. Next work should broaden selective producer scheduling without reintroducing full Stage2 score-copy pressure.

## Selective K Lead Broadening: 2026-06-23

Report:

- `forward_overlap_loop_20260623_selective_k_broaden_report.md`

Code/config changed:

- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - Preserved existing `KPIPE_SELECTIVE_POLICY=1` behavior.
  - Added policy 2 lead-2 set `{3,4,5,6,8,9,10,11,12,13,14}`.
  - Added policy 3 lead-2 set `{3,4,6}` plus `candidate_idx >= 8`.
  - Lead 1 remains always allowed; no K buffers or semaphores added.

Policy 1 baseline:

- Timing artifact `forward_overlap_loop_20260623_kpipe_selective_timing_gpu2.json`: p10/p50/p90/min `0.057312/0.058656/0.060448/0.056448 ms`.
- Trace artifact `forward_overlap_loop_20260623_kpipe_selective_qkwaitsplit_gpu2_summary.json`: all-index QK/K/score-copy `26,486/12,687/3,431 cycles`; long-set K `1,376 cycles`.
- Remaining non-long K wait under policy 1 came mainly from idx `5=2,811`, `9=1,873`, `11=3,125`, `13=2,257`, plus smaller idx `15=464`, `7=385`.

Policy 2:

- Build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=2`
  - Result: success.
- Smoke: `forward_overlap_loop_20260623_kpipe_selective_policy2_smoke_gpu2.json`, finite `true`, decoded timeline records `0`.
- Timing: `forward_overlap_loop_20260623_kpipe_selective_policy2_timing_gpu2.json`, p10/p50/p90/min `0.059450/0.060736/0.062314/0.058368 ms`, finite `true`, timeline records `0`.
- Trace: `forward_overlap_loop_20260623_kpipe_selective_policy2_qkwaitsplit_gpu2_summary.json`, decoded `232`, all-index QK/K/score-copy `28,112/10,530/6,920 cycles`.
- Decision: rejected as a performance candidate. K wait improved only `2,157 cycles` vs policy 1 while score-copy increased `3,489 cycles`, QK wait increased, and direct timing regressed.

Policy 3:

- Build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=3`
  - Result: success.
- Smoke: `forward_overlap_loop_20260623_kpipe_selective_policy3_smoke_gpu2.json`, finite `true`, decoded timeline records `0`.
- Timing: `forward_overlap_loop_20260623_kpipe_selective_policy3_timing_gpu2.json`, p10/p50/p90/min `0.060202/0.061568/0.063907/0.059104 ms`, finite `true`, timeline records `0`.
- Trace: `forward_overlap_loop_20260623_kpipe_selective_policy3_qkwaitsplit_gpu2_summary.json`, decoded `232`, all-index QK/K/score-copy `25,702/7,255/7,992 cycles`.
- Decision: rejected as a performance candidate. K wait improved `5,432 cycles` vs policy 1, but score-copy increased `4,561 cycles` and direct timing regressed.

Final decision:

- No broader policy beat policy 1.
- Keep policy 1 as the best accepted compile-gated candidate only; do not default it.
- Policies 2 and 3 remain compile-gated for diagnostics, but are rejected as timing candidates.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_overlap_loop_20260623_selective_k_broaden_final_restore_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0`, config unchanged.
- Direct import verification:
  - `read_mxfp4_forward_timeline()` length `0`.
  - h4/s2048 config remains `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`.
  - Active shared object is `tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so`, rebuilt as default, not candidate/timeline.

Next bottleneck:

- Broader static lead-2 policies reduce K payload wait but recreate score-copy/reuse pressure. Next work should make lead-2 selection score-copy-aware or reduce score-copy pressure before broadening K prefetch further.

## Score-Copy-Aware Selective KPIPE: 2026-06-23

Report:

- `forward_overlap_loop_20260623_scorecopy_aware_kpipe_report.md`

Code/config changed:

- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - Preserved policies 1-3 exactly.
  - Added policy 4: policy 1 plus lead-2 idx `5`, set `{3,4,5,6,8,10,12,14}`.
  - Added policy 5: policy 1 plus lead-2 idx `11`, set `{3,4,6,8,10,11,12,14}`.
  - Added policy 6: policy 1 plus lead-2 idx `5` and `11`, set `{3,4,5,6,8,10,11,12,14}`.
  - Lead 1 remains always allowed; no K buffers or semaphores added.

Timing and smoke:

- Policy 4:
  - Build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=4`
  - Smoke: `forward_overlap_loop_20260623_scorecopy_aware_policy4_smoke_gpu2.json`, finite `true`, timeline raw/decoded `0/0`.
  - Timing: `forward_overlap_loop_20260623_scorecopy_aware_policy4_timing_gpu2.json`, p10/p50/p90/min/max `0.063114/0.064672/0.067286/0.062176/0.091264 ms`, finite `true`, timeline raw/decoded `0/0`.
- Policy 5:
  - Build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=5`
  - Smoke: `forward_overlap_loop_20260623_scorecopy_aware_policy5_smoke_gpu2.json`, finite `true`, timeline raw/decoded `0/0`.
  - Timing: `forward_overlap_loop_20260623_scorecopy_aware_policy5_timing_gpu2.json`, p10/p50/p90/min/max `0.058429/0.059408/0.060426/0.057696/0.086240 ms`, finite `true`, timeline raw/decoded `0/0`.
- Policy 6:
  - Build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=6`
  - Smoke: `forward_overlap_loop_20260623_scorecopy_aware_policy6_smoke_gpu2.json`, finite `true`, timeline raw/decoded `0/0`.
  - Timing: `forward_overlap_loop_20260623_scorecopy_aware_policy6_timing_gpu2.json`, p10/p50/p90/min/max `0.058941/0.060192/0.062346/0.057952/0.066432 ms`, finite `true`, timeline raw/decoded `0/0`.

Trace:

- Policy 5:
  - Timeline build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=1 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=5`
  - Trace: `forward_overlap_loop_20260623_scorecopy_aware_policy5_qkwaitsplit_gpu2_summary.json`, decoded `232`, all-index QK/K/score-copy `26,346/11,711/4,315 cycles`, long-set `9,013/1,857/2,285 cycles`.
- Policy 6:
  - Timeline build: `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=1 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=6`
  - Trace: `forward_overlap_loop_20260623_scorecopy_aware_policy6_qkwaitsplit_gpu2_summary.json`, decoded `232`, all-index QK/K/score-copy `27,674/12,280/5,071 cycles`, long-set `9,803/2,496/2,514 cycles`.

Decision:

- No new score-copy-aware policy beat existing policy 1 (`0.058656 ms` p50).
- Policy 5 was the best new policy (`0.059408 ms` p50) and reduced all-index K payload wait by `976 cycles`, but score-copy rose by `884 cycles` and direct timing still regressed.
- Policy 6 confirmed that combining idx `5` and `11` worsens score-copy/QK pressure.
- Policy 4 was rejected by timing before trace.
- Keep policy 1 as the best accepted compile-gated selective K candidate only; do not default it. Policies 4-6 remain diagnostic compile gates.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_overlap_loop_20260623_scorecopy_aware_final_restore_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`, config unchanged.
- Direct import verification:
  - `read_mxfp4_forward_timeline()` length `0`.
  - h4/s2048 config remains `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`.
  - Active shared object is `tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so`, rebuilt as default, not candidate/timeline.

Next bottleneck:

- The residual K-wait slots are real, but static lead-2 additions quickly move the limiter into score-copy/reuse. Further work should reduce score-copy pressure or make producer selection dynamic/score-state-aware before broadening K prefetch again.

## Next-Move Forward Profile: 2026-06-23
Report: `forward_overlap_loop_20260623_nextmove_profile_report.md`.
Scope: forward-only MXFP4 FA4 h4/s2048 on physical GPU2. No backward code was touched.

Timing p50 h4/s2048, warmup20/iters100:
- `stage0_default`: 0.066800 ms p50, p10/p90 0.064666/0.069101, finite True, timeline-off raw/decoded 0/0.
- `policy1_best`: 0.059664 ms p50, p10/p90 0.058317/0.061702, finite True, timeline-off raw/decoded 0/0.
- `policy5_scorecopy_aware`: 0.062528 ms p50, p10/p90 0.060506/0.065357, finite True, timeline-off raw/decoded 0/0.
- `full_stage2`: 0.059200 ms p50, p10/p90 0.057856/0.060182, finite True, timeline-off raw/decoded 0/0.
- `scorepre_stage1`: 0.062944 ms p50, p10/p90 0.061376/0.064358, finite True, timeline-off raw/decoded 0/0.

Timeline all-index totals:
- `stage0_default`: QK/K/score/PVissue/PVout/epi 28,256/14,697/3,029/3,172/4,375/2,095 cycles.
- `policy1_best`: QK/K/score/PVissue/PVout/epi 26,354/12,679/3,145/3,230/4,480/1,966 cycles.
- `policy5_scorecopy_aware`: QK/K/score/PVissue/PVout/epi 26,479/11,422/4,653/3,137/4,514/1,979 cycles.
- `full_stage2`: QK/K/score/PVissue/PVout/epi 26,517/5,084/10,771/3,115/4,481/1,973 cycles.
- `scorepre_stage1`: QK/K/score/PVissue/PVout/epi 12,547/2,927/0/3,119/4,621/1,962 cycles.

NCU key results:
- `stage0_default`: 49.888 us, TC 3.05, issue 7.19, eligible 0.37, long SB 3.61, TMA 0.06.
- `policy1_best`: 48.992 us, TC 3.11, issue 7.56, eligible 0.39, long SB 3.42, TMA 0.06.
- `policy5_scorecopy_aware`: 49.280 us, TC 3.08, issue 7.47, eligible 0.39, long SB 3.42, TMA 0.06.
- `full_stage2`: 48.768 us, TC 3.17, issue 7.59, eligible 0.39, long SB 3.44, TMA 0.06.

Decision: fresh 100-sample h4/s2048 data makes `full_stage2` the narrow candidate winner, but not a default change. Score-copy is a pressure/proxy signal: policy 5 loses because its modest K relief is offset by score-copy growth; full Stage2 wins here because K relief is large enough despite high score-copy. Next move is to validate full Stage2 across seeds/shapes, then reduce score-copy/reuse pressure inside the broad lead-2 family.

Optional checks: h4/s1024 p50 default/policy1/policy5 = 0.042048/0.042480/0.041024 ms; h16/s2048 p50 default/policy1/policy5 = 0.069824/0.064864/0.066176 ms. BF16 TK h4/s2048 p50 `0.096128 ms`.

Final state: restored with `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0`; final smoke finite true, timeline raw/decoded `0/0`; direct timeline length `0`; h4/s2048 config unchanged: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`.

## Hot-Plate Slot Scheduler: 2026-06-23

Report: `forward_hotplate_slot_scheduler_20260623_report.md`.
Checkpoint: `forward_hotplate_slot_scheduler_20260623_checkpoint.md`.

Code/config changed:

- Added default-off gates `HOTPLATE_SLOT_SCHED` and `HOTPLATE_POLICY` to the forward Makefile.
- Added timeline events for `hotplate_slot_release` and QK target-slot reusable waits.
- Added parser deltas for PV issue to slot release and slot release to next QK issue.
- Implemented Policy A as PV-owned reusable-slot release and Policy B as reusable-state wait plus V/P-scale prestage ordering.

Policy results:

- Policy A built, but GPU2 smoke hung twice and was killed. Failure mode: in the current role layout, QK/issue can wait on the target slot before the producer/quant/PV path reaches the PV-owned release point, creating an ordering cycle. Rejected pending larger schedule rewrite.
- Policy B command family: `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=2`.
- Policy B smoke: `forward_hotplate_slot_scheduler_20260623_policy2_smoke_gpu2.json`, finite `true`, timeline raw/decoded `0/0`.

h4/s2048 timing, GPU2 warmup20/iters100:

- Stage0/default: `0.066800 ms` p50, `forward_overlap_loop_20260623_nextmove_stage0_default_timing_gpu2.json`.
- Current best full Stage2: `0.059200 ms` p50, `forward_overlap_loop_20260623_nextmove_full_stage2_timing_gpu2.json`.
- Hot-plate Policy B: `0.059440 ms` p50, `forward_hotplate_slot_scheduler_20260623_policy2_timing_gpu2.json`.
- Decision: Policy B is `11.0%` lower time than Stage0/default, but `0.4%` slower than full Stage2 p50.

Timeline medians:

- Default trace `forward_hotplate_slot_scheduler_20260623_default_trace2_gpu2_summary.json`: QK wait `1136`, score-copy wait `189.5`, QK issue->commit `361`, PV issue->release `-3645`, release->next QK `4055.5` cycles.
- Policy B trace `forward_hotplate_slot_scheduler_20260623_policy2_trace_gpu2_summary.json`: QK wait `2051`, score-copy wait `1076.5`, QK issue->commit `290`, PV issue->release `-3612`, release->next QK `446` cycles.
- Slot release did not become PV-owned in accepted Policy B; negative PV issue->release shows it remains the existing early score-copy/slot-ready release. Policy B reduces release->next-QK delay but increases score-copy/reuse pressure.

NCU:

- Policy B NCU: `forward_hotplate_slot_scheduler_20260623_policy2_ncu_summary_gpu2.json`, duration `48.384 us`, regs/thread `168`, barriers `2`, dynamic smem `98.0 KiB`, issue active `7.57`, eligible warps `0.39`, long scoreboard `3.44`.
- Existing Stage0/full Stage2 NCU durations: `49.888/48.768 us`. NCU slightly favored Policy B, but wall p50 still favored full Stage2.

Shape sweep:

- h4/s1024: Stage0/policy5/PolicyB p50 `0.042048/0.041024/0.041728 ms`; Policy B loses to policy5.
- h4/s2048: Stage0/fullStage2/PolicyB p50 `0.066800/0.059200/0.059440 ms`; Policy B loses narrowly to full Stage2.
- h4/s4096: Policy B measured `0.094720 ms`; no fresh baseline/current-best rerun in this task group.
- h16/s2048: Stage0/policy1/PolicyB p50 `0.069824/0.064864/0.066176 ms`; Policy B loses to policy1.

Decision:

- Do not default the hot-plate scheduler.
- Suggested selector from measured data: full Stage2 for h4/s2048, policy5 score-copy-aware for h4/s1024, policy1 selective-K for h16/s2048.
- Next work should remove the Policy A role-ordering deadlock before trying true PV-owned slot release again.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_hotplate_slot_scheduler_20260623_final_restore_build_gpu2.status` = `0`.
- Final GPU2 smoke:
  - `forward_hotplate_slot_scheduler_20260623_final_restore_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_hotplate_slot_scheduler_20260623_final_restore_direct_timeline_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

## Hot-Plate Service-PV-First Follow-up: 2026-06-23

Report: `forward_hotplate_service_pv_first_20260623_report.md`.
Checkpoint: `forward_hotplate_service_pv_first_20260623_checkpoint.md`.

Code/config changed:

- Added `HOTPLATE_POLICY=3`: PV-owned release, prestage enabled, online QK two-ahead suppressed.
- Added `HOTPLATE_POLICY=4`: dynamic nonblocking PV-release probe using `hotplate_slot_release_epoch[slot]`, with prestage enabled.
- Split hotplate semantics into PV release signal, blocking PV-owned release, and nonblocking probe. Blocking policies use the semaphore; probe policy uses a shared epoch flag.

Results:

- Policy 3 built, but GPU2 smoke timed out after 120s: `forward_hotplate_service_pv_first_20260623_policy3_smoke_gpu2.status` = `124`. Suppressing two-ahead is not enough; blocking QK on PV-owned release is still unsafe in this role layout.
- Initial Policy 4 mbarrier-probe variant was finite but slow (`0.116096 ms` p50), showing unused release mbarrier phases are too expensive/backpressure-prone for observation.
- Epoch Policy 4 smoke: `forward_hotplate_service_pv_first_20260623_policy4_epoch_smoke_gpu2.json`, finite `true`, timeline raw/decoded `0/0`.
- Epoch Policy 4 h4/s2048 timing: `forward_hotplate_service_pv_first_20260623_policy4_epoch_timing_gpu2.json`, p10/p50/p90/min `0.057728/0.059232/0.062797/0.056224 ms`, finite `true`, timeline raw/decoded `0/0`.
- Comparison p50: full Stage2 current best `0.059200 ms`, prior hotplate Policy B `0.059440 ms`, epoch Policy 4 `0.059232 ms`.

Timeline:

- Trace: `forward_hotplate_service_pv_first_20260623_policy4_epoch_trace_gpu2_summary.json`, decoded `276`.
- Median cycles: PV issue->slot release `318`, release->next QK `2186.5`, QK slot probe `204`, QK score-copy wait `505`, QK wait `1947`, QK issue->commit `307`, PV issue->commit `194`.
- Probe-ready parse: `forward_hotplate_service_pv_first_20260623_policy4_epoch_probe_ready_gpu2.json`, QK probes `14`, ready `0`, not ready `14`, ready fraction `0.0`.

Decision:

- Nonblocking dynamic probing solves the deadlock/backpressure failure mode and keeps wall time near current best, but it does not unlock a new win yet.
- The measured reason is clear: true PV-owned release happens after PV issue, but it is never ready at the current QK decision point. Blocking on it is wrong; observing it and falling back is safe.
- Next scheduler step should move the QK decision point later or add a pending-QK state machine that issues only when the target slot epoch is ready, while doing independent work instead of waiting.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_hotplate_service_pv_first_20260623_final_restore_build_gpu2.status` = `0`.
- Final GPU2 smoke:
  - `forward_hotplate_service_pv_first_20260623_final_restore_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_hotplate_service_pv_first_20260623_final_restore_direct_timeline_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

## Pending-QK Hot-Plate Scheduler: 2026-06-24

Report: `forward_pending_qk_scheduler_20260624_report.md`.
Checkpoint: `forward_pending_qk_scheduler_20260624_checkpoint.md`.

Code/config changed:

- Added `HOTPLATE_POLICY=5`: minimal late-QK reorder. It queues the normal QK candidate in `run_iteration`, moves the actual issue after the local PV service point, does not block on PV release, and falls back to the existing legal issue path.
- Added `HOTPLATE_POLICY=6`: explicit pending-QK metadata state. It suppresses QK two-ahead, carries one pending candidate, and issues only when the PV-release epoch says the target slot is reusable.
- No default behavior change; all new behavior is behind `HOTPLATE_SLOT_SCHED=1`.

Policy 5:

- Build: `forward_pending_qk_scheduler_20260624_policy5_build_timelineoff_gpu2.log`, status `0`.
- Smoke: `forward_pending_qk_scheduler_20260624_policy5_smoke_gpu2.json`, finite `true`, timeline raw/decoded `0/0`.
- h4/s2048 timing: `forward_pending_qk_scheduler_20260624_policy5_timing_gpu2.json`, p10/p50/p90/min `0.058074/0.058864/0.060579/0.056096 ms`, finite `true`, timeline raw/decoded `0/0`.
- p50 is `11.88%` lower than Stage0/default (`0.066800 ms`) and `0.57%` lower than full Stage2 reference (`0.059200 ms`).
- Trace: `forward_pending_qk_scheduler_20260624_policy5_trace_gpu2_summary.json`, decoded `276`.
- Median cycles: QK wait `1905`, score-copy wait `617.5`, QK slot probe `183`, QK issue->commit `302`, PV issue->commit `191`, PV issue->slot release `320`, release->next QK `2232`, PV commit->output wait `274.5`, epilogue `1965`.
- Epoch-ready parse: `forward_pending_qk_scheduler_20260624_policy5_probe_ready_gpu2.json`, QK probes `14`, ready `0`, not ready `14`, ready fraction `0.0`.
- NCU: `forward_pending_qk_scheduler_20260624_policy5_ncu_summary_gpu2.json`, duration `48.832 us`, regs/thread `168`, barriers `2`, dynamic smem `98.0 KiB`, issue active `7.51`, eligible warps `0.39`, long scoreboard `3.38`.

Policy 6:

- Build: `forward_pending_qk_scheduler_20260624_policy6_build_timelineoff_gpu2.log`, status `0`.
- Smoke: `forward_pending_qk_scheduler_20260624_policy6_smoke_gpu2.status` = `124`, stdout/stderr empty.
- Rejected. Strict one-pending QK can withhold a needed score tile until downstream work waits for it; one pending slot without an alternate legal target is insufficient.

Shape sweep, sequential Policy 5 artifacts:

- h4/s1024: `forward_pending_qk_scheduler_20260624_policy5_shape_s1024_h4_seq_timing_gpu2.json`, p50 `0.039840 ms`, beats prior policy5-scorecopy `0.041024 ms`.
- h4/s2048: p50 `0.058864 ms`, beats full Stage2 `0.059200 ms`.
- h16/s2048: `forward_pending_qk_scheduler_20260624_policy5_shape_s2048_h16_seq_timing_gpu2.json`, p50 `0.066176 ms`, loses to policy1 `0.064864 ms`.
- h4/s4096: `forward_pending_qk_scheduler_20260624_policy5_shape_s4096_h4_seq_timing_gpu2.json`, p50 `0.096416 ms`, loses to previous hotplate policy2 `0.094720 ms`.
- Concurrently launched shape artifacts without `_seq_` are ignored for timing comparisons.

Decision:

- Policy 5 is the best safe pending-QK candidate and should remain compile-gated.
- It improves h4/s1024 and h4/s2048, but it does not make PV epochs ready at the QK decision point. The win is schedule-pressure/order, not successful PV-epoch exploitation.
- Suggested selector: Policy 5 for h4/s1024 and h4/s2048; policy1 selective-K for h16/s2048; previous hotplate/Policy 2 reference for h4/s4096 until a fresh full comparison is run.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_pending_qk_scheduler_20260624_final_restore_build_gpu2.status` = `0`.
- Final GPU2 smoke:
  - `forward_pending_qk_scheduler_20260624_final_restore_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_pending_qk_scheduler_20260624_final_restore_direct_timeline_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

## Forward Bottleneck Diagnostics: 2026-06-24

Report: `forward_bottleneck_diagnostics_20260624_report.md`.
Checkpoint: `forward_bottleneck_diagnostics_20260624_checkpoint.md`.

Diagnostics only; no optimization patch.

Verified candidate flag families:

- `stage0_default`: `MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`.
- `policy1_best`: `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=1 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`.
- `full_stage2`: `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`.
- `hotplate_policy2`: `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=2`.
- `hotplate_policy4_epoch`: `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=4`.
- `pending_policy5`: `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=5`.

Fresh timing matrix:

- Artifact: `forward_bottleneck_diagnostics_20260624_timing_matrix_summary.json`.
- All 6 builds and all 24 shape timings completed status `0`, finite `true`, timeline raw/decoded `0/0`, warmup `20`, iters `100`.
- p50 winners in this rebuild window: h4/s1024 `policy1_best` `0.040688 ms`; h4/s2048 `full_stage2` `0.058720 ms`; h16/s2048 `hotplate_policy4_epoch` `0.064832 ms`; h4/s4096 `policy1_best` `0.094864 ms`.
- Fresh matrix did not reproduce the prior Policy 5 h4/s1024/h4/s2048 win; Policy 5 p50s were `0.041840`, `0.060464`, `0.066176`, `0.096256 ms` for h4/s1024, h4/s2048, h16/s2048, h4/s4096.

Timeline:

- Artifact: `forward_bottleneck_diagnostics_20260624_timeline_matrix_summary.json`.
- All 6 timeline-on builds and all 24 trace runs completed status `0`.
- Policy 5 probe-ready stayed `0/6`, `0/14`, and `0/30`, so the win mechanism is schedule order/pressure, not successful PV-owned slot reuse.
- h4/s2048 Policy 5 vs policy4 lowers QK wait `1976 -> 1773`, K wait `259 -> 193`, score-copy `557 -> 520`, slot probe `215.5 -> 182.5`, PV output wait `295.5 -> 288.5`, and epilogue `1980 -> 1965`.
- h16/s2048 Policy 5 vs policy1 lowers K wait `298 -> 221` but increases QK wait `1294 -> 1891` and score-copy `194 -> 596`.
- h4/s4096 Policy 5 vs policy1 lowers K wait `304 -> 198` but increases QK wait `1313 -> 1953` and score-copy `199 -> 682.5`.

NCU:

- Artifact: `forward_bottleneck_diagnostics_20260624_ncu_matrix_summary.json`.
- Required comparisons plus fresh h4/s4096 policy1 reference completed status `0`, finite `true`.
- h4/s2048 full Stage2 vs Policy 5: duration `48.384 -> 49.184 us`, TC active `3.15 -> 3.04`, issue active `7.55 -> 7.49`.
- h16/s2048 policy1 vs Policy 5: duration `55.072 -> 56.192 us`, TC active `11.11 -> 10.86`, issue active `26.79 -> 26.67`, eligible warps `0.38 -> 0.37`.
- h4/s4096 policy2 vs Policy 5: duration `85.696 -> 85.504 us`, TC active `7.05 -> 6.86`, issue active `15.32 -> 15.31`.

Decision:

- Policy 5 should remain compile-gated; it is not robust enough for default or unconditional selector use from this matrix.
- Next code target is score-copy/reuse lifetime reduction in the dynamic/late-QK family. More static K prefetch is not the primary target because Policy 5 already reduces K wait where it loses, but creates score-copy/reuse and scheduler pressure.
- A shape selector is the lowest-risk immediate policy layer, but the next optimization work should make a dynamic slot scheduler avoid score-copy backpressure or choose another legal QK target rather than strict one-pending gating.

Final state:

- Restored active build:
  - `timeout 900s make -B tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_bottleneck_diagnostics_20260624_final_restore_build_gpu2.status` = `0`.
- Final GPU2 smoke:
  - `forward_bottleneck_diagnostics_20260624_final_restore_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_bottleneck_diagnostics_20260624_final_restore_direct_timeline_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

## Forward Score-Reuse Pressure Experiments: 2026-06-24

Report: `forward_score_reuse_pressure_experiments_20260624_report.md`.
Checkpoint: `forward_score_reuse_pressure_experiments_20260624_checkpoint.md`.

Implemented and profiled Experiment A as compile-gated `HOTPLATE_POLICY=7`:

- Build flags: `MXFP4_FWD_TIMELINE=0/1 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=7`.
- Mechanism: early `preconsume_score_copy_for_idx()` at the original scheduling point, then late QK issue with queued preconsume set to `-1`.
- Timing-off build status `0`; timeline-on build status `0`.
- GPU2 timing p50s: h4/s1024 `0.043232 ms`, h4/s2048 `0.060000 ms`, h16/s2048 `0.066896 ms`, h4/s4096 `0.099280 ms`.
- All timing runs finite `true`, timeline raw/decoded `0/0`.
- Timeline confirmed `0` late-path `qk_score_copy_wait` events, but Policy 7 lost to the fresh per-shape best references:
  - h4/s1024 policy1_best `0.040688 ms`.
  - h4/s2048 full_stage2 `0.058720 ms`.
  - h16/s2048 hotplate_policy4_epoch `0.064832 ms`.
  - h4/s4096 policy1_best `0.094864 ms`.
- Policy7 slot-probe ready/not-ready: h4/s1024 `1/5`, h4/s2048 `1/13`, h16/s2048 `1/13`, h4/s4096 `1/29`.
- NCU skipped because the implemented candidate was not best and the losses were not ambiguous.

Audits/decisions:

- Experiment B `HOTPLATE_POLICY=8`: audit-only rejection. Active scorepack path reaches `arrive_score_copy_done(idx, buf)` after all active score TMEM loads and `tensor_load_wait`; moving earlier would permit QK overwrite before score data is safely registerized.
- Experiment C `HOTPLATE_POLICY=9`: rejected. A nonblocking score-copy probe is not enough; safe multi-target QK reorder would also need candidate-owned K/K-scale slot and phase state because `issue_next_qk()` advances `k_idx`, `k_phase`, and `k_sc_slot`.
- Experiment D shape selector: design-only artifact `forward_score_reuse_pressure_experiments_20260624_shape_selector_design.json`. Current runtime selector chooses config strings, but the overlap policies are extension build macros, so a clean selector needs multiple compiled entrypoints or side-by-side modules.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_score_reuse_pressure_experiments_20260624_final_restore_build_defaultoff_gpu2.status` = `0`.
- Final GPU2 smoke:
  - `forward_score_reuse_pressure_experiments_20260624_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_score_reuse_pressure_experiments_20260624_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

## Forward Selector + QK Candidate Refactor: 2026-06-24

Report: `forward_selector_qk_candidate_refactor_20260624_report.md`.
Checkpoint: `forward_selector_qk_candidate_refactor_20260624_checkpoint.md`.

Selector:

- Implemented a results-local build-and-benchmark selector harness: `forward_selector_qk_candidate_refactor_20260624_selector_harness.py`.
- Output: `forward_selector_qk_candidate_refactor_20260624_selector_harness_summary.json`.
- Selected fresh bests: h4/s1024 `policy1_best`, h4/s2048 `full_stage2`, h16/s2048 `hotplate_policy4_epoch`, h4/s4096 `policy1_best`.
- Four-shape mean p50: selector `0.064776 ms`, default/off `0.068472 ms`, always policy1 `0.065640 ms`, always full-stage2 `0.066012 ms`, always hotplate policy4 `0.065900 ms`.
- Selector speedup: `1.057x` vs default/off, `1.013x` vs always policy1, `1.019x` vs always full-stage2, `1.017x` vs always policy4.
- Production selector still needs multiple compiled entrypoints or side-by-side extension modules; current Python selector only switches runtime config strings, not macro policy builds.

Policy10 QK candidate refactor:

- Added `QkIssueCandidate` state in `fwd_streaming_kernel.inc` with `next_idx`, `copy_buf`, `preconsume_idx`, `final_q_for_task`, `score_slot`, `k_idx`, `k_phase`, `k_sc_slot`, `k_sc_phase`, and reuse phase snapshots.
- Policy10 build flags: `KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=10`.
- Timeline-off and timeline-on builds completed status `0`.
- Policy10 timing p50s: h4/s1024 `0.044016 ms`, h4/s2048 `0.059312 ms`, h16/s2048 `0.068800 ms`, h4/s4096 `0.101104 ms`; all finite and timeline raw/decoded `0/0`.
- Same-source full-stage2 reference p50s: h4/s1024 `0.040928 ms`, h4/s2048 `0.059712 ms`, h16/s2048 `0.065808 ms`, h4/s4096 `0.096128 ms`.
- Decision: policy10 did not prove parity; h4/s2048 was parity-like, but h16/s2048 and h4/s4096 regressed.
- Timeline summary: `forward_selector_qk_candidate_refactor_20260624_policy10_timeline_summary.json`; no PV-owned slot probe events, score-copy wait remained dominant.

Readiness and policy11:

- Readiness classification artifact: `forward_selector_qk_candidate_refactor_20260624_readiness_classification_summary.json`.
- K/K-scale readiness was not made into a nonblocking probe because issue uses route-specific local/cluster wait and expected-byte ownership.
- Score-copy is only partially probeable; a ready probe must transfer ownership and update `p_copy_phase_mask`.
- Policy11 two-candidate issue was not implemented because policy10 was not parity-clean and readiness ownership was not proven.
- NCU skipped because no final candidate beat a fresh reference.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_selector_qk_candidate_refactor_20260624_final_restore_build_defaultoff_gpu2.status` = `0`.
- Final GPU2 smoke:
  - `forward_selector_qk_candidate_refactor_20260624_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_selector_qk_candidate_refactor_20260624_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

## Forward Candidate-Owned K State: 2026-06-24

Report: `forward_candidate_owned_k_state_20260624_report.md`.
Checkpoint: `forward_candidate_owned_k_state_20260624_checkpoint.md`.

Implemented policy12 candidate-owned QK issue state in `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc` and restored a separate direct full-stage2 QK issue path so the reference no longer routes through the candidate helper.

Policy12 candidate captures `next_idx`, `copy_buf`, `preconsume_idx`, `final_q_for_task`, `score_slot`, K slot/phase, K-scale slot/phase, score-copy buf/phase, pair score-copy phases, alias-scale reuse phase, hotplate slot reuse phase, and spare reuse phase. Candidate issue uses phase-specific waits and advances the same global masks exactly once.

Parity results, warmup `20`, iters `100`, GPU2, timeline off:

- h4/s1024: full-stage2 `0.041824 ms`, policy12 `0.041840 ms`, `+0.04%`.
- h4/s2048: full-stage2 `0.060080 ms`, policy12 `0.059376 ms`, `-1.17%`.
- h16/s2048: full-stage2 `0.066256 ms`, policy12 `0.067360 ms`, `+1.67%`.
- h4/s4096: full-stage2 `0.095712 ms`, policy12 `0.094944 ms`, `-0.80%`.
- All timing runs finite; timeline-off raw/decoded `0/0`.

Timeline-on parity, warmup `5`, iters `10`, GPU2:

- h4/s2048 raw/decoded matched `248/248`; QK wait `2140 -> 2108`, score-copy `1147 -> 1191.5` cycles.
- h16/s2048 raw/decoded matched `248/248`; QK wait `2106 -> 2086`, score-copy `1220.5 -> 1224` cycles.
- h4/s4096 raw/decoded matched `504/504`; QK wait `2128 -> 2124`, score-copy `1201 -> 1215.5` cycles.
- No new QK/K/K-scale/score-copy wait spike; event counts matched.

Readiness and policy13:

- Readiness classification artifact: `forward_candidate_owned_k_state_20260624_readiness_classification_summary.json`.
- Policy13 was not implemented because K/K-scale and output/spare readiness still lack a proven safe non-consuming probe for a candidate that may be abandoned. Score-copy is safe only when the chosen candidate immediately consumes ownership and updates `p_copy_phase_mask`.
- NCU skipped because policy12 proved no-op parity but did not create a scheduling win requiring NCU.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_candidate_owned_k_state_20260624_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_candidate_owned_k_state_20260624_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_candidate_owned_k_state_20260624_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

## Forward True Producer/Consumer Pipeline Rewrite: 2026-06-24

Report: `forward_true_pipeline_rewrite_20260624_report.md`.
Checkpoint: `forward_true_pipeline_rewrite_20260624_checkpoint.md`.
Combined summary: `forward_true_pipeline_rewrite_20260624_combined_summary.json`.

Implemented in `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`:

- Policy20 (`HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=20`): no-op ownership packaging through `FwdPipelineJob` for QK and PV/P issue. Preserves current issue order.
- Policy21 (`HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=21`): first committed producer/consumer pipeline. QK producer reserves jobs into a two-slot committed ring, advances K/K-scale ownership at reservation time, drains jobs without abandonment, and consumes score-copy completion only through committed jobs.
- No new hot-path full-CTA barrier was added.

Policy20 parity, timeline off, warmup `20`, iters `100`, GPU2:

- h4/s1024: full-stage2 `0.041680 ms`, policy20 `0.040608 ms`, `-2.57%`.
- h4/s2048: full-stage2 `0.059936 ms`, policy20 `0.061120 ms`, `+1.98%`.
- h16/s2048: full-stage2 `0.066832 ms`, policy20 `0.065824 ms`, `-1.51%`.
- h4/s4096: full-stage2 `0.097728 ms`, policy20 `0.095776 ms`, `-2.00%`.
- Decision: accepted as parity-clean enough to trust policy21; all finite and timeline-off raw/decoded `0/0`.

Policy21 timing, timeline off, warmup `20`, iters `100`, GPU2:

- h4/s1024: `0.040512 ms`, `-2.80%` vs full-stage2.
- h4/s2048: `0.058752 ms`, `-1.98%` vs full-stage2.
- h16/s2048: `0.066608 ms`, `-0.34%` vs full-stage2.
- h4/s4096: `0.099360 ms`, `+1.67%` vs full-stage2.
- First policy21 smoke initially hit a launch failure; root cause was reservation not advancing K/K-scale ownership, so two queued jobs could capture the same K slot/phase. Fixed by advancing `k_idx`/`k_phase`/`k_sc_slot` at policy21 reservation and skipping the second advance at queued issue.

Timeline-on result, warmup `5`, iters `10`, GPU2:

- Policy20 matched full-stage2 event counts: h4/s2048 `248`, h16/s2048 `248`, h4/s4096 `504`.
- Policy21 removed score-copy wait events: h4/s2048 `14 -> 0`, h16/s2048 `14 -> 0`, h4/s4096 `30 -> 0`.
- Policy21 QK wait dropped: h4/s2048 `2075 -> 860`, h16/s2048 `2091 -> 901`, h4/s4096 `2127 -> 928` cycles.
- Policy21 slot-release-to-next-QK spacing increased sharply: h4/s2048 `445 -> 1580`, h16/s2048 `510 -> 1632`, h4/s4096 `464 -> 1654.5` cycles. This explains the mixed timing and h4/s4096 regression.

NCU h4/s2048 comparison:

- Full-stage2: duration `48.640 us`, TC active `3.15%`, tensor pipe `1.45%`, issue active `7.53%`, eligible warps `0.39`, long scoreboard `3.42`, barrier `0.23`, regs/thread `168`, dynamic smem `98.0 KiB`.
- Policy21: duration `48.736 us`, TC active `3.10%`, tensor pipe `1.44%`, issue active `7.60%`, eligible warps `0.39`, long scoreboard `3.25`, barrier `0.22`, regs/thread `168`, dynamic smem `98.0 KiB`.
- NCU does not confirm the h4/s2048 p50 win as a one-launch duration win; policy21 lowers long scoreboard but slightly worsens duration and tensor activity.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_true_pipeline_rewrite_20260624_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_true_pipeline_rewrite_20260624_final_restore_shape_s2048_h4_smoke_gpu2_summary.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_true_pipeline_rewrite_20260624_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

Decision: partial success. Policy20 is a clean ownership base. Policy21 proves finite committed ownership and removes score-copy waits without unsafe probes, but the conservative delayed producer ring trades that for excessive producer spacing. Next target is a policy22 priority variant that issues immediately when score-slot pressure is low while preserving committed K/K-scale reservation semantics.

## Forward Policy22 Eager Committed Producer: 2026-06-25

Report: `forward_policy22_eager_committed_producer_20260625_report.md`.
Checkpoint: `forward_policy22_eager_committed_producer_20260625_checkpoint.md`.
Combined summary: `forward_policy22_eager_committed_producer_20260625_combined_summary.json`.

Implemented compile-gated `HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=22` in `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc` and added timeline decoder support for `qk_job_reserved`, producer skip reason events, and `qk_reserved_to_issue_cycles`.

Policy22 preserves policy21 committed ownership:

- K/K-scale ownership advances at committed reservation time.
- Reserved jobs drain through the committed path and are not abandoned.
- Queued issue skips a second K/K-scale advance.
- QK score-copy wait events remain removed by committed preconsume.

Timing-off results, warmup `20`, iters `100`, GPU2:

- h4/s1024: full-stage2 `0.041408 ms`, policy21 `0.041680 ms`, policy22 `0.046144 ms`, policy22 `+11.44%` vs full.
- h4/s2048: full-stage2 `0.059328 ms`, policy21 `0.059152 ms`, policy22 `0.061024 ms`, policy22 `+2.86%` vs full.
- h16/s2048: full-stage2 `0.065056 ms`, policy21 `0.065888 ms`, policy22 `0.066080 ms`, policy22 `+1.57%` vs full.
- h4/s4096: full-stage2 `0.095200 ms`, policy21 `0.099728 ms`, policy22 `0.102768 ms`, policy22 `+7.95%` vs full.

Timeline-on result, warmup `5`, iters `10`, GPU2:

- Score-copy wait events stayed removed for policy22: h4/s2048 `0`, h16/s2048 `0`, h4/s4096 `0`.
- Policy22 slot-release-to-next-QK worsened instead of improving:
  - h4/s2048: full `466.5`, policy21 `1654`, policy22 `1832.5` cycles.
  - h16/s2048: full `496.5`, policy21 `1604.5`, policy22 `1827` cycles.
  - h4/s4096: full `474.5`, policy21 `1605.5`, policy22 `1823.5` cycles.
- Policy22 reserved->QK issue was `1251`, `1293`, and `1269` cycles respectively, showing the current-job committed preconsume wait moved ahead of QK issue.
- Producer skip counters were all `0`; this implementation chose producer-first and did not yet implement a useful low-pressure skip predicate.

NCU skipped because policy22 did not improve p50 on any shape and the timeline failure is direct: score-copy pressure moved from QK score-copy wait events into pre-issue committed preconsume.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy22_eager_committed_producer_20260625_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy22_eager_committed_producer_20260625_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy22_eager_committed_producer_20260625_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

Decision: failure for policy22 as implemented. It is finite and preserves score-copy wait removal, but it does not reduce the target slot-release-to-next-QK gap. The next attempt needs a real committed low-pressure predicate instead of issuing a newly reserved job by waiting its score-copy completion before QK issue.

## Forward Policy23 Ready-Only Eager Producer: 2026-06-25

Report: `forward_policy23_ready_only_eager_producer_20260625_report.md`.
Checkpoint: `forward_policy23_ready_only_eager_producer_20260625_checkpoint.md`.
Combined summary: `forward_policy23_ready_only_eager_producer_20260625_combined_summary.json`.
NCU summary: `forward_policy23_ready_only_eager_producer_20260625_policy23_ncu_summary.json`.

Implemented compile-gated `HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=23` as a ready-only eager committed-producer variant. Policy23 preserves policy21 ownership semantics, adds nonblocking score-copy readiness via `score_copy_preconsumed(idx)` or successful `try_wait(p_copy_done[buf], phase)`, and avoids policy22's blocking current-QK score-copy preconsume.

Timing-off results, warmup `20`, iters `100`, GPU2:

- h4/s1024: full-stage2 `0.040400 ms`, policy21 `0.041456 ms`, policy22 `0.042528 ms`, policy23 `0.040688 ms`; policy23 `+0.71%` vs full, `-1.85%` vs policy21.
- h4/s2048: full-stage2 `0.059632 ms`, policy21 `0.064512 ms`, policy22 `0.065056 ms`, policy23 `0.061088 ms`; policy23 `+2.44%` vs full, `-5.31%` vs policy21.
- h16/s2048: full-stage2 `0.065360 ms`, policy21 `0.066928 ms`, policy22 `0.066128 ms`, policy23 `0.069424 ms`; policy23 `+6.22%` vs full, `+3.73%` vs policy21.
- h4/s4096: full-stage2 `0.097312 ms`, policy21 `0.096176 ms`, policy22 `0.097920 ms`, policy23 `0.100608 ms`; policy23 `+3.39%` vs full, `+4.61%` vs policy21.

Timeline-on result, warmup `5`, iters `10`, GPU2:

- Score-copy waits stayed removed for policy23: h4/s2048 `0`, h16/s2048 `0`, h4/s4096 `0`.
- Policy23 eager-ready fired for every reserved job: h4/s2048 `15/15`, h16/s2048 `15/15`, h4/s4096 `31/31`; fallback and skip counters were all `0`.
- Slot-release-to-next-QK worsened versus policy21 and policy22:
  - h4/s2048: policy21 `1660.0`, policy22 `1822.5`, policy23 `2010.0` cycles.
  - h16/s2048: policy21 `1640.0`, policy22 `1779.5`, policy23 `1980.0` cycles.
  - h4/s4096: policy21 `1613.5`, policy22 `1806.5`, policy23 `2004.0` cycles.
- Reserved-to-QK issue remained high: h4/s2048 `1468`, h16/s2048 `1434`, h4/s4096 `1441` cycles.

NCU policy23 timeline-off, GPU2:

- h4/s2048: duration `49.856 us`, SM throughput `7.49%`, tensor pipe `1.42%`, issue-active `32.46% active`, eligible warps `0.38`, long scoreboard `3.36`, barrier `0.22`, regs/thread `168`, dynamic smem `100352 B`.
- h4/s4096: duration `87.840 us`, SM throughput `15.22%`, tensor pipe `3.14%`, issue-active `33.25% active`, eligible warps `0.38`, long scoreboard `3.52`, barrier `0.18`, regs/thread `168`, dynamic smem `100352 B`.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy23_ready_only_eager_producer_20260625_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy23_ready_only_eager_producer_20260625_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy23_ready_only_eager_producer_20260625_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

Decision: failure as a performance policy. Policy23 is finite and avoids policy22's blocking score-copy preconsume, but score-copy-ready is not a sufficient eager issue predicate. It eagerly issues every reserved job and worsens the release-to-next-QK gap. The next target should be dynamic slot-state pressure control or a role-ordering rewrite that gates eager issue on more than score-copy readiness.

## Forward Policy24 Release-Triggered QK: 2026-06-25

Report: `forward_policy24_release_triggered_qk_20260625_report.md`.
Checkpoint: `forward_policy24_release_triggered_qk_20260625_checkpoint.md`.
Combined summary: `forward_policy24_release_triggered_qk_20260625_combined_summary.json`.
NCU summary: `forward_policy24_release_triggered_qk_20260625_policy24_ncu_summary.json`.

Implemented compile-gated `HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=24` as a release-triggered committed-QK scheduler. The release-signal audit found that for policy21-24 the timeline `hotplate_slot_release` and true score-slot reusable signal come from quant/P-materialization `arrive_score_copy_done` and `p_copy_done[slot]`, not the older PV-owned `hotplate_slot_reusable` path. Policy24 sets a shared per-slot release epoch after `p_copy_done` arrival and lets the QK producer poll it with volatile shared-memory reads, without adding a full CTA barrier.

Policy24 preserves policy21 committed ownership: K/K-scale ownership advances at reservation, queued issue skips the second advance, reserved jobs drain through the committed path, and no unsafe K/K-scale probe is used. It does not issue a newly reserved single job at reservation time. A release can trigger only the queued head job when the release slot matches `job.score_slot` and satisfies `job.idx - SCORE_TMEM_SLOTS`; otherwise policy24 falls back to policy21 delayed behavior. Policy25 was not implemented because policy24 fired, so the conservative match was not too restrictive.

Timing-off results, warmup `20`, iters `100`, GPU2:

- h4/s1024: full-stage2 `0.041872 ms`, policy21 `0.041760 ms`, policy23 `0.041008 ms`, policy24 `0.042160 ms`; policy24 `+0.69%` vs full, `+0.96%` vs policy21.
- h4/s2048: full-stage2 `0.059968 ms`, policy21 `0.063936 ms`, policy23 `0.059344 ms`, policy24 `0.061584 ms`; policy24 `+2.69%` vs full, `-3.68%` vs policy21.
- h16/s2048: full-stage2 `0.065088 ms`, policy21 `0.065696 ms`, policy23 `0.069184 ms`, policy24 `0.067536 ms`; policy24 `+3.76%` vs full, `+2.80%` vs policy21.
- h4/s4096: full-stage2 `0.096320 ms`, policy21 `0.098864 ms`, policy23 `0.097600 ms`, policy24 `0.102880 ms`; policy24 `+6.81%` vs full, `+4.06%` vs policy21.

Timeline-on result, warmup `5`, iters `10`, GPU2:

- Policy24 release-triggered issue fired: h4/s2048 `13/15`, h16/s2048 `13/15`, h4/s4096 `29/31`.
- Fallbacks were `1` per traced shape; mismatched-slot count was `0`.
- QK score-copy waits stayed removed: all traced policy24 shapes had score-copy wait count `0`.
- Slot-release-to-next-QK worsened sharply:
  - h4/s2048: policy21 `1593.0`, policy23 `2012.5`, policy24 `6272.5` cycles.
  - h16/s2048: policy21 `1660.0`, policy23 `1984.5`, policy24 `5691.0` cycles.
  - h4/s4096: policy21 `1587.5`, policy23 `1978.5`, policy24 `5856.5` cycles.
- Reserved-to-QK issue was also high: h4/s2048 `2514`, h16/s2048 `2709`, h4/s4096 `2559` cycles.

NCU policy24 timeline-off, GPU2:

- h4/s2048: duration `51.168 us`, SM throughput `7.45%`, tensor pipe `1.39%`, issue-active `32.66% active`, eligible warps `0.38`, long scoreboard `3.38`, barrier `0.25`, regs/thread `168`, dynamic smem `100352 B`.
- h4/s4096: duration `89.952 us`, SM throughput `15.03%`, tensor pipe `3.08%`, issue-active `33.32% active`, eligible warps `0.38`, long scoreboard `3.51`, barrier `0.21`, regs/thread `168`, dynamic smem `100352 B`.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy24_release_triggered_qk_20260625_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy24_release_triggered_qk_20260625_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy24_release_triggered_qk_20260625_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

Decision: failure as a performance policy. Policy24 proves the true release signal can be observed cheaply and can trigger queued QK issue, but it fires too late in the current role/order structure and worsens the target slot-release-to-next-QK gap. The next useful direction is a role-ordering rewrite or scheduler placement that lets the producer observe release immediately, not at a later delayed QK scheduling point.

## Forward Policy25 Release-Adjacent QK Tick: 2026-06-26

Report: `forward_policy25_release_adjacent_qk_tick_20260626_report.md`.
Checkpoint: `forward_policy25_release_adjacent_qk_tick_20260626_checkpoint.md`.

Implemented compile-gated `HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=25` as the nearest legal release-adjacent QK producer tick. The legal ownership audit found that QK TCGEN issue belongs to the issue warpgroup's QK lane (`qk_issue_lane`, timeline `threadIdx.x == 256`), while the true score-slot release path runs in quant/P-materialization `arrive_score_copy_done` after `p_copy_done[slot]`. Direct release-path QK issue is illegal because the quant path is the wrong warpgroup/lane and does not own the live QK producer state. The earliest legal existing placement is after the decoupled QK/PV branch in `run_iteration`, gated on `qk_issue_lane`.

Policy25 preserves policy21 committed ownership: K/K-scale ownership advances at reservation, queued issue uses the reserved job snapshot, queued jobs drain through the committed path, and the policy21 delayed drain remains finite fallback. The tick issues at most one queued committed job and requires a queued reserved job, an initial-safe or released score slot, and nonblocking score-copy preconsume readiness.

Timing-off results, warmup `20`, iters `100`, GPU2:

- h4/s1024: full-stage2 `0.041312 ms`, policy21 `0.041008 ms`, policy24 `0.041280 ms`, policy25 `0.042832 ms`; policy25 `+3.68%` vs full, `+4.45%` vs policy21.
- h4/s2048: full-stage2 `0.060928 ms`, policy21 `0.061344 ms`, policy24 `0.060608 ms`, policy25 `0.065088 ms`; policy25 `+6.83%` vs full, `+6.10%` vs policy21.
- h16/s2048: full-stage2 `0.068160 ms`, policy21 `0.067648 ms`, policy24 `0.071008 ms`, policy25 `0.072720 ms`; policy25 `+6.69%` vs full, `+7.50%` vs policy21.
- h4/s4096: full-stage2 `0.099760 ms`, policy21 `0.099104 ms`, policy24 `0.097984 ms`, policy25 `0.104080 ms`; policy25 `+4.33%` vs full, `+5.02%` vs policy21.

Timeline-on result, warmup `5`, iters `10`, GPU2:

- Policy25 tick reached/issued all queued jobs: h4/s2048 `15/15`, h16/s2048 `15/15`, h4/s4096 `31/31`.
- It recorded exactly one no-job tick per traced shape and zero slot-not-ready, score-copy-blocked, K-not-committed, or fallback-drain events.
- Score-copy waits stayed removed: all traced policy25 shapes had score-copy wait count `0`.
- Same-source slot-release-to-next-QK remained too late: policy25 h4/s2048 `5266.5`, h16/s2048 `5099.5`, h4/s4096 `5046.0` cycles, versus policy21 `4217.0`, `4001.0`, `4247.0` and policy24 `6282.0`, `5355.5`, `5487.0`.
- Reserved-to-QK remained high: h4/s2048 `2242`, h16/s2048 `2665`, h4/s4096 `2399` cycles.

NCU was not run because policy25 did not improve timing and the timeline was not ambiguous. Policy26 was not implemented because no earlier legal producer point was found without a broader role-ordering rewrite.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy25_release_adjacent_qk_tick_20260626_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy25_release_adjacent_qk_tick_20260626_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy25_release_adjacent_qk_tick_20260626_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

Decision: failure as a performance policy, partial success as a diagnostic. Policy25 proves the nearest legal QK-producer tick can drain queued committed jobs without score-copy waits, but the current legal placement is still too far from the true score release. The next direction is a deeper role-ordering rewrite or a producer-side state transition closer to quant release.

## Forward Policy26 Producer-Owned Release Handoff: 2026-06-27

Report: `forward_policy26_producer_owned_release_handoff_20260627_report.md`.
Checkpoint: `forward_policy26_producer_owned_release_handoff_20260627_checkpoint.md`.

Implemented compile-gated `HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=26` as a producer-owned release handoff. The feasibility audit found one legal bounded-wait location: the QK producer lane after a committed QK job is reserved and the committed ring has pressure. This lane can poll the release epoch without blocking quant/P release, and the head job already owns its committed K/K-scale snapshots. The wait uses `2048` polling iterations and falls back to policy21 delayed drain.

Policy26 preserves policy21 committed ownership: K/K-scale ownership advances at reservation, queued issue uses the reserved job snapshot, no second owner advance occurs, and reserved jobs drain through finite fallback. It reuses the policy24/25 release epoch from `arrive_score_copy_done` and adds timeline events for wait enter, matched, timeout, already present, score-copy blocked, fallback drain, no job, and not committed.

Timing-off results, warmup `20`, iters `100`, GPU2:

- h4/s1024: full-stage2 `0.040128 ms`, policy21 `0.042048 ms`, policy25 `0.043600 ms`, policy26 `0.040560 ms`; policy26 `+1.08%` vs full, `-3.54%` vs policy21.
- h4/s2048: full-stage2 `0.060320 ms`, policy21 `0.061264 ms`, policy25 `0.064608 ms`, policy26 `0.061792 ms`; policy26 `+2.44%` vs full, `+0.86%` vs policy21.
- h16/s2048: full-stage2 `0.067664 ms`, policy21 `0.068464 ms`, policy25 `0.072032 ms`, policy26 `0.074304 ms`; policy26 `+9.81%` vs full, `+8.53%` vs policy21.
- h4/s4096: full-stage2 `0.098864 ms`, policy21 `0.098976 ms`, policy25 `0.105392 ms`, policy26 `0.101184 ms`; policy26 `+2.35%` vs full, `+2.23%` vs policy21.

Timeline-on result, warmup `5`, iters `10`, GPU2:

- Producer wait entered on every future-release head job: h4/s2048 `14`, h16/s2048 `14`, h4/s4096 `30`.
- Wait matched release while waiting: `0` on all traced shapes.
- Wait timed out: `0` on all traced shapes.
- Release was already present at wait entry: h4/s2048 `14`, h16/s2048 `14`, h4/s4096 `30`.
- Score-copy waits stayed removed: all traced policy26 shapes had score-copy wait count `0`.
- Slot-release-to-next-QK remained late: policy26 h4/s2048 `5033.0`, h16/s2048 `5185.0`, h4/s4096 `5266.5` cycles, versus policy21 `4100.5`, `4002.0`, `4266.5` and policy25 `5041.0`, `4929.0`, `5014.5`.
- Reserved-to-QK was high: h4/s2048 `5369`, h16/s2048 `5201`, h4/s4096 `5467` cycles.

NCU was not run because policy26 did not improve major-shape p50 versus full-stage2/policy21, did not improve h4/s4096, and the timeline was not ambiguous. Policy27/28 were not implemented because release was already present at wait entry; longer/shorter wait bounds would not fix placement.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy26_producer_owned_release_handoff_20260627_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy26_producer_owned_release_handoff_20260627_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy26_producer_owned_release_handoff_20260627_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

Decision: failure as a handoff policy, useful as a feasibility diagnostic. Policy26 is legal and finite, but the selected producer-owned wait point is still too late: every entered wait sees release already present, so it cannot reduce the true release-to-QK gap. A deeper role-ordering change must get the legal QK producer to the committed head before quant/P release occurs.

## Forward Policy27 Early Score-Slot Release: 2026-06-27

Report: `forward_policy27_early_score_slot_release_20260627_report.md`.
Checkpoint: `forward_policy27_early_score_slot_release_20260627_checkpoint.md`.

Implemented compile-gated `HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=27` as a diagnostic-only early score-unused marker. The score-slot lifetime audit found that the active scorepack/prescaled/direct-rescale path loads score TMEM through `fp4pv_load_score_half_tmem` / `fp4pv_load_score_quarter_tmem` and waits with `tensor_load_wait()` before later work uses registers/P-stage/P-scale/output state. However, it did not prove a clean common marker earlier than `arrive_score_copy_done(...)` across active and adjacent compile-time paths, so policy27 marks the score slot immediately before the existing full release and QK does not consume it.

Policy27 timing-off results, warmup `20`, iters `100`, GPU2:

- h4/s1024: full-stage2 `0.040688 ms`, policy21 `0.042128 ms`, policy26 `0.041952 ms`, policy27 `0.041920 ms`.
- h4/s2048: full-stage2 `0.059648 ms`, policy21 `0.061216 ms`, policy26 `0.064544 ms`, policy27 `0.059536 ms`.
- h16/s2048: full-stage2 `0.065952 ms`, policy21 `0.067488 ms`, policy26 `0.071760 ms`, policy27 `0.067984 ms`.
- h4/s4096: full-stage2 `0.098464 ms`, policy21 `0.096368 ms`, policy26 `0.102544 ms`, policy27 `0.098208 ms`.

Policy27 timeline-on diagnostics, warmup `5`, iters `10`, GPU2:

- Marker counts matched expected releases: h4/s2048 `16/16`, h16/s2048 `16/16`, h4/s4096 `32/32`.
- Early-unused to full release was only `143.0`, `145.5`, and `145.0` cycles.
- Full release to next QK was `4261.0`, `4468.5`, and `4547.0` cycles.
- Early-unused to next QK was `4400.0`, `4626.0`, and `4693.0` cycles, slightly larger because the marker is before the same full-release event.
- Early reuse consumed/fallback counts were `0` because policy28/29 were not implemented.

Policy28/29 were hard-gated off. The measured recoverable window at the only proven marker is about 145 cycles, not a meaningful early score-slot reuse point relative to the 4k+ cycle release-to-QK gap. NCU was not run because no early reuse policy was enabled and policy27 did not create a >1% improvement or ambiguous profile.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy27_early_score_slot_release_20260627_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy27_early_score_slot_release_20260627_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy27_early_score_slot_release_20260627_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`.

Decision: partial diagnostic success, no early-reuse implementation. Policy27 proves coverage and shows the currently proven marker is effectively the existing release point. A useful next step needs a source-level marker immediately after all score TMEM loads and `tensor_load_wait()` that is proven common across the active path, or a role-ordering rewrite that makes such a marker safely consumable by the QK producer.

## Forward Policy28 Source Last Score-Load Marker: 2026-06-28

Report: `forward_policy28_source_last_score_load_marker_20260628_report.md`.
Checkpoint: `forward_policy28_source_last_score_load_marker_20260628_checkpoint.md`.

Implemented compile-gated `HOTPLATE_SLOT_SCHED=1 HOTPLATE_POLICY=28` as a diagnostic-only source-level marker. The active config is `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix`, mapped in `fwd_host_dispatch.inc` to the direct scorepack/prescaled/direct-rescale specialization. The active score path uses direct `tcgen05.ld.sync.aligned.32x32b.x32.b32` loads into `scores_reg`, followed by `tensor_load_wait()`. Policy28 marks `qk_source_score_unused` immediately after that wait and before `arrive_score_copy_done(idx, buf)`. QK does not consume the marker.

Policy28 timeline-on diagnostics, warmup `5`, iters `10`, GPU2:

- h4/s2048: source markers/full releases `16/16`, source-to-full-release `139.5` cycles, full-release-to-next-QK `4544.0` cycles, source-to-next-QK `4689.0` cycles, finite `true`.
- h16/s2048: source markers/full releases `16/16`, source-to-full-release `143.0` cycles, full-release-to-next-QK `4324.0` cycles, source-to-next-QK `4464.5` cycles, finite `true`.
- h4/s4096: source markers/full releases `32/32`, source-to-full-release `141.0` cycles, full-release-to-next-QK `4460.0` cycles, source-to-next-QK `4606.5` cycles, finite `true`.

Policy29 was not implemented. The marker count and placement gates passed for the active branch, but the reuse-window gate failed: the marker is only about `140` cycles before current full release, below the `>= 500` cycle bar and not meaningful against the `4.3k-4.5k` cycle release-to-next-QK gap.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy28_source_last_score_load_marker_20260628_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy28_source_last_score_load_marker_20260628_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy28_source_last_score_load_marker_20260628_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

Decision: diagnostic success, no reuse implementation. The active branch's current full release is already nearly adjacent to the final score TMEM load/wait. The next target is not another score-unused marker; it is the role ordering or scheduler path that leaves the legal QK producer roughly `4k+` cycles behind release.

## Forward Policy30 Release-to-QK Scheduler Lag: 2026-06-28

Report: `forward_policy30_release_to_qk_scheduler_lag_20260628_report.md`.
Checkpoint: `forward_policy30_release_to_qk_scheduler_lag_20260628_checkpoint.md`.

Implemented `HOTPLATE_POLICY=30` as a diagnostic-only release-to-QK scheduler trace. Added producer-loop, active-role, candidate-seen, head-ready/not-ready, release-seen, issue-attempt, issue-blocked, and issue-success events. The trace keeps policy21 committed ownership and does not change score-slot reuse or QK issue order.

Policy30 timeline-on diagnostics, warmup `5`, iters `10`, GPU2:

- h4/s2048: release-to-next-QK `6884.0` cycles, release-to-producer-loop `430.0`, release-to-candidate `635.5`, release-to-head-ready `5596.5`, head-ready-to-issue `1303.0`, dominant issue depth `q2` `14/15`.
- h16/s2048: release-to-next-QK `7468.0` cycles, release-to-producer-loop `491.0`, release-to-candidate `666.5`, release-to-head-ready `6016.5`, head-ready-to-issue `1445.0`, dominant issue depth `q2` `14/15`.
- h4/s4096: release-to-next-QK `7159.0` cycles, release-to-producer-loop `462.0`, release-to-candidate `672.5`, release-to-head-ready `5805.0`, head-ready-to-issue `1319.5`, dominant issue depth `q2` `30/31`.

Policy30 result: release visibility is not the bottleneck. The legal QK producer sees the release in about `0.43k-0.49k` cycles and sees a committed candidate in about `0.64k-0.67k` cycles. The large gap is that the released-slot future QK is not the committed FIFO head until about `5.6k-6.0k` cycles after release. No `qk_producer_head_not_ready` events were observed.

Implemented `HOTPLATE_POLICY=31` as the only low-risk tweak identified by Policy30: if the committed queue has one release-ready head, drain it before reserving the next candidate. It preserves FIFO legality and policy21 ownership.

Policy31 timing-off, warmup `20`, iters `100`, GPU2:

- h4/s1024: `0.041760 ms` median, `0.039520 ms` min; policy21 baseline `0.041008/0.039296`.
- h4/s2048: `0.060032 ms` median, `0.057696 ms` min; policy21 baseline `0.061344/0.057344`.
- h16/s2048: `0.068736 ms` median, `0.066304 ms` min; policy21 baseline `0.067648/0.064768`.
- h4/s4096: `0.099456 ms` median, `0.097312 ms` min; policy21 baseline `0.099104/0.093632`.

Policy31 timeline-on showed mostly worse release-to-next-QK:

- h4/s2048: policy30 `6884.0` cycles, policy31 `7203.0`.
- h16/s2048: policy30 `7468.0` cycles, policy31 `8195.0`.
- h4/s4096: policy30 `7159.0` cycles, policy31 `7334.5`.

Decision: diagnostic success, Policy31 negative result. The remaining target is role ordering or a dynamic slot-state scheduler that avoids placing the released-slot future QK behind older FIFO work without reintroducing score-copy backpressure.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy30_release_to_qk_scheduler_lag_20260628_final_restore_build_defaultoff_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy30_release_to_qk_scheduler_lag_20260628_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy30_release_to_qk_scheduler_lag_20260628_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy32 Ready-Set Scheduler: 2026-06-28

Report: `forward_policy32_ready_set_scheduler_20260628_report.md`.
Checkpoint: `forward_policy32_ready_set_scheduler_20260628_checkpoint.md`.

Implemented `HOTPLATE_POLICY=32` as a diagnostic-only shadow ready-set scheduler. It added ready-set timeline events for candidate insert, release-ready, shadow issueable/pick, FIFO-head blocking, actual issue, pool full, order constraint, real pick, and fallback. Policy32 does not change QK issue order.

Policy32 timeline-on diagnostics, warmup `5`, iters `10`, GPU2:

- h4/s2048: finite `true`, raw/decoded `459/459`, candidate inserts `15`, FIFO blocks ready `14`, shadow release->pick `2288.0` cycles, actual release->next-QK `9368.0` cycles, estimated headroom `6985.5` cycles.
- h16/s2048: finite `true`, raw/decoded `459/459`, candidate inserts `15`, FIFO blocks ready `14`, shadow release->pick `2392.0` cycles, actual release->next-QK `10329.5` cycles, estimated headroom `7619.5` cycles.
- h4/s4096: finite `true`, raw/decoded `939/939`, candidate inserts `31`, FIFO blocks ready `30`, shadow release->pick `2307.5` cycles, actual release->next-QK `9507.0` cycles, estimated headroom `7085.0` cycles.

Policy32 hard gates passed for a real ready-set attempt: release-ready non-head candidates were blocked in nearly every steady-state opportunity, the estimated headroom was `~7k` cycles, the existing two-entry pool was enough to expose the blocker, K/K-scale ownership was captured in the committed job record, and selection occurred in the legal QK producer lane.

Implemented `HOTPLATE_POLICY=33` as a conservative real ready-set tail issue experiment. It preserves policy21 committed ownership, only issues from the captured job record, and blocks same-score-slot tail bypass. The timing-off build succeeded, but the real policy failed a minimal h4/s1024 GPU2 smoke with `TimeoutError: Timed out while waiting for timeline direct prealloc timing to complete after 5000 ms`. The earlier timing matrix attempts produced no JSON artifacts and empty logs, so they were invalid timing data. Policy33 did not reach valid timing or timeline profiling.

Decision: shadow diagnostic success, real QK-only ready-set issue failure. The bottleneck is confirmed as FIFO head-of-line blocking, but QK issue order cannot be changed alone because the downstream PV/score-ready consumer remains FIFO-sensitive. The next move is a removable/compacting committed job table or role-ordering rewrite that lets the whole committed job state transition, including PV consumption, follow the selected ready job.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy32_ready_set_scheduler_20260628_restore_default_off_build_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy32_ready_set_scheduler_20260628_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy32_ready_set_scheduler_20260628_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy34 Whole-Job Promotion Scheduler: 2026-06-28

Report: `forward_policy34_whole_job_promotion_20260628_report.md`.
Checkpoint: `forward_policy34_whole_job_promotion_20260628_checkpoint.md`.

Implemented `HOTPLATE_POLICY=34` as a diagnostic-only shadow whole-job promotion scheduler. It added whole-job timeline events `87-97`, decoded them in `forward_overlap_loop_20260622_timeline_driver.py`, and did not change real QK issue order. An initial hook placement produced zero candidates because the post-reservation sample was nested under the Policy32 ready-set compile gate; after moving the Policy34 hook outside that gate, the corrected diagnostics exposed the intended signal.

Policy34 timeline-on diagnostics, GPU2:

- h4/s2048, warmup `5`, iters `10`: finite `true`, raw/decoded `444/444`, candidates `14`, release-ready `14`, score/PV hazard rejects `14`, global head-rotation rejects `14`, shadow picks `0`, release->wholejob-ready `2185.5` cycles, actual release->QK `9331.0` cycles.
- h16/s2048, warmup `5`, iters `10`: finite `true`, raw/decoded `444/444`, candidates `14`, release-ready `14`, score/PV hazard rejects `14`, global head-rotation rejects `14`, shadow picks `0`, release->wholejob-ready `2533.5` cycles, actual release->QK `10671.5` cycles.
- h4/s4096 required warmup `5`, iters `10`: timed out after `240s` with no JSON, not accepted as timing data.
- h4/s4096 short, warmup `1`, iters `3`: finite `true`, raw/decoded `908/908`, candidates `30`, release-ready `30`, score/PV hazard rejects `30`, global head-rotation rejects `30`, shadow picks `0`, release->wholejob-ready `2256.5` cycles, actual release->QK `9701.0` cycles.

Policy34 gate decision: failed for Policy35. Release-ready non-head jobs exist, but every one is rejected by hard whole-lifecycle hazards: the committed QK queue is not the owner of PV consumption or score-slot release. A real promotion would need QK issue, score-ready transition, PV issue, score-copy completion, and score-slot release to consume the same promoted job. Policy35 was not implemented; repeating Policy33's QK-only tail pick would be invalid.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `forward_policy34_whole_job_promotion_20260628_restore_default_off_build_gpu2.log`.
- Final GPU2 smoke:
  - `forward_policy34_whole_job_promotion_20260628_final_restore_shape_s2048_h4_smoke_gpu2.log`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy34_whole_job_promotion_20260628_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy36 Lifecycle Ownership Rewrite Path: 2026-06-28

Report: `forward_policy36_lifecycle_ownership_rewrite_20260628_report.md`.
Checkpoint: `forward_policy36_lifecycle_ownership_rewrite_20260628_checkpoint.md`.

Implemented `HOTPLATE_POLICY=36` as a diagnostic-only lifecycle correlation pass. It added lifecycle timeline events `98-109`, decoded them in `forward_overlap_loop_20260622_timeline_driver.py`, and did not change scheduling. The diagnostic correlates committed QK queue jobs with independently prepared `iter_job` PV/release state.

Policy36 timeline-on diagnostics, GPU2:

- h4/s2048, warmup `5`, iters `10`: finite `true`, raw/decoded `525/525`, selected tails `14`, PV ownership blockers `14`, release ownership blockers `14`, same selected job `0`, selected tail vs latest `iter_job` idx gap `+2`, release->selected-tail `2333` cycles, selected-tail->PV-issue-begin `11914.5` cycles, slot release->next QK `9295.5` cycles.
- h16/s2048, warmup `5`, iters `10`: finite `true`, raw/decoded `525/525`, selected tails `14`, PV ownership blockers `14`, release ownership blockers `14`, same selected job `0`, selected tail vs latest `iter_job` idx gap `+2`, release->selected-tail `2766` cycles, selected-tail->PV-issue-begin `13410.5` cycles, slot release->next QK `10955` cycles.
- h4/s4096 short, warmup `1`, iters `3`: failed before JSON/CSV artifact creation. The task command was silent until killed after several minutes; a bounded rerun with 60s CUDA event timeout raised `TimeoutError: Timed out while waiting for timeline direct prealloc timing to complete after 60000 ms`.

Policy36 gate decision: failed for Policy37. The selected release-ready tail is consistently a future same-slot QK job, not the current PV/release-owned job. `selected_tail_with_latest_iter_match_count` was `0`; `selected_tail_with_latest_iter_mismatch_count` was `14`. A real prototype must make one lifecycle table entry own QK issue, score-ready, PV issue, score-copy completion, and score-slot release. Policy37 was not implemented.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy36_lifecycle_ownership_rewrite_20260628_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy36_lifecycle_ownership_rewrite_20260628_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy38 Lifecycle Table Prototype: 2026-06-29

Report: `forward_policy38_lifecycle_table_prototype_20260629_report.md`.
Checkpoint: `forward_policy38_lifecycle_table_prototype_20260629_checkpoint.md`.

Implemented `HOTPLATE_POLICY=38` as a diagnostic-only lifecycle table prototype. It added a compile-gated depth-4 shared lifecycle table, events `110-124`, and decoder summary `timeline_lifecycle_table_summary`. The table records QK reserve/issue/score-ready, iter/PV prepare and issue, score-copy done, score-slot release, selected-tail table hit/miss, PV/release readiness, occupancy, overwrite hazard, same-score-slot hazard, and retire markers. It does not change scheduling.

Policy38 did not pass the gate. The required timeline-on build command with `timeout 900s` timed out twice. An extended `timeout 1800s` recovery build succeeded, but the required h4/s2048 trace hung without producing JSON. Minimal h4/s1024 bounded runs failed before and after a nested `warp::elect_leader()` fix with `TimeoutError: Timed out while waiting for timeline direct prealloc timing to complete after 5000 ms`. h16/s2048 was not attempted because h4 did not pass the minimal runtime gate.

Policy39 was not implemented. The table did not produce accepted h4/h16 evidence for hit/miss, occupancy, overwrite, same-score-slot, PV readiness, or release readiness behavior, so it cannot yet be claimed safe for representing the current PV/release owner and selected QK tail simultaneously.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy38_lifecycle_table_prototype_20260629_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy38_lifecycle_table_prototype_20260629_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy40 Lifecycle Table Bisect And Repair: 2026-06-29

Report: `forward_policy40_lifecycle_table_bisect_repair_20260629_report.md`.
Checkpoint: `forward_policy40_lifecycle_table_bisect_repair_20260629_checkpoint.md`.

Implemented staged diagnostic policies `HOTPLATE_POLICY=40..45` to bisect the unstable Policy38 lifecycle table:

- Policy40: skeleton table, no updates.
- Policy41: QK reserve/issue/score-ready updates.
- Policy42: iter prepare marker.
- Policy43: PV issue marker.
- Policy44: score-copy done and score-slot release updates.
- Policy45: selected-tail classification.

The failing sites were isolated and repaired:

- The original full iter table update created excessive timeline-on compile pressure. It was replaced with `lifecycle_table_mark_iter_prepared()`, an exact-entry, append-only marker on `threadIdx.x == 288`.
- The original full PV table update caused a minimal h4/s1024 CUDA-event timeout. It was replaced with `lifecycle_table_mark_pv_issued()`, also exact-entry and append-only on `threadIdx.x == 288`.
- `tk_fa4/fp4_fa4_fwd/Makefile` now exposes `NVCC_THREADS ?= 4` and `NVCC_SPLIT_COMPILE ?= 4` instead of hard-coded serial nvcc flags, allowing the required 900s timeline-on builds to complete without changing policy semantics.

Minimal h4/s1024 GPU2 stage smokes all passed after repair:

- Policy40: finite `true`, raw/decoded `253/253`, occupancy `0`.
- Policy41: finite `true`, raw/decoded `275/275`, qk inserts `7`, max occupancy `4`, overwrite hazards `3`, same-score hazards `5`.
- Policy42: finite `true`, raw/decoded `293/293`, iter updates `8`, overwrite hazards `3`, same-score hazards `8`.
- Policy43: finite `true`, raw/decoded `304/304`, PV updates `8`, overwrite hazards `3`, same-score hazards `11`.
- Policy44: finite `true`, raw/decoded `321/321`, score-copy/release updates `8/8`, retire `8`, overwrite hazards `0`, same-score hazards `6`.
- Policy45: finite `true`, raw/decoded `339/339`, selected-tail hit/miss `6/0`, tail PV-ready `0/6`, tail release-ready `0/6`.

Accepted Policy45 diagnostics, GPU2:

- h4/s2048, warmup `5`, iters `10`: finite `true`, raw/decoded `707/707`, median/min `0.171088/0.165152` ms, qk inserts `15`, iter/PV/release updates `16/16/16`, selected-tail hit/miss `14/0`, max occupancy `4`, overwrite hazards `0`, same-score hazards `14`, tail PV-ready `0/14`, tail release-ready `0/14`, release-to-selected-tail `5260` cycles, release-to-next-QK `19213` cycles.
- h16/s2048, warmup `5`, iters `10`: finite `true`, raw/decoded `707/707`, median/min `0.237856/0.236192` ms, qk inserts `15`, iter/PV/release updates `16/16/16`, selected-tail hit/miss `14/0`, max occupancy `4`, overwrite hazards `0`, same-score hazards `14`, tail PV-ready `0/14`, tail release-ready `0/14`, release-to-selected-tail `6138` cycles, release-to-next-QK `22869.5` cycles.

Policy46 was not attempted. The repaired table is stable and useful as a diagnostic, but the hard gates failed: same-score-slot hazards remain and the selected tail still lacks PV/release readiness, so there is no explicit ownership path for a real scheduler.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `results/mxfp4_fa4_forward_recover_20260629_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy40_lifecycle_table_bisect_repair_20260629_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy46 Two-Identity Score-Slot Scheduler: 2026-06-29

Report: `forward_policy46_two_identity_score_slot_scheduler_20260629_report.md`.
Checkpoint: `forward_policy46_two_identity_score_slot_scheduler_20260629_checkpoint.md`.

Implemented `HOTPLATE_POLICY=46` as a diagnostic two-identity score-slot lifecycle model with separate current-owner and future-candidate identities per score slot. The owner side is updated only by iter/PV/score-copy/release code; the candidate side is updated only by QK reservation/issue/score-ready and selected-tail classification. Added timeline events `125..141` and decoder summary `timeline_two_identity_summary`.

Policy46 passed the required h4/s2048 and h16/s2048 gates on GPU2:

- h4/s2048: finite `true`, raw/decoded `704/704`, median/min `0.151408/0.148416` ms, selected hit/missing `14/0`, legal/illegal handoff `14/0`, owner mismatch `0`, candidate overwrite `0`, release-to-legal `3908.5` cycles, release-to-issue `12875.0` cycles.
- h16/s2048: finite `true`, raw/decoded `704/704`, median/min `0.191872/0.188928` ms, selected hit/missing `14/0`, legal/illegal handoff `14/0`, owner mismatch `0`, candidate overwrite `0`, release-to-legal `3979.5` cycles, release-to-issue `15413.0` cycles.

Conclusion: the remaining same-score-slot conflict is a valid `current owner -> future candidate` handoff, not an illegal overlap/overwrite.

Because Policy46 gates passed, attempted `HOTPLATE_POLICY=47` as a conservative table-backed handoff issue prototype. Policy47 compiled with extended `timeout 1800s`, but the required h4/s1024 minimal smoke failed with `TimeoutError: Timed out while waiting for timeline direct prealloc timing to complete after 5000 ms`, then the outer command exited `124`. No h4/h16 s2048 Policy47 profiling was attempted.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy46_two_identity_score_slot_scheduler_20260629_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy46_two_identity_score_slot_scheduler_20260629_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy47 Handoff-Issue Bisect: 2026-06-30

Report: `forward_policy47_handoff_issue_bisect_20260630_report.md`.
Checkpoint: `forward_policy47_handoff_issue_bisect_20260630_checkpoint.md`.

Implemented staged handoff-issue bisect policies `HOTPLATE_POLICY=48..52` while preserving the Policy46 two-identity model and leaving old failed Policy47 available:

- Policy48: read-only handoff legality query at the ready-set decision point.
- Policy49: candidate-ready diagnostic mark only.
- Policy50: read-only would-bypass accounting with forced fallback.
- Policy51: single-shot actual handoff issue/removal.
- Policy52: extra preissue-only split, same one-shot/in-flight bookkeeping as Policy51 but forced fallback before actual QK issue and ready-set tail removal.

Results on GPU2:

- Policy48 h4/s1024: finite `true`, raw/decoded `405/405`, legal/illegal query `6/0`, fallback `6`, stale owner `0`.
- Policy48 h4/s2048: finite `true`, raw/decoded `861/861`, median/min `0.194224/0.192064` ms, legal/illegal query `14/0`.
- Policy48 h16/s2048: finite `true`, raw/decoded `861/861`, median/min `0.232832/0.229408` ms, legal/illegal query `14/0`.
- Policy49 h4/s1024: finite `true`, raw/decoded `411/411`, candidate-ready marks `6`, owner mismatch `0`, candidate overwrite `0`.
- Policy50 h4/s1024: finite `true`, raw/decoded `423/423`, would-bypass `6`, fallback-after-would `6`, stale owner `0`.
- Policy51 h4/s1024: extended build succeeded, but runtime timed out with `TimeoutError: Timed out while waiting for timeline direct prealloc timing to complete after 5000 ms`; outer command exited `124`; no JSON was written.
- Policy52 h4/s1024: finite `true`, raw/decoded `417/417`, preissue-only `1`, would-bypass `6`, single-shot block `5`, stale owner `0`, owner mismatch `0`, candidate overwrite `0`.

Conclusion: legal handoff query, candidate-ready publication, would-bypass classification, one-shot guard, in-flight marker, and preissue fence are all finite. The smallest failing diff is the actual tail handoff issue/removal path in Policy51: `issue_committed_fwd_pipeline_qk_job(tail_job)`, followed by `tail_job.state = FWD_PIPELINE_JOB_EMPTY` and `--qk_pipeline_count`. Larger h4/s2048 and h16/s2048 scheduler profiles were not attempted because Policy51 failed the h4/s1024 gate.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy47_handoff_issue_bisect_20260630_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy47_handoff_issue_bisect_20260630_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy53 Shadow Handoff Scheduler: 2026-06-30

Report: `forward_policy53_shadow_handoff_scheduler_20260630_report.md`.
Checkpoint: `forward_policy53_shadow_handoff_scheduler_20260630_checkpoint.md`.

Implemented `HOTPLATE_POLICY=53` as a shadow-copy dry-run and `HOTPLATE_POLICY=54` as the first real shadow issue prototype while preserving the Policy46 two-identity model. The shadow path records events `154..163` and the decoder reports `timeline_shadow_handoff_summary`. The prototype does not directly remove the future tail from the original ready-set and does not decrement `qk_pipeline_count` for shadow issue.

Ready-set audit result: committed QK jobs are managed as a two-entry FIFO ring. Normal drain issues the head, clears it, rotates `qk_pipeline_head`, and decrements `qk_pipeline_count`. Policy47/51 had failed by issuing/removing a future tail out of order. Policy53/54 split that cluster by first proving shadow-copy construction finite, then trying copied-job issue without direct tail removal/count mutation.

Results on GPU2:

- Policy53 h4/s1024: finite `true`, raw/decoded `399/399`, median/min `441.3741455078125/441.3741455078125` ms, shadow candidates/copies/fallbacks `6/6/0`, owner mismatch `0`, candidate overwrite `0`, legal/illegal handoff `6/0`, release-to-legal `4872.5` cycles, normal release-to-issue `17816.0` cycles.
- Policy54 build with `timeout 900s` timed out at compile (`nvcc: Terminated`, `make` error `255`); rebuild with `timeout 1800s` succeeded.
- Policy54 h4/s1024: failed the minimal gate. The driver raised `TimeoutError: Timed out while waiting for timeline direct prealloc timing to complete after 5000 ms`, then the outer command exited `124`; no Policy54 JSON artifact was written.

Conclusion: shadow copy is safe and finite, but shadow issue is not finite even without direct ready-set tail removal or `qk_pipeline_count` decrement. The smallest failing stage found here is the copied-job `issue_committed_fwd_pipeline_qk_job(shadow_job)` call itself. This shifts the blocker from pure FIFO count/head corruption toward QK issue/barrier ownership or duplicate outstanding identity semantics. Policy55/56 duplicate cleanup was not profiled because Policy54 did not pass h4/s1024.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.

## Forward Policy125 Middle Deferred-Preconsume Repair: 2026-07-03

Report: `forward_policy125_middle_deferred_preconsume_repair_20260703_report.md`.
Checkpoint: `forward_policy125_middle_deferred_preconsume_repair_20260703_checkpoint.md`.

Implemented compile-gated Policy125 deferred-preconsume legality diagnostics for the middle handoff score/preconsume blocker:

- Added middle deferred-preconsume timeline events `365..373`.
- Added Policy125 legality probes at the confirmed default/head point after one-head capacity repair.
- Proved the later normal duplicate/scheduling point exists without issuing or suppressing the QK.
- Updated the timeline driver summary/deltas for defer possible, normal consume possible, registered, and consumed events.

GPU2 Policy125 timeline results:

- h4/s1024: finite `true`, confirmed default/head `3`, capacity repaired `3`, defer possible `3`, normal consume point `3`, already preconsumed `0`, phase ready `0`, impossible `0`, sanity `0`.
- h4/s2048: finite `true`, confirmed default/head `12`, capacity repaired `12`, defer possible `12`, normal consume point `12`, already preconsumed `0`, phase ready `0`, impossible `0`, sanity `0`.
- Timing medians: h4/s1024 release->defer possible `10666` cycles and defer possible->normal point `959` cycles; h4/s2048 release->defer possible `10371` cycles and defer possible->normal point `1000` cycles.

Policy126 behavior attempts:

- Direct explicit-owned issue with `preconsume_idx=-1` plus manual deferred registration timed out at compile after `1800s` with `nvcc: Terminated`, make error `255`.
- Smaller wrapper-reuse attempt using the existing Policy104 `append_and_issue_explicit_owned_second_qk_job()` path also timed out after `1800s` in `cicc` with `nvcc: Terminated`, make error `255`.

Conclusion: stop condition 3. Policy125 proves deferred preconsume is legal at the middle handoff point, but Policy126 behavior is blocked by compile-time/code-size before profiling. The current source leaves the finite Policy125 diagnostic path and gates Policy126 behavior off (`STATIC_ONLINE_MXFP4_MIDDLE_DEFER_PRECONSUME_BEHAVIOR = false`).

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.

## Forward Policy120 Middle Explicit QK Handoff: 2026-07-02

Report: `forward_policy120_middle_explicit_qk_handoff_20260702_report.md`.
Checkpoint: `forward_policy120_middle_explicit_qk_handoff_20260702_checkpoint.md`.

Implemented compile-gated middle handoff diagnostics and a guarded Policy122 behavior attempt:

- Policy120 records the pre-release lookahead intent at the producer-entry poll.
- Policy121 confirms a saved intent at score-slot release and classifies it at the next producer poll.
- Policy122 attempts a confirmed default/head explicit-owned issue only after the confirmed intent has become the normal default/head.

GPU2 timeline results:

- Policy120 h4/s1024: finite `true`, producer polls/pre-intents `6/2`, release confirm disabled.
- Policy120 h4/s2048: finite `true`, producer polls/pre-intents `14/2`, release confirm disabled.
- Policy121 h4/s1024: finite `true`, pre-intent/release-confirmed/confirmed-next-poll/default-head `5/5/5/5`, release->next-poll median `1201` cycles, pre-intent->release median `8228` cycles, sanity `0`.
- Policy121 h4/s2048: finite `true`, pre-intent/release-confirmed/confirmed-next-poll/default-head `12/11/11/11`, release->next-poll median `1127` cycles, pre-intent->release median `9054` cycles, sanity `0`.
- Policy122 direct h4/s1024: finite `true`, confirmed default/head `3`, explicit issue begin/done `0/0`, capacity fallback `3`.
- Policy122 one-head-drain h4/s1024: finite `true`, confirmed default/head `5`, explicit issue begin/done `0/0`, score/preconsume fallback `5`.

Matched Stage2 reference timing from same-day artifacts:

- h4/s1024 p50/min `0.042832/0.039904` ms from `forward_policy104_viability_20260702_base_stage2_h4_s1024_gpu2.json`.
- h4/s2048 p50/min `0.061904/0.060096` ms from `forward_policy104_viability_20260702_base_stage2_h4_s2048_gpu2.json`.

Conclusion: stop condition 3. The middle handoff itself works and is early: saved pre-release intents are confirmed at release and become default/head at the next producer poll. However, Policy122 cannot legally issue in the current nonblocking producer path. The direct path is blocked by QK FIFO capacity; draining the immediately prior normal head removes that blocker but exposes a concrete score/preconsume readiness blocker. The deferred-preconsume prototype timed out in compile and was reverted, leaving the finite nonblocking fallback path in source.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy53_shadow_handoff_scheduler_20260630_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy53_shadow_handoff_scheduler_20260630_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy59 Head-Only Handoff: 2026-06-30

Report: `forward_policy59_head_only_handoff_20260630_report.md`.
Checkpoint: `forward_policy59_head_only_handoff_20260630_checkpoint.md`.

Implemented staged head-only handoff policies `HOTPLATE_POLICY=59..62` while preserving the Policy46 two-identity score-slot model:

- Policy59: read-only FIFO-head position diagnostic.
- Policy60: head-only would-issue accounting with forced fallback.
- Policy61: actual issue only when the legal handoff candidate is already the normal committed FIFO head.
- Policy62: larger-shape profiling of the finite head-only issue path.

Results on GPU2:

- Policy59 h4/s1024: finite `true`, raw/decoded `412/412`, head/non-head hits `1/6`, no legal `0`, issue `0`.
- Policy59 h4/s2048: finite `true`, raw/decoded `879/879`, median/min `0.200208/0.196960` ms, head/non-head hits `4/14`.
- Policy59 h16/s2048: finite `true`, raw/decoded `883/883`, median/min `0.217792/0.217152` ms, head/non-head hits `8/14`.
- Policy60 h4/s1024: finite `true`, raw/decoded `414/414`, would/fallback-after-would `1/1`.
- Policy61 h4/s1024: finite `true`, raw/decoded `414/414`, issue begin/done/fail `1/1/0`.
- Policy62 h4/s2048: finite `true`, raw/decoded `884/884`, median/min `0.213296/0.208960` ms, head/non-head hits `3/14`, issue done/fail `3/0`.
- Policy62 h16/s2048: finite `true`, raw/decoded `902/902`, median/min `0.249088/0.246240` ms, head/non-head hits `9/14`, issue done/fail `9/0`.

All Policy59-62 runs kept the two-identity gates clean: illegal-before-release `0`, owner mismatch `0`, candidate overwrite `0`, selected missing `0`.

Conclusion: legal handoff candidates are sometimes the normal FIFO head, and actual issue is finite in that case. This explains the previous Policy47/51 and Policy54 failures as non-head/copied-job QK issue semantics rather than score-slot legality failures. The head-only path does not improve performance: Policy62 h4/s2048 `0.213296/0.208960` ms is slower than Policy48 h4/s2048 `0.194224/0.192064` ms and Policy46 h4/s2048 `0.151408/0.148416` ms; Policy62 h16/s2048 `0.249088/0.246240` ms is slower than Policy48 h16/s2048 `0.232832/0.229408` ms and Policy46 h16/s2048 `0.191872/0.188928` ms.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy59_head_only_handoff_20260630_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy59_head_only_handoff_20260630_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy63 Head-By-Construction Scheduler: 2026-07-01

Report: `forward_policy63_head_by_construction_scheduler_20260701_report.md`.
Checkpoint: `forward_policy63_head_by_construction_scheduler_20260701_checkpoint.md`.

Implemented staged head-by-construction policies `HOTPLATE_POLICY=63..67` while preserving the Policy46 two-identity model:

- Policy63: diagnostic-only safe pre-issue reorder-window classification.
- Policy64: dry-run post-reorder accounting.
- Policy65: local metadata reconstruction dry-run.
- Policy66: timeout-debug split, real FIFO slot swap followed by immediate restore and fallback drain.
- Policy67: persistent real FIFO swap so the promoted legal candidate becomes the normal FIFO head and is issued only by existing normal drain.

Results on GPU2:

- Policy63 h4/s1024: finite `true`, raw/decoded `406/406`, safe reorder `5`, non-head legal `6`, final-Q rejects `1`.
- Policy63 h4/s2048: finite `true`, raw/decoded `863/863`, median/min `0.202544/0.199456` ms, safe reorder `13`, non-head legal `14`, final-Q rejects `1`.
- Policy63 h16/s2048: finite `true`, raw/decoded `872/872`, median/min `0.230592/0.228544` ms, safe reorder `13`, non-head legal `14`, final-Q rejects `1`.
- Policy64 h4/s1024: finite `true`, raw/decoded `421/421`, would-swap/would-become-head/predicted-issue `5/5/5`.
- Policy65 h4/s1024: finite `true`, raw/decoded `426/426`, sanity ok/fail `5/0`.
- Policy66 h4/s1024: finite `true`, raw/decoded `442/442`, real swap begin/done/fail `5/5/0`, normal issue `0`, fallback `5`.
- Policy67 h4/s1024: failed the minimal gate. The driver raised `TimeoutError: Timed out while waiting for timeline direct prealloc timing to complete after 5000 ms`; the outer command exited `124`, and no Policy67 JSON was emitted.

Conclusion: a local pre-issue metadata reorder window exists, and real in-place FIFO slot mutation is finite when restored before issue. The smallest failing behavioral mutation is leaving the swapped order live so the legal future candidate becomes `qk_pipeline_head` and the existing normal drain issues it. Larger h4/s2048/h16 profiling was not attempted because Policy67 failed h4/s1024. The next viable scheduler point is upstream at reservation/producer selection, before QK owner cursors and score-copy/K/K-scale side effects are bound to the original FIFO order.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy63_head_by_construction_scheduler_20260701_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy63_head_by_construction_scheduler_20260701_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy69 Upstream Reservation Roadmap: 2026-07-01

Report: `forward_policy69_upstream_reservation_roadmap_20260701_report.md`.
Checkpoint: `forward_policy69_upstream_reservation_roadmap_20260701_checkpoint.md`.

Implemented diagnostic-only upstream reservation policies while preserving the Policy46 two-identity model:

- Policy69: reservation-time candidate knowability.
- Policy71: side-effect binding timeline and release horizon, covering the Policy70/71 observability goals.
- Policy74: would-select, candidate-owned resource dry-run, and deferred-binding dry-run, covering the Policy72/73/74 goals.

Results on GPU2:

- Policy69 h4/s1024: finite `true`, raw/decoded `388/388`, reserve/id/release-known `7/7/7`, tail-blocked/opportunity `6/5`.
- Policy69 h4/s2048: finite `true`, raw/decoded `820/820`, median/min `0.164192/0.160576` ms, reserve/id/release-known `15/15/15`, tail-blocked/opportunity `14/13`.
- Policy69 h16/s2048: finite `true`, raw/decoded `820/820`, median/min `0.205456/0.201632` ms, reserve/id/release-known `15/15/15`, tail-blocked/opportunity `14/13`.
- Policy71 h4/s1024: finite `true`, raw/decoded `444/444`, bind append/K cursor/score-copy/twoid/lifecycle/issue all `7`, release-before/after-reserve `7/0`.
- Policy74 h4/s1024: finite `true`, raw/decoded `465/465`, would-select/head `5/0`, reject-not-owned/defer-blocked `5/5`, actual-select `0`.
- Policy74 h4/s2048: finite `true`, raw/decoded `999/999`, median/min `0.189328/0.186688` ms, would-select/head `13/0`, reject-not-owned/defer-blocked `13/13`, actual-select `0`.
- Policy74 h16/s2048: finite `true`, raw/decoded `1004/1004`, median/min `0.221680/0.211232` ms, would-select/head `13/0`, reject-not-owned/defer-blocked `13/13`, actual-select `0`.

Conclusion: the future candidate is knowable and release-known at the current reservation call, but all useful opportunities are already tail-blocked behind an existing bound FIFO head. K/K-scale cursor state and score-copy/preconsume state bind in original reservation order. Phase 3 behavior was not attempted because the dry-run gates show no safe small current behavior case; the next viable direction is a larger upstream producer-selection intent/bind split that chooses reservation order before those side effects bind.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy69_upstream_reservation_roadmap_20260701_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy69_upstream_reservation_roadmap_20260701_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy80 Producer Intent/Bind Split: 2026-07-01

Report: `forward_policy80_producer_intent_bind_split_20260701_report.md`.
Checkpoint: `forward_policy80_producer_intent_bind_split_20260701_checkpoint.md`.

Implemented diagnostic-only producer intent/bind split policies while preserving the Policy46 two-identity model:

- Policy80: pre-bind producer intent enumeration.
- Policy81: selected-intent dry-run.
- Policy82: bind-plan dry-run.

Results on GPU2:

- Policy80 h4/s1024: finite `true`, raw/decoded `363/363`, query/default/lookahead/desired `7/7/6/6`.
- Policy80 h4/s2048: finite `true`, raw/decoded `762/762`, median/min `0.143712/0.141184` ms, query/default/lookahead/desired `15/15/14/14`.
- Policy80 h16/s2048: finite `true`, raw/decoded `763/763`, median/min `0.189168/0.186880` ms, query/default/lookahead/desired `15/15/14/14`.
- Policy81 h4/s1024: finite `true`, raw/decoded `381/381`, desired `6`, predicted-head `1`, sequence-gap rejects `6`, selected dry-run `0`, default chosen `7`.
- Policy82 h4/s1024: finite `true`, raw/decoded `417/417`, bind-plan complete/missing `6/0`, sequence-gap rejects `6`, selected dry-run `0`, actual selected bind `0`.

Conclusion: the desired future candidate can be represented before resource binding only as a synthetic one-step lookahead. Resource fields can be planned, but selecting that lookahead would skip the still-required default QK work and create a sequence gap. Policy83+ behavior was not attempted because the h4/s1024 hard gate failed before mutation. The next viable direction is a real upstream unbound-intent window that keeps both default and desired future intents alive before binding.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy80_producer_intent_bind_split_20260701_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy80_producer_intent_bind_split_20260701_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy90 Unbound Intent Window: 2026-07-01

Report: `forward_policy90_unbound_intent_window_20260701_report.md`.
Checkpoint: `forward_policy90_unbound_intent_window_20260701_checkpoint.md`.

Implemented diagnostic-only unbound intent window policies while preserving the Policy46 two-identity model:

- Policy90: two-intent unbound window enumeration.
- Policy91: lookahead-first ordering legality dry-run.
- Policy92: two-step bind-order plan dry-run plus Phase 4 alternative diagnostics.

Results on GPU2:

- Policy90 h4/s1024: finite `true`, raw/decoded `386/386`, window/default/lookahead `6/7/6`, preserved-default `6`, missing fields `0`.
- Policy90 h4/s2048: finite `true`, raw/decoded `818/818`, median/min `0.173024/0.166688` ms, window/default/lookahead `14/15/14`, preserved-default `14`.
- Policy90 h16/s2048: finite `true`, raw/decoded `818/818`, median/min `0.208928/0.207456` ms, window/default/lookahead `14/15/14`, preserved-default `14`.
- Policy91 h4/s1024: finite `true`, raw/decoded `416/416`, order dry-run `6`, legal `0`, rejects default-before-lookahead/K-cursor/score-copy `6/6/6`, FIFO capacity `5`, final-Q/causal `1`.
- Policy92 h4/s1024: finite `true`, raw/decoded `455/455`, bind lookahead/default plans `6/6`, two-step complete/missing `0/6`, depth-3 present `5`, default-first-second plan `1`, rewrite boundary required `6`.

Conclusion: a local unbound default/lookahead window is representable and both bind plans are computable, but lookahead-first remains illegal. The exact blockers are default-before-lookahead QK sequence order, K/K-scale cursor ownership order, and score-copy/preconsume ownership order. Policy93+ behavior was not attempted because h4/s1024 hard gates failed before mutation.

Final state:

- Restored active build:
  - `timeout 900s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 smoke:
  - `forward_policy90_unbound_intent_window_20260701_final_restore_shape_s2048_h4_smoke_gpu2.json`
  - finite `true`, timeline raw/decoded `0/0`.
- Direct import verification:
  - `forward_policy90_unbound_intent_window_20260701_final_restore_direct_timeline_read_gpu2.json`
  - `read_mxfp4_forward_timeline()` length `0`, raw count `0`.

## Forward Policy100 Explicit QK Ownership Scheduler: 2026-07-02

Report: `forward_policy100_explicit_qk_ownership_scheduler_20260702_report.md`.
Checkpoint: `forward_policy100_explicit_qk_ownership_scheduler_20260702_checkpoint.md`.

Implemented compile-gated explicit QK ownership policies under `HOTPLATE_SLOT_SCHED=1`:

- Policy100: explicit-owned normal-head no-op issue path.
- Policy101: persistent default/lookahead owned identities, original issue order; simultaneous two-job resource binding records K/K-scale cursor alias fallback.
- Policy102: default-first immediate-second attempt; finite but completed `0` second jobs because immediate score-copy preconsume was not ready.
- Policy104: deferred-preconsume default-first immediate-second behavior; finite and completes real second jobs.

Results on GPU2:

- Policy100 h4/s1024: finite `true`, raw/decoded `371/371`, owned query/default/issue `7/7/7`, fields missing `0`.
- Policy100 h4/s2048: finite `true`, raw/decoded `779/779`, owned query/default/issue `15/15/15`, fields missing `0`.
- Policy101 h4/s1024: finite `true`, raw/decoded `396/396`, lookahead/queue/original-order `6/6/6`, K fallback `6`, fields missing `0`.
- Policy101 h4/s2048: finite `true`, raw/decoded `836/836`, lookahead/queue/original-order `14/14/14`, K fallback `14`, fields missing `0`.
- Policy102 h4/s1024: finite `true`, raw/decoded `404/404`, attempts/completions `6/0`, score/lifecycle fallback `5/1`.
- Policy104 h4/s1024 timeline: finite `true`, raw/decoded `347/347`, attempts/completions `3/3`, capacity/score/lifecycle fallback `0/0/0`, sanity `0`.
- Policy104 h4/s2048 timeline: finite `true`, raw/decoded `719/719`, attempts/completions `7/7`, capacity/score/lifecycle fallback `0/0/0`, sanity `0`.
- Policy104 no-timeline timing: h4/s1024 p50/min `0.044288/0.042560` ms; h4/s2048 p50/min `0.063504/0.061408` ms.
- Restored default/off timing: h4/s1024 p50/min `0.041840/0.040256` ms; h4/s2048 p50/min `0.062416/0.059680` ms.

Conclusion: explicit per-job ownership is implementable and Policy104 proves finite real behavior, but it is not a wall-time win (`~5.9%` slower p50 at h4/s1024 and `~1.7%` slower p50 at h4/s2048 versus restored default/off). Larger h16/s2048, h4/s4096, and NCU were skipped because the finite behavior policy did not meet the improvement gate. The concrete blocker found and repaired was score-copy/preconsume order: non-deferred Policy102 cannot immediately issue the second job because the future preconsume identity is not ready at that point.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off timing:
  - `forward_policy100_explicit_qk_ownership_scheduler_20260702_defaultoff_h4_s1024_timing_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`.
  - `forward_policy100_explicit_qk_ownership_scheduler_20260702_defaultoff_h4_s2048_timing_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`.

## Forward Policy104 Viability Check: 2026-07-02

Report: `forward_policy104_viability_check_20260702_report.md`.
Checkpoint: `forward_policy104_viability_check_20260702_checkpoint.md`.

Implemented targeted Policy104 viability diagnostics:

- Added second-QK timing-position events for selected, release matched, issue begin/done, deferred preconsume registered/consumed, normal duplicate point, and disabled-second ablation.
- Added Policy107 as the check-only ablation: Policy104 path reaches legal second selection/release match, then disables the actual second issue.
- Extended the timeline driver to pair the relevant score-slot release with second-QK selected/begin/done, default-QK done, deferred preconsume, and normal-point events.

Matched timeline-off timing on GPU2, warmup/iters `5/20`:

- h4/s1024: base_stage0 `0.041344/0.038784` ms, base_stage2 `0.042832/0.039904` ms, Policy100 `0.045904/0.044224` ms, Policy101 `0.045520/0.043648` ms, Policy104 `0.042848/0.042016` ms, Policy107 `0.045008/0.042432` ms.
- h4/s2048: base_stage0 `0.061552/0.058496` ms, base_stage2 `0.061904/0.060096` ms, Policy100 `0.065392/0.063776` ms, Policy101 `0.064064/0.061152` ms, Policy104 `0.064240/0.062336` ms, Policy107 `0.062832/0.060896` ms.

Policy104 h4/s2048 is `+3.77%` p50 slower than the matched `KPIPE_STAGE=2` baseline, so h16/s2048, h4/s4096, and NCU were skipped by the task gate.

Timing-position diagnostic on Policy104 h4/s2048:

- Explicit counters: second selected/release matched/issue begin/issue done `7/7/7/7`, deferred preconsume registered/consumed `7/7`, normal duplicate point `7`, sanity fail `0`.
- Median deltas: release -> selected `6696` cycles, release -> issue begin `14709` cycles, release -> issue done `19633` cycles, default QK done -> second issue begin `389` cycles, second issue begin -> done `5032` cycles, second done -> normal point `866` cycles, release -> normal point `20390` cycles, second done -> deferred preconsume consumed `1358` cycles.

Conclusion: Policy104 is finite and correctness-useful, but not a plausible performance winner in the current post-reservation placement. The second QK completes only `866` cycles before the normal schedule point, while the behavior path costs `+0.001408 ms` over the check-only Policy107 at h4/s2048. The next viable move is to move explicit-owned binding/issue earlier into the producer-loop/release handoff, or retain Policy104 as a correctness primitive.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy104_viability_20260702_final_restore_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`.

## Forward Policy110 Early Explicit QK Handoff: 2026-07-02

Report: `forward_policy110_early_explicit_qk_handoff_20260702_report.md`.
Checkpoint: `forward_policy110_early_explicit_qk_handoff_20260702_checkpoint.md`.

Implemented compile-gated early handoff diagnostics:

- Policy110: release-side latch at the same release/epoch path used by Policy104, with candidate index `release_idx + SCORE_TMEM_SLOTS` and late Policy104-style selection-point timing.
- Policy111: earliest QK producer-entry poll before normal reservation, validating the release latch against the current default/lookahead window.
- Policy112/113 were not attempted because Policy111 failed the hard gate.

GPU2 timeline results:

- Policy110 h4/s1024: finite `true`, raw/decoded `418/418`, release latch/candidate `8/6`, later Policy104-style selection `6`, release->selection median `12606` cycles.
- Policy110 h4/s2048: finite `true`, raw/decoded `884/884`, release latch/candidate `16/14`, later Policy104-style selection `14`, release->selection median `12498` cycles.
- Policy111 h4/s1024: finite `true`, raw/decoded `431/431`, producer poll `6`, candidate derived `0`, no-candidate/not-lookahead fallback `3/4`, release->producer_poll median `1071` cycles.
- Policy111 h4/s2048: finite `true`, raw/decoded `911/911`, producer poll `14`, candidate derived `0`, no-candidate/not-lookahead fallback `3/12`, release->producer_poll median `1146.5` cycles.

Matched timeline-off Stage2 baseline on current source, GPU2, warmup/iters `5/20`:

- h4/s1024 p50/min `0.044368/0.042912` ms.
- h4/s2048 p50/min `0.060176/0.058016` ms.

Conclusion: stop condition 3. The producer-entry hook is earlier in clock time, but it cannot legally carry the intended one-step lookahead candidate. The release latch becomes valid after the producer poll where that candidate was lookahead; at the first post-release producer poll the same candidate is already default/head, while the current lookahead has not been released. Policy112 was therefore not eligible.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.

## Forward Policy126 Compile Repair: 2026-07-03

Report: `forward_policy126_compile_repair_20260703_report.md`.
Checkpoint: `forward_policy126_compile_repair_20260703_checkpoint.md`.

Implemented compile-time Policy126 sub-gates using `KPIPE_SELECTIVE_POLICY=1261..1267`:

- 1261 behavior reaches confirmed default/head and returns;
- 1262 constructs the explicit-owned job;
- 1263 issues with `preconsume_idx=-1`;
- 1264 registers deferred preconsume;
- 1265 suppresses the duplicate normal schedule point;
- 1266 consumes deferred preconsume at the normal point;
- 1267 adds timeline events.

Compile bisection:

- Policy125 timeline baseline: success, `real 1103.98`.
- Policy126 behavior off with timeline on: success, `real 1010.05`.
- Policy126 stages 1261..1266 with timeline off: all success; final 1266 rebuild `real 462.73`.
- Policy126 stage 1267 with timeline on: 1800 s timeout, `nvcc: Terminated`, make error 255.
- Reduced-event stage 1267 alternative: still stuck in `cicc`; terminated at `real 1334.02`, `nvcc: Terminated`, make error 255.

Finite behavior:

- Stage1266 h4/s1024 GPU2 smoke: finite `true`, raw/decoded timeline `0/0`.
- Stage1266 h4/s2048 GPU2 smoke: finite `true`, raw/decoded timeline `0/0`.

Matched GPU2 timing, warmup/iters `5/20`:

- Stage2 baseline h4/s1024 p50/min `0.043664/0.041952` ms.
- Policy126 stage1266 h4/s1024 p50/min `0.047856/0.045600` ms, `+9.60%` p50.
- Stage2 baseline h4/s2048 p50/min `0.060672/0.058592` ms.
- Policy126 stage1266 h4/s2048 p50/min `0.070256/0.068128` ms, `+15.80%` p50.

Conclusion: Policy126 now has a finite behavior build through stage1266, but the smallest remaining compiler blocker is stage1267 timeline-on instrumentation. The finite behavior is not competitive with matched Stage2, so larger profiling was skipped.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `real 316.71`.
- Final GPU2 default/off smoke:
  - `forward_policy126_compile_repair_final_restore_h4_s1024_gpu2_summary.json`: finite `true`, raw/decoded `0/0`.

## Forward Policy126 Overhead Optimization: 2026-07-03

Report: `forward_policy126_overhead_optimization_20260703_report.md`.
Checkpoint: `forward_policy126_overhead_optimization_20260703_checkpoint.md`.

Added cheap timeline-off Policy126 counters and two optimized sub-stages:

- `KPIPE_SELECTIVE_POLICY=1268`: compact lookahead/stale bookkeeping, avoiding full `FwdPipelineJob` snapshots for diagnostic-only lookahead/stale payloads in timeline-off builds.
- `KPIPE_SELECTIVE_POLICY=1269`: additionally delays the full default/head job snapshot until a confirmed pending intent is actually default/head.
- `STATIC_ONLINE_MXFP4_MIDDLE126_ENABLE_TIMELINE_EXTRA` is now `stage == 1267`, so stage1268/1269 do not inherit the known stage1267 timeline-on compile blocker.

Stage timing matrix, GPU2, warmup/iters `5/20`, timeline off:

- Stage2 baseline:
  - h4/s1024 p50/min `0.042320/0.040480` ms.
  - h4/s2048 p50/min `0.062496/0.060576` ms.
- Policy126 stage1261:
  - h4/s1024 p50/min `0.044336/0.042784` ms, `+0.002016` vs baseline.
  - h4/s2048 p50/min `0.070944/0.068032` ms, `+0.008448` vs baseline.
- Policy126 stage1262:
  - h4/s1024 p50/min `0.045744/0.044224` ms.
  - h4/s2048 p50/min `0.068944/0.066688` ms.
- Policy126 stage1263:
  - h4/s1024 runtime timed out/empty JSON; h4/s2048 skipped.
- Policy126 stage1264:
  - h4/s1024 runtime timed out/empty JSON.
  - h4/s2048 p50/min `0.067984/0.066560` ms.
- Policy126 stage1265:
  - h4/s1024 p50/min `0.046128/0.044640` ms.
  - h4/s2048 p50/min `0.073712/0.071936` ms.
- Policy126 stage1266:
  - h4/s1024 p50/min `0.045936/0.043456` ms.
  - h4/s2048 p50/min `0.070352/0.068256` ms.

Cheap counter validation, stage1266, GPU2:

- h4/s1024: confirmed default/head `16`, explicit issue begin/done `16/16`, defer registered `16`, duplicate suppressed `16`, defer consumed `16`, stale `12`, all fallbacks/sanity `0`.
- h4/s2048: confirmed default/head `45`, explicit issue begin/done `45/45`, defer registered `45`, duplicate suppressed `45`, defer consumed `45`, stale `226`, all fallbacks/sanity `0`.

Optimized variants, GPU2, warmup/iters `5/20`:

- Stage1268:
  - h4/s1024 p50/min `0.045696/0.043744` ms.
  - h4/s2048 p50/min `0.068512/0.066432` ms.
- Stage1269:
  - h4/s1024 p50/min `0.045712/0.043808` ms.
  - h4/s2048 p50/min `0.068272/0.066176` ms.
- Matched Stage2 after compact-source rebuild:
  - h4/s1024 p50/min `0.042848/0.040992` ms.
  - h4/s2048 p50/min `0.060640/0.058144` ms.

Conclusion: stop condition 1/2. Stage1269 improves h4/s2048 by `0.002080` ms (`2.96%`) versus stage1266, but remains `+12.59%` versus matched Stage2. The issue/defer/suppress/consume counters match exactly, so the remaining loss is not deferred preconsume or duplicate suppression. The overhead source is the middle intent/scaffolding path visible at stage1261, especially stale intent churn (`271` observed intents, `226` stale at h4/s2048). Next single code move should be normal-path integration: confirmed default/head intent becomes the normal QK issue path with scalar per-slot state, instead of a sidecar middle-intent window.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `real 294.31`.
- Final GPU2 default/off smoke:
  - `forward_policy126_overhead_restored_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`.

## Forward Policy126 Normal-Path Integration: 2026-07-03

Report: `forward_policy126_normal_path_integration_20260703_report.md`.
Checkpoint: `forward_policy126_normal_path_integration_20260703_checkpoint.md`.

Implemented compile-gated Policy126 normal-path integration variants `KPIPE_SELECTIVE_POLICY=1270..1274`.

- Policy1270 release-epoch-only normal issue was finite but overtriggered: h4/s2048 issued/deferred/consumed `364` normal-path jobs.
- Policy1271 confirmed one-shot normal-ready token was finite but still reintroduced middle stale churn: h4/s2048 `normal_path_direct_issue_done=199`, `confirmed_seen_next_poll=312`, `confirmed_state_stale=113`.
- Policy1272 release-epoch plus prior-head ownership removed sidecar stale churn and retained useful opportunities: h4/s2048 `normal_path_direct_issue_done=52`, `normal_prior_head_drained=52`, `normal_fallback_capacity=48`, `normal_fallback_final=8`.
- Policy1273 counters-off is the best behavior policy: h4/s1024 p50/min `0.042608/0.039872` ms and h4/s2048 p50/min `0.063600/0.061120` ms.

Matched GPU2 timing, warmup/iters `5/20`:

- Stage2 baseline: h4/s1024 `0.042576/0.040896` ms, h4/s2048 `0.059456/0.057952` ms.
- Stage1269 sidecar baseline: h4/s1024 `0.044704/0.042688` ms, h4/s2048 `0.068480/0.067200` ms.
- Policy1273: h4/s2048 is `7.13%` faster than stage1269, but still `6.97%` slower than matched Stage2.

Conclusion: normal-path integration is a real improvement over sidecar polishing, but the current 1273 formulation does not plausibly win outright without reducing branch/register/control footprint or making the prior-head handoff a first-class normal queue operation.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success, `real 307.84`.
- Final GPU2 default/off smoke:
  - `forward_policy126_normal_path_restored_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Policy1275 Control-Overhead Optimization: 2026-07-05

Report: `forward_policy1275_control_overhead_optimization_20260705_report.md`.
Checkpoint: `forward_policy1275_control_overhead_optimization_20260705_checkpoint.md`.

Implemented `HOTPLATE_POLICY=126 KPIPE_SELECTIVE_POLICY=1275` as a first-class prior-head handoff inside `schedule_committed_fwd_pipeline_qk_job(...)`, after normal `reservation_job` construction and before normal append. The old policy1270-1274 `schedule_qk_candidate(...)` normal-path branch is now gated off for 1275+.

Added queue-handoff counters and a 1276 counters-off variant:

- 1275 h4/s1024 smoke: finite `true`, raw/decoded `0/0`, queue handoffs `20`, fallback capacity `16`, fallback final `8`.
- 1275 h4/s2048 smoke: finite `true`, raw/decoded `0/0`, queue handoffs `52`, fallback capacity `48`, fallback final `8`.
- 1276 smokes: finite `true`, raw/decoded `0/0`, all counters `0`.

Matched GPU2 timing, warmup/iters `5/20`, timeline off:

- Stage2: h4/s1024 `0.042832/0.040480` ms, h4/s2048 `0.059856/0.058048` ms.
- Policy1273 matched: h4/s1024 `0.046032/0.043456` ms, h4/s2048 `0.065600/0.063424` ms.
- Policy1275 counters on: h4/s1024 `0.044000/0.041280` ms, h4/s2048 `0.063888/0.062848` ms.
- Policy1276 counters off: h4/s1024 `0.045264/0.043008` ms, h4/s2048 `0.065152/0.062848` ms.

Build-pressure notes:

- Stage2, 1273, 1275, and 1276 selected `kernel_fp4pv` ptxas lines all showed `128` registers, `24` bytes stack, `32` spill-store bytes, `92` spill-load bytes, and `1280` bytes smem.
- The build logs did not expose a selected-kernel ptxas distinction that explains the remaining delta.

Conclusion: Policy1275 is a real same-source improvement over 1273 on h4/s2048 (`-2.61%` p50), but still `+6.73%` versus matched Stage2 and not a robust win against the older 2026-07-03 1273 number. Stop here: the next useful move is deeper normal queue-state simplification, not another sidecar scheduler or counter-stripping tweak.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy1275_restored_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Policy1277 Native Queue-Pair Handoff: 2026-07-06

Report: `forward_policy1277_native_queue_pair_handoff_20260706_report.md`.
Checkpoint: `forward_policy1277_native_queue_pair_handoff_20260706_checkpoint.md`.

Implemented native two-entry queue-pair handoff:

- `1277`: after normal append, queue drain helper recognizes `[prior head, current tail]`, drains prior normally, then issues/removes current with deferred-current-preconsume semantics.
- `1278`: same behavior with counters compiled out.
- 1275/1276 before-append handoff remains gated to 1275/1276 only; 1277+ does not use the old before-append predicate chain or `schedule_qk_candidate(...)` sidecar.

Counter evidence, 1277 smoke:

- h4/s1024: native pair seen/match `64`, release-ready/useful handoffs `20`, fallback final `20`, fallback lifecycle `24`.
- h4/s2048: native pair seen/match `368`, release-ready/useful handoffs `52`, fallback final `52`, fallback lifecycle `264`.
- 1278 smokes: finite, raw/decoded `0/0`, all counters `0`.

Matched GPU2 timing, warmup/iters `5/20`, timeline off:

- Stage2: h4/s1024 `0.043504/0.041088` ms, h4/s2048 `0.063152/0.061600` ms.
- Policy1275: h4/s1024 `0.044288/0.042752` ms, h4/s2048 `0.064096/0.062080` ms.
- Policy1277 counters on: h4/s1024 `0.044496/0.043168` ms, h4/s2048 `0.065552/0.063488` ms.
- Policy1278 counters off: h4/s1024 `0.042768/0.041184` ms, h4/s2048 `0.061984/0.060160` ms.

Conclusion: 1278 is the first successful native queue-pair result. It beats matched 1275 by `3.30%` on h4/s2048 and beats the matched Stage2 p50 by `1.85%`. The counters-on 1277 variant is slow because it instruments all adjacent/fallback pair observations (`368` seen for only `52` useful h4/s2048 handoffs).

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy1277_restored_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Policy1280 Native-Pair Follow-Up Optimizations: 2026-07-06

Report: `forward_policy1280_native_pair_followups_20260706_report.md`.
Checkpoint: `forward_policy1280_native_pair_followups_20260706_checkpoint.md`.

Reproduced Policy1278 first against matched Stage2 on GPU2, timeline off, warmup/iters `5/20`:

- Stage2: h4/s1024 `0.043088/0.040128` ms, h4/s2048 `0.060768/0.058656` and repeat `0.061904/0.059616` ms.
- Policy1278: h4/s1024 `0.043056/0.041888` ms, h4/s2048 `0.064672/0.063040` and repeat `0.064256/0.062208` ms.

Implemented native-pair follow-ups without returning to sidecar prediction:

- 1280/1281: release-ready latch.
- 1282/1283: append-time candidacy scalar.
- 1284/1285: release-pending event-triggered drain.
- 1286: stripped fast helper preserving critical queue/release/preconsume/issue checks.

Counter evidence:

- 1280 h4/s2048 kept the old shape: `368` seen, `52` useful handoffs, lifecycle fallback `264`.
- 1282 h4/s2048 changed behavior: `294` seen, `126` useful handoffs, but p50 `0.066000` ms.
- 1284 h4/s2048 reduced failed checks: `80` seen, `76` useful handoffs, but p50 `0.070064` ms.
- Counters-off variants report all counters zero by construction.

Best follow-up:

- Policy1286 counters off: h4/s1024 `0.044608/0.042304` and repeat `0.044576/0.042400` ms; h4/s2048 `0.064576/0.063264` and repeat `0.064448/0.062688` ms.

Conclusion: no follow-up beat matched Stage2, and 1286 only tied reproduced 1278 within noise while adding queue-state bookkeeping. Failed-pair checks are not the decisive bottleneck; the next useful move is normal queue/control simplification or earlier ownership selection that makes the handoff the normal head path.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy1280_restored_smoke_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Reproduce Fast Policy1278: 2026-07-06

Report: `forward_reproduce_fast_1278_20260706_report.md`.
Checkpoint: `forward_reproduce_fast_1278_20260706_checkpoint.md`.

Forensics-only task. No new algorithms were implemented.

Current matched GPU2 repeat matrix, timeline off, warmup/iters `5/20`:

- Stage2 h4/s2048: `0.062512/0.060288`, `0.059936/0.057856`, `0.061760/0.059968` ms.
- Policy1278 h4/s2048: `0.069152/0.066880`, `0.066496/0.064544`, `0.064592/0.063008` ms.
- Stage2 h4/s1024: `0.042688/0.040384` ms.
- Policy1278 h4/s1024: `0.047392/0.045120` ms.

Reverse-order check:

- Policy1278 first: h4/s2048 `0.064352/0.061152` ms.
- Stage2 second: h4/s2048 `0.061136/0.058464` ms.

Historical comparison:

- Historical fast Policy1278 artifact: `forward_policy1278_timing_h4_s2048_gpu2_stdout.json`, `0.061984/0.060160` ms.
- 1280 rerun 1278 artifacts: `0.064672/0.063040` and `0.064256/0.062208` ms.
- Current 1278 matches the slower 1280 rerun band, not the fast historical band.

Source drift finding:

- 1280/1282/1284/1286 behavior gates are compile-time false for `KPIPE_SELECTIVE_POLICY=1278`.
- The 1280-added native-pair shared state and initialization are active for 1278 because they are gated by `STATIC_ONLINE_MXFP4_POLICY126_NATIVE_QUEUE_PAIR_HANDOFF` (`>=1277`).
- Selected-kernel ptxas footprint is unchanged across historical/current 1278 and Stage2: `128` registers, `24` B stack, `32` B spill stores, `92` B spill loads, `1280` B smem.

Conclusion: fast 1278 did not reproduce. General GPU slowdown and ordering are unlikely because current Stage2 is faster than historical Stage2 and reverse order still favors Stage2. Most actionable explanation is either a one-off historical 1278 timing or 1278-specific source drift from active 1280-added shared state/init. Next step should be a narrow no-algorithm bisection that gates the 1280 follow-up shared arrays/initialization to `>=1280` and reruns Stage2/1278.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_reproduce_fast_1278_restored_smoke_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Policy1278 Clean-State Bisection: 2026-07-06

Report: `forward_policy1278_clean_state_bisection_20260706_report.md`.
Checkpoint: `forward_policy1278_clean_state_bisection_20260706_checkpoint.md`.

Implemented a no-new-algorithm cleanup gate:

- Added `STATIC_ONLINE_MXFP4_POLICY126_NATIVE_PAIR_FOLLOWUP_STATE`.
- Gated 1280+ follow-up shared state/storage and init to native pair policies `>=1280`.
- Policy1278 native queue-pair behavior is otherwise unchanged: normal append, two-entry native pair helper, counters compiled out.

Cleaned Policy1278 smokes:

- h4/s1024: finite `true`, raw/decoded `0/0`, counters `0`.
- h4/s2048: finite `true`, raw/decoded `0/0`, counters `0`.

Matched GPU2 timing, timeline off, warmup/iters `5/20`:

- Stage2 h4/s2048: `0.060192/0.058528`, `0.061504/0.059488`, `0.063280/0.060896` ms.
- Cleaned Policy1278 h4/s2048: `0.065728/0.063712`, `0.066784/0.064640`, `0.064160/0.062016` ms.
- Stage2 h4/s1024: `0.042832/0.040480` ms.
- Cleaned Policy1278 h4/s1024: `0.044912/0.043168` ms.

Conclusion: cleaned 1278 did not recover the historical fast `0.061984/0.060160` ms result and did not beat matched Stage2. It only improved p50 by `0.30%` versus the prior reproduction best `0.064352/0.061152`, with a worse min, so treat as noise. The 1280-added follow-up shared state/init was not the meaningful cause of the missing fast run. Keep the cleanup only as source hygiene if desired; do not keep optimizing this path.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy1278_clean_restored_smoke_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Policy1290 Queue-Control Simplification: 2026-07-07

Report: `forward_policy1290_queue_control_simplification_20260707_report.md`.
Checkpoint: `forward_policy1290_queue_control_simplification_20260707_checkpoint.md`.

Baseline GPU2 matrix, timeline off, warmup/iters `5/20`:

- Stage2: h4/s1024 `0.042688/0.041056` ms; h4/s2048 `0.061504/0.059936` and `0.059376/0.057888` ms.
- Current 1278: h4/s1024 `0.044112/0.042720` ms; h4/s2048 `0.070416/0.067616` and `0.067680/0.065120` ms.

Implemented queue-control simplification variants:

- 1290/1291: inline tail issue inside `drain_one_committed_fwd_pipeline_qk_job()`.
- 1292/1293: pressure-only inline tail issue.
- 1294: stripped inline fast path.
- Old after-append native-pair helper is gated off for 1290+.
- 1280 follow-up gates are fenced below 1290 so release-latch/append-candidacy/event state does not leak into the queue-control family.

Counter evidence:

- 1290 h4/s2048: `224` adjacent pairs seen, `196` release-ready/useful tail issues, capacity fallback `60`, final fallback `28`.
- 1292 h4/s2048: same `224` seen and `196` useful tail issues, capacity fallback removed, final fallback `28`.

Timing:

- 1291: h4/s1024 `0.044432/0.042656`; h4/s2048 `0.065008/0.064032`, repeat `0.064416/0.062592`.
- 1293: h4/s1024 `0.043184/0.041056`; h4/s2048 `0.062176/0.059936`, repeat `0.063744/0.062112`.
- 1294: h4/s1024 `0.054368/0.050688`; h4/s2048 `0.065840/0.063168`, repeat `0.063584/0.062496`.

Conclusion: Policy1293 is the best current native queue candidate. It beats matched current 1278 by `8.13%` on best h4/s2048 p50 and is `4.72%` slower than the best matched Stage2 p50. The repeat still beats 1278 by `5.82%`. Pressure-only placement removes useless non-pressure checks while preserving all useful handoffs. 1294 did not help.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy1290_restored_smoke_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Policy1295 Cheap Pressure Tail: 2026-07-08

Report: `forward_policy1295_cheap_pressure_tail_20260708_report.md`.

Matched GPU2 matrix, timeline off, warmup/iters `5/20`:

- Stage2: h4/s1024 `0.042544/0.039712` ms; h4/s2048 `0.062080/0.060064` and `0.060720/0.057696` ms.
- Policy1293: h4/s1024 `0.043712/0.041856` ms; h4/s2048 `0.064288/0.062304` and `0.062208/0.059968` ms.

Implemented policy variants:

- 1295: cheap pressure-only tail block inside the normal committed queue drain path.
- 1296: hoisted/precomputed cheap predicates.
- 1297: split pressure/full-queue eligibility before head issue.
- 1298: shortest successful tail queue update, setting `qk_pipeline_count = 0`.
- Added default-off `POLICY126_COUNTERS` build flag so the same policy can be smoked with counters on and timed with counters off.

Candidate timing:

- 1295: h4/s1024 `0.045232/0.043232`; h4/s2048 `0.063536/0.061376`, repeat `0.063680/0.061728`.
- 1296: h4/s1024 `0.043744/0.041472`; h4/s2048 `0.065168/0.061600`, repeat `0.064208/0.062816`.
- 1297: h4/s1024 `0.044272/0.043136`; h4/s2048 `0.063760/0.061248`, repeat `0.064128/0.061088`.
- 1298: h4/s1024 `0.044144/0.041376`; h4/s2048 `0.062304/0.059232`, repeat `0.067648/0.065568`, third `0.064016/0.061888`.

Counter evidence:

- 1295 h4/s2048 counters-on smoke: `224` seen, `196` release-ready/useful tail issues, `196` defer consumed, `28` final fallbacks.
- 1298 h4/s2048 counters-on smoke: same `224/196/196/28` pattern.

Codegen:

- Selected Stage2 streaming-live kernel: `168` registers, `2` barriers, `0` B stack, `1904` B smem.
- Selected 1293/1295/1296/1297/1298 streaming-live kernels: `168` registers, `2` barriers, `160` B stack, `2080` B smem.

Conclusion: none of 1295-1298 robustly improves over matched 1293 or Stage2. 1298 nearly tied 1293 once but did not repeat. The useful tail issue count stays at `196`, so the problem is not opportunity loss; these source-level tail-block simplifications do not remove the scheduler-family footprint or make the pressure-tail handoff cheap enough. Do not replace Policy1293 with 1295-1298.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy1295_restored_smoke_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## Forward Policy1299 Proxy Release Signal: 2026-07-09

Report: `forward_policy1299_proxy_release_signal_20260709_report.md`.

Matched GPU2 matrix, timeline off, warmup/iters `5/20`:

- Stage2: h4/s1024 `0.042832/0.040256` ms; h4/s2048 `0.060672/0.058752` and `0.060544/0.058688` ms.
- Policy1293: h4/s1024 `0.044304/0.042656` ms; h4/s2048 `0.066208/0.065056` and `0.063760/0.062880` ms.

Implemented `KPIPE_SELECTIVE_POLICY=1299` as a pressure-only proxy-release tail readiness variant:

- counters-off timing path does not load/check `hotplate_slot_release_epoch[...]` for tail readiness,
- counters-on diagnostic path computes real release-ready and records proxy candidates, real release-ready candidates, and proxy false positives,
- final-tile and preconsume guards are preserved,
- append-time helper scheduling remains disabled.

Timing:

- 1299 h4/s1024: `0.042816/0.041248` ms.
- 1299 h4/s2048: `0.061568/0.060064`, repeat `0.061696/0.059872` ms.
- Best 1299 h4/s2048 p50 is `3.44%` faster than matched 1293 and `1.69%` slower than matched Stage2.

Proxy safety:

- h4/s1024: proxy candidates `36`, real release-ready `36`, false positives `0`, issued tails `36`.
- h4/s2048: proxy candidates `196`, real release-ready `196`, false positives `0`, issued tails `196`.
- h4/s4096 stress: proxy candidates `900`, real release-ready `900`, false positives `0`, issued tails `900`.

Codegen:

- Stage2 selected streaming-live kernel: `168` registers, `2` barriers, `0` B stack, `1904` B smem.
- 1293 selected streaming-live kernel: `168` registers, `2` barriers, `160` B stack, `2080` B smem.
- 1299 selected streaming-live kernel: `168` registers, `2` barriers, `160` B stack, `2080` B smem.

Conclusion: keep 1299 as the best native queue candidate from this branch. The proxy did not false-positive on required or stress shapes and improved h4/s2048 versus 1293, but it did not change stack/smem and still trails Stage2.

Final state:

- Restored active build:
  - `timeout 1800s make -B -C tk_fa4/fp4_fa4_fwd forward MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0`
  - Result: success.
- Final GPU2 default/off smoke:
  - `forward_policy1299_restored_smoke_h4_s1024_gpu2_stdout.json`: finite `true`, raw/decoded `0/0`, all Policy126 counters `0`.

## 2026-07-09 - Policy1300 scheduler footprint reduction

Task file: `session6_policy1300_scheduler_footprint_20260709.md`.

Fresh matched GPU2 baseline, timeline off, warmup/iters `5/20`:

- Stage2/off: h4/s1024 `0.043920/0.042304` ms; h4/s2048 `0.061408/0.059680`, repeat `0.063168/0.061024` ms; selected ptxas `168 regs, 2 barriers, 1904 B smem`.
- Policy1299: h4/s1024 `0.047696/0.045856` ms; h4/s2048 `0.067232/0.065280`, repeat `0.062880/0.060064` ms; selected ptxas `168 regs, 2 barriers, 160 B stack, 2080 B smem`.

Implemented and profiled:

- Policy1300 lean counters-off proxy tail path: h4/s1024 `0.042976/0.041024`; h4/s2048 `0.064176/0.062368`, repeat `0.062416/0.060224`; ptxas unchanged.
- Policy1301 standalone drain-pair gating excluding older native queue-pair umbrella: h4/s1024 `0.046176/0.043232`; h4/s2048 `0.061024/0.059456`, repeat `0.061568/0.060544`; ptxas unchanged.
- Policy1302 specialized tail issue helper: h4/s1024 `0.043456/0.041792`; h4/s2048 `0.061984/0.059488`, repeat `0.062928/0.060864`; ptxas unchanged.
- Policy1303 proven-wins combination excluding 1302: h4/s1024 `0.047696/0.045728`; h4/s2048 `0.062432/0.060544`, repeat `0.061120/0.060096`; ptxas unchanged.

Best candidate: Policy1301. Counters-on proxy safety:

- h4/s1024: proxy candidate/real-ready/false-positive `36/36/0`, issued tails `36`.
- h4/s2048: `196/196/0`, issued tails `196`.
- h4/s4096 stress: `900/900/0`, issued tails `900`.

Conclusion: Policy1301 is the best continuation point in this family, but this is primarily a useful negative footprint result. The selected `160 B stack / 2080 B smem` footprint survives 1300-1303, so the next real target is a deeper committed-queue state/owner representation rewrite rather than more local tail-branch surgery.

Final state restored:

- Default/off build command succeeded with `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0 MXFP4_FWD_TIMELINE=0`.
- Default/off h4/s1024 smoke `forward_policy1300_restore_default_h4_s1024_gpu2_stdout.json`: finite `true`, `0.042688/0.041376` ms.

## 2026-07-09 - Policy1304 scalar committed-QK queue rewrite

Task file: `session6_policy1304_scalar_queue_20260709.md`.

Report: `forward_policy1304_scalar_queue_20260709_report.md`.

Fresh matched GPU2 baseline, timeline off, counters off, warmup/iters `5/20`:

- Stage2/off: h4/s1024 `0.041920/0.040448` ms; h4/s2048 `0.060384/0.058848`, repeat `0.064992/0.061824` ms; selected ptxas `168 regs, 2 barriers, 1904 B smem`.
- Policy1301: h4/s1024 `0.043632/0.041056` ms; h4/s2048 `0.062816/0.060192`, repeat `0.065312/0.062816` ms; selected ptxas `168 regs, 2 barriers, 160 B stack, 2080 B smem`.

Implemented and profiled:

- Policy1304 scalar two-slot `QkIssueCandidate` queue with short-lived full-job materialization: h4/s1024 `0.042640/0.041504`; h4/s2048 `0.062688/0.060800`, repeat `0.061472/0.059680`; selected ptxas `168 regs, 2 barriers, 144 B stack, 2080 B smem`.
- Policy1305 direct candidate issue helper: h4/s1024 `0.044224/0.042368`; h4/s2048 `0.063232/0.060416`, repeat `0.062896/0.061472`; ptxas unchanged from 1304.
- Policy1306 packed scalar queue record `{next_idx, preconsume_idx, meta}`: h4/s1024 `0.042416/0.040832`; h4/s2048 `0.063088/0.060864`, repeat `0.063104/0.061920`; selected ptxas `168 regs, 2 barriers, 32 B stack, 2080 B smem`.
- Policy1307 proven-combo gate, excluding unproven direct issue: h4/s1024 `0.043104/0.041280`; h4/s2048 `0.062560/0.060608`, repeat `0.063280/0.061408`; selected ptxas `168 regs, 2 barriers, 32 B stack, 2080 B smem`.

Best footprint candidate: Policy1306/1307. It removes most selected stack overhead but does not reduce the surviving `2080 B smem` scheduler footprint or beat Stage2.

Counters-on Policy1307 proxy safety:

- h4/s1024: proxy candidate/real-ready/false-positive `36/36/0`, issued tails `36`.
- h4/s2048: `196/196/0`, issued tails `196`.
- h4/s4096 stress: `900/900/0`, issued tails `900`.

Conclusion: the scalar queue rewrite proves the full two-entry job object was responsible for stack pressure, but not for the remaining smem delta. The next target should be the broader scheduler state enabled by the native/proxy policy gate, not another local queue record packing pass.

Final state restored:

- Default/off build command succeeded with `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0 MXFP4_FWD_TIMELINE=0`.
- Default/off h4/s1024 smoke `forward_policy1304_restore_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, `0.045824/0.043584` ms, raw/decoded timeline `0/0`.

## 2026-07-09 - Policy1308 shape sweep and shape-specific optimization

Task file: `session6_policy1308_shape_sweep_20260709.md`.

Report: `forward_policy1308_shape_sweep_20260709_report.md`.

Fresh GPU2 shape matrix, timeline/counters off:

- Required supported H/S matrix covered H `2,4,8,16` with S `1024,2048,4096`; H `1` is unsupported by the harness for this config.
- Stress S8192 ran for H `4` and `8` with warmup/iters `3/10`; normal shapes used `5/20`.
- All supported runs were finite with raw/decoded timeline `0/0`.

Selected ptxas:

- Stage2/off: `168 regs, 2 barriers, 1904 B smem`.
- Policy1301: `168 regs, 2 barriers, 160 B stack, 2080 B smem`.
- Policy1307: `168 regs, 2 barriers, 32 B stack, 2080 B smem`.
- Policy1308: `168 regs, 2 barriers, 160 B stack, 2080 B smem`.

Shape result:

- Stage2 won every supported first-pass shape except h8/s1024.
- h8/s1024 first-pass: Stage2 `0.044016/0.040800`, Policy1301 `0.043632/0.041376` (`+0.88%`), Policy1307 `0.044288/0.041920`.
- h8/s1024 repeat check:
  - Stage2 p50s: `0.044016`, `0.041808`, `0.043072`; median `0.043072`.
  - Policy1301 p50s: `0.043632`, `0.041632`, `0.041536`; median `0.041632`, `+3.46%` versus Stage2.
  - Policy1308 p50s: `0.043280`, `0.045232`, noisy and not a robust default replacement.

Policy decisions:

- Implemented Policy1308 as a compile-time shape-specific alias to the 1301-style native/proxy behavior by limiting scalar/packed queue policies to 1304-1307. No runtime shape branch was added in-kernel.
- Did not attempt Policy1309: long-S regions were not close or winning. h4/s8192: Policy1307 `-2.22%`, Policy1301 `-4.31%`; h8/s8192 worse.
- Did not attempt Policy1310: high-H regions did not improve. h16/s2048 and h16/s4096 remained slower than Stage2.

Proxy safety for Policy1308 counters-on:

- h4/s2048: proxy candidate/real-ready/false-positive `196/196/0`, issued tails `196`, final fallbacks `28`.
- h8/s8192: `7688/7688/0`, issued tails `7688`, final fallbacks `248`.
- h16/s4096: `3600/3600/0`, issued tails `3600`, final fallbacks `240`.
- h8/s1024: `72/72/0`, issued tails `72`, final fallbacks `24`.

Conclusion: keep Stage2 globally. Policy1301 remains a narrow opt-in candidate for h8/s1024 if cheap compile-time per-shape selection is available, but this sweep does not justify global shape-specific dispatch or more local native/proxy scheduler polishing.

Final state restored:

- Default/off build command succeeded with `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0 MXFP4_FWD_TIMELINE=0`.
- Default/off h4/s1024 smoke `forward_policy1308_restore_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, `0.041408/0.039680` ms, raw/decoded timeline `0/0`.

## 2026-07-10 - Policy1311 shared-memory footprint compileout

Task file: `session6_policy1311_smem_compileout_20260710.md`.

Report: `forward_policy1311_smem_compileout_20260710_report.md`.

Result: the selected-kernel `+176 B` smem delta versus Stage2 was fully attributed and removed for timeline-off/counters-off Policy1311+ timing builds.

Selected ptxas:

- Stage2/off fresh: `168 regs, 2 barriers, 1904 B smem`.
- Policy1307 fresh: `168 regs, 2 barriers, 32 B stack, 2080 B smem`.
- Policy1311 final: `168 regs, 2 barriers, 32 B stack, 1904 B smem`.
- Policy1312 final: `168 regs, 2 barriers, 32 B stack, 1904 B smem`.
- Policy1313 final: `168 regs, 2 barriers, 32 B stack, 1904 B smem`.
- Policy1314 timing: `168 regs, 2 barriers, 32 B stack, 1904 B smem`.
- Policy1314 counters-on: `168 regs, 2 barriers, 160 B stack, 2080 B smem`.

Attribution:

- Middle intent state was accidentally live under `HOTPLATE_POLICY=126`: `10 arrays * 2 score slots * 4 B = 80 B`.
- Twoid/shadow state was also live through the middle diagnostic umbrella: `11 arrays * 2 score slots * 4 B = 88 B`, plus two scalar flags `8 B`.
- Total removed: `80 + 88 + 8 = 176 B`.
- Native-pair placeholders and score-unused epochs were checked by Policy1312 and did not move selected ptxas beyond the diagnostic-state removal.

Policy1314 timing, GPU2, timeline/counters off:

- h4/s2048: `0.063264/0.061248` ms, finite, raw/decoded timeline `0/0`; slightly better than Policy1307 run1 but still slower than Stage2 repeats.
- h8/s1024: `0.041168/0.039648` ms, finite, raw/decoded `0/0`; better than Stage2 and Policy1307 repeats.
- h4/s4096: `0.100640/0.098720` ms, finite, raw/decoded `0/0`; better than Policy1307, still slower than Stage2 p50.
- h8/s4096: `0.103952/0.102112` ms, finite, raw/decoded `0/0`; better than Policy1307, still slower than Stage2.

Policy1314 counters-on proxy safety:

- h4/s2048: proxy candidate/real-ready/false-positive `980/980/0`.
- h8/s1024: `360/360/0`.
- h8/s4096: `9000/9000/0`.

Conclusion: keep Stage2 as global default. Policy1314 is a cleaned native/proxy opt-in candidate with the smem delta removed and zero proxy false positives, but it does not justify replacing Stage2 globally. The next scheduler experiments should avoid broad diagnostic umbrella flags for behavior policies unless their shared storage has an explicit storage-active gate.

Final state restored:

- Default/off build command succeeded with `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0 MXFP4_FWD_TIMELINE=0`.
- Default/off h4/s1024 smoke `forward_policy1311_restore_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, `0.044288/0.042496` ms, raw/decoded timeline `0/0`.

## 2026-07-10 - Stage2-only profiling and bottleneck report

Task file: `session6_stage2_profile_20260710.md`.

Report: `forward_stage2_profile_20260710_report.md`.

Result: profiled Stage2/off as the product path only. No new scheduler policies were implemented.

Stage2/off build flags:

- `MXFP4_FWD_TIMELINE=0 KPIPE_STAGE=2 SCORE_REUSE_PIPE_STAGE=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 KPIPE_SELECTIVE_POLICY=0 POLICY126_COUNTERS=0`

Selected ptxas:

- Stage2/off: `168 regs, 2 barriers, 1904 B smem`, `0` stack/spills.

Timing matrix on GPU2, timeline off:

- h4/s2048: finite `true`, p50/min `0.061392/0.059520` ms, raw/decoded `0/0`.
- h4/s4096: finite `true`, `0.100800/0.098784` ms, raw/decoded `0/0`.
- h8/s1024: finite `true`, `0.042752/0.040768` ms, raw/decoded `0/0`.
- h8/s4096: finite `true`, `0.101456/0.098368` ms, raw/decoded `0/0`.
- h16/s4096: finite `true`, `0.189920/0.186464` ms, raw/decoded `0/0`.
- h4/s8192: finite `true`, `0.184288/0.182720` ms with `3/10` warmup/iters, raw/decoded `0/0`.

NCU profiling:

- NCU 2025.3.1 was available.
- h4/s2048 compact NCU: duration `49.632 us`, issue `7.64%`, TC `3.10%`, TMA `0.06%`, memory `2.82%`, DRAM `0.51%`, eligible/issued `0.39/0.34`; top stalls long_scoreboard `3.38`, wait `1.78`, short_scoreboard `0.56`, no_instruction `0.52`, barrier `0.24`.
- h8/s1024 compact NCU: duration `31.072 us`, issue `7.51%`, TC `2.66%`, TMA `0.06%`, memory `2.35%`, DRAM `0.82%`, eligible/issued `0.36/0.31`; top stalls long_scoreboard `3.34`, wait `1.82`, no_instruction `0.64`, short_scoreboard `0.60`, barrier `0.35`.
- h8/s4096 compact NCU timed out after 900s before a data row; minimal retry completed with duration `87.872 us`, issue `30.53%`, TC `13.51%`, eligible/issued `0.40/0.35`; top stalls long_scoreboard `3.47`, wait `1.75`, barrier `0.21`.

Diagnosis:

- Stage2 is not barrier/semaphore dominated; barrier stalls are low compared with long_scoreboard and wait.
- It is not global-memory/TMA-bandwidth dominated; TMA, SM memory, and DRAM utilization are low in the compact profiles.
- TC utilization is low, especially on h4/s2048 and h8/s1024, but the root symptom is ready-work starvation: low eligible warps and high scoreboard/wait stalls.
- h8/s1024 is not qualitatively different from h4/s2048 in NCU. Treat its previous native/proxy wins as a narrow/noisy shape-specific corner, not a global scheduler direction.
- Next useful work should source-attribute the long-scoreboard/wait path inside Stage2, especially score/P/PV dependency readiness, before more QK-tail scheduler experiments.

Final state restored:

- Default/off build command succeeded with `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0 MXFP4_FWD_TIMELINE=0`.
- Default/off h4/s1024 smoke `forward_stage2_profile_restore_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, `0.044016/0.041632` ms, raw/decoded timeline `0/0`.

## 2026-07-10 - Stage2 bottleneck source follow-up

Task file: `session6_stage2_bottleneck_followup_20260710.md`.

Report: `forward_stage2_bottleneck_followup_20260710_report.md`.

Result: source-correlated NCU was attempted but was too slow/incomplete for a usable source stall table, so Stage2 timeline-only markers were added under `MXFP4_FWD_TIMELINE`.

Baseline Stage2/off timing, GPU2, timeline off:

- h4/s2048: finite `true`, p50/min `0.062256/0.059904` ms, raw/decoded `0/0`.
- h8/s1024: finite `true`, `0.041600/0.039840` ms, raw/decoded `0/0`.
- h8/s4096: finite `true`, `0.101152/0.099008` ms, raw/decoded `0/0`.
- Selected ptxas: `168 regs, 2 barriers, 1904 B smem`, `0` stack/spills.

Marker attribution:

- QK K/K-scale waits are only about `180-240` cycles and QK issue-to-commit about `345-375` cycles.
- The exposed chain is P/softmax/PV: `score_tmem_load_to_p_ready` median about `4990/4991/5257` cycles for h4/s2048, h8/s1024, h8/s4096.
- `output_pv_wait` is also large: about `4120/3275.5/4056.5` cycles.
- Conclusion: the next target is P-ready/PV consume readiness, not QK-tail/native/proxy scheduling and not register-only cleanup.

Follow-up variants:

- New opt-in Stage2 `pready_vtma_vstma` sibling: same ptxas as Stage2, finite but slower in matched direct-prealloc timing: h4/s2048 `0.067328` vs `0.060096`, h8/s1024 `0.043184` vs `0.041856`, h8/s4096 `0.112752` vs `0.100720`.
- Existing `lazyorescale_vtma_vstma`: finite but slower and spills (`8 B stack`, `4 B stores`, `32 B loads`): h4/s2048 `0.062304`, h8/s1024 `0.044912`, h8/s4096 `0.105120`.

Final state restored:

- Default/off build command succeeded with `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0 MXFP4_FWD_TIMELINE=0`.
- Default/off h4/s1024 smoke `forward_stage2_bottleneck_restore_default_off_h4_s1024_gpu2_stdout.json`: finite `true`, `0.047104/0.046624` ms, raw/decoded timeline `0/0`.

## 2026-07-10 - QK epilogue fusion explore

Task file: `session6_qk_epilogue_fusion_explore_20260710.md`.

Report: `forward_qk_epilogue_fusion_explore_20260710_report.md`.

Baseline Stage2/off timing, GPU2, timeline off:

- h4/s2048: finite `true`, p50/min `0.059792/0.058208` ms.
- h8/s1024: finite `true`, `0.041360/0.040000` ms.
- h8/s4096: finite `true`, `0.101824/0.098496` ms.
- Selected ptxas: `168 regs, 2 barriers, 1904 B smem`, `0` stack/spills.

Marker attribution added events 381-389 under `MXFP4_FWD_TIMELINE` only. The largest real sub-block was P exp/weight + payload pack: `2198.5/2231.5/2223.5` median cycles for h4/s2048, h8/s1024, h8/s4096. Publish/P-scale/P-ready work was another about `1.25k + 1.26k` cycles. QK issue-to-commit remained only `343/336/348` cycles.

Decision: no QK-epilogue fusion candidate was implemented after attribution. Candidate A/B fail the task gate because the remaining work depends on online-softmax row recurrence/P publish state, and moving the largest block would shift thousands of cycles into the QK issue path rather than producing a surgical win.

Artifacts:

- `forward_qk_epilogue_stage2_marker_driver.py`
- `forward_qk_epilogue_stage2_markers_h4_s2048_gpu2.json`
- `forward_qk_epilogue_stage2_markers_h8_s1024_gpu2.json`
- `forward_qk_epilogue_stage2_markers_h8_s4096_gpu2.json`

Final state restored:

- Default/off build command succeeded with `KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0 MXFP4_FWD_TIMELINE=0`.
- Default/off h4/s1024 smoke `forward_qk_epilogue_restore_default_off_h4_s1024_gpu2.json`: finite `true`, `0.043184/0.038848` ms, raw/decoded timeline `0/0`.

## 2026-07-10 - Stage2 P-chain overlap and exp/pack optimization

Task file: `session6_stage2_pchain_overlap_20260710.md`.

Report: `forward_stage2_pchain_overlap_20260710_report.md`.

Implemented sparse compile-gated P-chain stamps plus real A/B/C and exact two-pack
candidates. All selected kernels remained `168 regs, 2 barriers, 1904 B smem`, with
zero stack/spills.

Low-perturbation Stage2 attribution, h4/s2048 / h8/s1024 / h8/s4096:

- scale derivation to exp/pack: `1356/1349/1354` cycles.
- P-scale TMEM store wait: `65/65/66` cycles.
- payload publish: `52/52/52` cycles.
- publish to P/P-scale ready: `273/273/274` cycles.
- ready to PV issue: `65/67/69` cycles.

Candidate decisions:

- A overlapped the TMEM store but increased scale-to-exp/pack live-range cost;
  rejected and selector removed.
- B exposed `72-82` cycles by publishing real current-P readiness before row-sum
  recurrence bookkeeping, but did not repeat across supporting shapes; rejected and
  selector removed.
- C combined A+B and repeated at h4/s2048 (`-1.27%`) and h8/s4096 (`-1.47%`), but
  regressed h8/s1024 (`+0.98%`). Retained only as explicit/default-off `pchainc`.
- Exact pack2 did not shorten exp/pack and its h8/s4096 first-pass gain reversed;
  rejected and implementation removed.

Compact h8/s4096 NCU for C versus Stage2: duration `86.336` versus `88.256 us`,
long-scoreboard `3.40` versus `3.47`, wait `1.72` versus `1.75`, and eligible warps
`0.41` versus `0.40`, with identical resources. Stage2 remains the global default.

Final state restored:

- Build succeeded with `MXFP4_FWD_TIMELINE=0 MXFP4_FWD_PCHAIN_STAMPS=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0`.
- Default/off h4/s1024 smoke `forward_stage2_pchain_restore_default_off_h4_s1024_gpu2.json`: finite `true`, `0.041840/0.040608` ms, timeline raw/decoded `0/0`, P-chain stamp API empty.

## 2026-07-11 - Stage2 paired ALU EX2 and FP4 packing

Task file: `session6_stage2_ex2_alu_pack_20260711.md`.

Report: `forward_stage2_ex2_alu_pack_20260711_report.md`.

Ported the exact local FA4 paired degree-3 `e2e_asm2` PTX and restricted it to the
Stage2 score-derived prescaled P payload loop. The SM100a probe contains zero
`MUFU.EX2` and three packed `FFMA2.FTZ` per emulated pair. Across 2,000,074 scalar
values, max absolute/relative EX2 error was `4.625320e-4/8.778174e-5`; 4-of-16
payload-word mismatch rate was `1.999928e-5`, and causal `-inf` mapped to zero.

Cadence result:

- e16 (4-of-16) reduced SASS `MUFU.EX2` `257 -> 193`, added 96 packed FFMA2,
  retained `168 regs, 2 barriers, 1904 B smem`, and introduced no stack/spills.
- Reverse repeats versus Stage2: h4/s2048 `-3.52%`, h8/s1024 `-1.98%`, and
  h8/s4096 `-5.12%`; scale-to-exp/pack fell `1354 -> 1189` and `1327 -> 1187`
  cycles on the two supporting stamp shapes.
- e10/e8 spilled and slowed; all-emulated was 16-28% slower. These selectors were
  removed.

Packing and final route:

- Interleaved packing changed SASS but did not beat the existing schedule across
  shapes, so both interleaved routes were removed.
- Retained explicit/default-off `ex2e16pc`, combining e16 with prior `pchainc`.
- Clean first/reverse p50 gains versus Stage2 were h4/s2048 `-5.15/-3.81%`,
  h8/s1024 `-1.83/-2.37%`, h8/s4096 `-4.59/-5.18%`, and h16/s4096
  `-3.76/-4.56%`.
- h8/s4096 NCU duration was `82.208` versus `87.904 us`; dynamic XU instructions
  fell 23.6%, FMA active rose `14.77 -> 22.66%`, issue active rose
  `30.57 -> 36.41%`, and long-scoreboard/wait fell `3.47/1.75 -> 2.75/1.26`.
- Combined high-head determinism stayed within Stage2's envelope. Standalone e16
  did not, so its selector was removed.

Final state restored:

- Forced build succeeded with `MXFP4_FWD_TIMELINE=0 MXFP4_FWD_PCHAIN_STAMPS=0
  KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0
  HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0`.
- Default/off h4/s1024 smoke is finite at `.044512/.041248` ms p50/min, selects
  ordinary Stage2, has timeline raw/decoded `0/0`, all policy counters zero, and
  no nonzero P-chain stamp intervals.

## 2026-07-11 - FP4-native exp/pack addendum

Task file: `session6_fp4_native_exp_pack_addendum_20260711.md`.

Results were added to `forward_stage2_ex2_alu_pack_20260711_report.md`.

Focused SM100a validation over 2,000,050 scalar inputs:

- F1 direct nibble exactly matched hardware E2M1 conversion of the same cubic
  output, but differed from native EX2+cvt at rate `4.499888e-5`.
- F2 degree-3 and degree-2 direct nibbles differed from native only 7 times
  (`3.499913e-6`), all in the directed threshold-adjacent set. Degree-3 exp error
  was `4.625320e-4/8.778174e-5` max abs/relative; degree-2 was
  `9.662628e-3/2.075664e-3`.
- F3 codes exactly matched F2 and the PRMT/DP4A half-unit sum had zero mismatches.
  Final probe classifiers had no data-dependent branches.

Kernel result:

- F1/F2 e16 removed 32 of 128 E2M1 conversions but spilled. Best ptxas was
  F1 `64/68 B`, F2 degree-3 `40/36 B`, and F2 degree-2 `68/72 B` stores/loads.
- F3 was spill-free and genuinely removed the P-loop instructions: whole-kernel
  `MUFU 257 -> 1`, `F2FP 144 -> 16`, and `E2M1 128 -> 0`. Replacement cost was
  2330 ISETP, 1299 FSEL, 2139 SEL, and 64 IDP.4A instructions.
- First/reverse p50 regressions versus Stage2 were F1 `+43%..+106%`, F2 degree-3
  `+39%..+101%`, F2 degree-2 `+41%..+102%`, and F3 `+165%..+357%` across
  h4/s2048, h8/s1024, and h8/s4096.
- Sparse scale-to-exp/pack grew from about `1.45k` cycles to `2.60k-3.35k` for
  F1/F2 and `12.55k-13.06k` for F3.
- F3 changed-denominator error was output relative L2 `0.165-0.171` and LSE max
  abs `0.286-0.398`; it was slower as well as materially less accurate.

Decision: reject F1, both F2 degrees, and F3. No neighboring cadence, pchainc
combination, h16 winner run, or NCU winner profile was justified. All addendum
selectors/configs/kernel behavior were removed; `ex2e16pc` remains the sole new
explicit/default-off route.

Final state restored:

- Clean build succeeded with `MXFP4_FWD_TIMELINE=0 MXFP4_FWD_PCHAIN_STAMPS=0
  KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0
  HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0`.
- Default Stage2 h4/s1024 is finite at `.041872/.039744` ms p50/min, with timeline
  raw/decoded `0/0`, all policy counters zero, and an empty P-chain stamp read.

## 2026-07-11 - EX2 degree and phase tuning

Task file: `session6_ex2_degree_phase_loop_20260711.md`.

Report: `forward_stage2_ex2_degree_phase_20260711_report.md`.

Result: no new route retained. Stage2 and committed e16pc reproduced the required
`257/0/144/128` and `193/96/144/128` MUFU/packed-FFMA2/F2FP/E2M1 counts, with
168 registers, two barriers, 1904 B smem, and no stack/spills.

- Degree-2 4-of-16/12/10 were spill-free; 4-of-8 spilled 8 B each way. Degree 2
  had LSE max error around `4.7e-4..9.6e-4` and did not repeat a timing win.
- Degree-3 tail/front/even/split masks had identical static counts and resources;
  tail normalized SASS was byte-identical to e16pc. Every non-tail mask lost both
  h8/s4096 timing orders.
- Degree-2 split shortened the local scale-to-exp/pack interval by about 47-48
  cycles, but isolated 60-sample confirmation was slower in five of six
  shape/order comparisons. Its only `-0.49%` result became `+0.19%` in reverse.
- A held-out degree-2 coefficient fit improved row-sum RMS but worsened maximum
  relative exp error (`2.075519e-3 -> 2.233492e-3`) and remained far behind degree
  3, so it was rejected before kernel integration.
- Row-sum interaction and NCU profiling gates did not open. All rejected helpers,
  selectors, configs, and behavior were removed; only the branch-free mask trait
  used by e16pc remains.

Final default/off build succeeded with all compile gates zero. Ordinary Stage2
h4/s1024 is finite at `.054176` ms p50, with an empty P-chain raw read and zero
interval counts. No commit or push was performed.

## 2026-07-11 - e16pc critical-path optimization

Task file: `session6_e16pc_critical_path_optimization_20260711.md`.

Report: `forward_e16pc_critical_path_optimization_20260711_report.md`.

Result: no new behavior route retained. A 21-slot, target-index sparse profiler now
correlates score load/reuse/P construction with issue-side output, V, descriptor,
P-ready, and actual PV issue using a combined CTA/task owner. All 132 required
idx0/idx2 Stage2/e16pc records were valid with no timeout or rejection. Disabled
e16pc SASS is byte-identical before/after the diagnostic changes.

- Matched Stage2/e16pc baseline reproduced 168 registers, 1904 B smem, no spills,
  and `257/0/144/128` versus `193/96/144/128` MUFU/FFMA2/F2FP/E2M1.
- At steady-state h4/s2048 and h8/s4096, e16pc producer P-ready is already about
  `.88-.96k` cycles ahead of issue-side P-ready. Output-rescale/V/descriptor work,
  not producer P construction, selects the PV issue point. PV follows issue-side
  P-ready in only `.13-.15k` cycles.
- Candidate A moved the exact qid denominator fold after P-ready while retaining
  168 registers and zero spills. Producer chain medians improved by 35, 120, and
  367 cycles on h4/s2048, h8/s1024, and h8/s4096, but producer-ready-to-PV stayed
  flat or grew and first/reverse wall time did not repeat.
- The apparent h4 result failed 60-sample confirmation: `+1.12%` first and
  `-0.28%` reverse versus e16pc. h16 candidate run-to-run output/LSE max abs reached
  `2.045e-3/1.078e-3`. Candidate A was removed.
- Profile gates rejected max-tree, payload/scale handoff, and scale-ILP variants;
  no NCU finalist was justified. Retain committed e16pc unchanged.

Final default/off build succeeded with all gates zero, including `KPIPE_STAGE=0`
and `PCHAIN_TARGET_IDX=0`. Ordinary Stage2 h4/s1024 is finite at
`.052304/.048512` ms p50/min; timeline is empty, P-chain raw is `[[]]`, and all 64
policy counters are zero. No commit or push was performed.

## 2026-07-11 - Issue-lane overlap and broad BF16 matrix

Task file: `session6_issue_lane_overlap_bf16_matrix_20260711.md`.

Report: `forward_issue_lane_overlap_bf16_matrix_20260711_report.md`.

Result: reject and remove both V-only and V+P overlap schedules. Their sparse
profiles proved the intended movement: at steady-state h8/s4096, V-ready moved
from 4287 cycles after issue entry to 447/462, and VP reduced output-ready-to-PV
from 1046 to 163 cycles. VP also grew issue-P-ready-to-PV from 132 to 442 cycles,
and neither schedule repeated a wall-time win. Candidate-vs-e16pc was not
bit-identical on larger shapes (output/LSE max `1.923e-3/5.817e-4`) inside the
existing Stage2/e16pc nondeterministic envelope, so the strict reordering gate also
failed. The exact rescale consume remains as a SASS-neutral helper; matched KPIPE2
Stage2/e16pc hashes were unchanged and all routes used 168 registers with no
selected spills.

The broad 57-cell retained-route matrix produced 37 robust measured FP4 wins, one
narrow win, 10 losses, and nine no-finite-FP4 cells. The 60-sample h32/s1024 result
was only `1.0078x` (`+0.416 us`). The measured boundary is H<=16 at S2048, H<=8 at
S4096, and H<=4 at S8192. S16384/32768 FP4 timed out independently in bounded
fresh processes while CuTe BF16 remained finite. Fullgrid repaired b4/s4096/h4
persistent liveness but still lost `0.170080` versus `0.145824` ms.

Representative loss profiles show the blocker is not the moved rescale wait.
h16/s4096 and h16/s8192 have a roughly 4.3k-cycle P chain, only 133-136 cycles from
issue-side P-ready to PV, long-scoreboard `3.46/3.51`, eligible warps `0.41/0.42`,
and TC activity `14.35/16.32%`. Closing measured deficits of 24 us through 889 us
requires a lower-cost P/softmax/PV or long-sequence decomposition, not another
issue-lane schedule.

Final default/off clean build succeeded with every requested gate zero. Ordinary
Stage2 h4/s1024 is finite/deterministic at `.051472/.046144` ms p50/min; timeline
is `[]`, P-chain raw is `[[]]`, and all 64 counters are zero. Rejected schedule
selectors/behavior were removed. External branch movement ended at
`ee3bd37cf76ea324c52c02f218daa67efe1f5c33` with origin matched. No commit or push
was performed.

## 2026-07-11/12 - Large-sequence and high-head adaptation

Task file: `session6_large_shape_high_head_adaptation_20260711.md`.

Report: `forward_large_shape_high_head_adaptation_20260711_report.md`.

Result: implemented and debugged the modern four-WG, two-producer score-derived
e16pc split-P route in full-P and legal K64-progressive forms. Retain split-full
only for `B=1,H=1,S>=16384`; keep K64 explicit/default-off and reject all high-head
split dispatch. On 60 samples, split-full is `.326560` versus old auto `.330544`
and BF16 `.400448` ms at S16384/H1, and `.623552` versus `.642288` and `.717200`
ms at S32768/H1.

- The required 22-cell matrix shows split-full/K64 are 10-14% slower than e16pc
  at high head counts. K64 moves legal PV issue about 180-240 cycles earlier but
  remains slower.
- NCU at H16/S4096, H16/S8192, and H32/S4096 shows split-full raises issue/eligible
  activity but lowers TC utilization and runs 11-12% longer. It uses 128 registers,
  two barriers, no stack/local spills, and 12208 B static shared.
- Per-half sparse clocks at idx0/2/16/31 and a 200-launch mixed-shape stress prove
  finite ownership/reuse with no run-to-run drift. The split idx0 producer chain is
  about 4.85k cycles versus e16pc's 4.13k, so the intended halved exposed chain was
  not achieved.
- Long-S fused execution is finite under fullgrid. A separate B4/S4096/H4 hang was
  traced to the persistent guard using `heads` instead of `batch*heads`; the
  batch-aware guard now forces finite FP4 fullgrid execution.
- Materialized-P was rejected after measurement: consumer-only floors were fast,
  but the producer timed out at H16/S4096 and returned non-finite LSE at H16/S8192
  and H16/S16384. No incomplete attention API was built.
- Final 57-cell regression: 40 robust wins, one parity, 16 losses, zero non-finite
  FP4 cells, and all 37 historical robust wins retained.

Final clean build has timeline/P-chain/KPIPE/score-reuse/selective/hotplate/counters
all zero. Low-shape, retained H1/S16384, and B4/S4096 smokes are finite; diagnostics
are empty and all 64 counters are zero. HEAD advanced externally from `d0e185e` to
`b8cc39d`; no commit or push was performed by this task.

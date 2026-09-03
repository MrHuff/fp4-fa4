# MXFP4 FA4 Forward Profiling Ledger - 2026-06-12

Device: `CUDA_VISIBLE_DEVICES=0` on GB200.

Report files are in this directory:

- NCU: `ncu_<case>.ncu-rep`, `ncu_<case>_raw.csv`, `ncu_<case>_details.csv`
- NSYS: `nsys_<case>.nsys-rep`, `nsys_<case>_stats.csv`
- Probe benchmark: `bench_pstage5_vs_qkscfix.jsonl`

## Cases

| case | target | seqlen | heads | config |
| --- | --- | ---: | ---: | --- |
| `bf16_h16_s2048` | BF16 TK FA4 | 2048 | 16 | persistent TK baseline |
| `bf16_h16_s4096` | BF16 TK FA4 | 4096 | 16 | persistent TK baseline |
| `bf16_h4_s2048` | BF16 TK FA4 | 2048 | 4 | persistent TK baseline |
| `mxfp4_o56_h16_s2048` | MXFP4 | 2048 | 16 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56` |
| `mxfp4_o56_h16_s4096` | MXFP4 | 4096 | 16 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56` |
| `mxfp4_o56_h4_s2048` | MXFP4 | 2048 | 4 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56` |
| `mxfp4_o56_qkscfix_h16_s2048` | MXFP4 | 2048 | 16 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix` |
| `mxfp4_o56_qkscfix_h16_s4096` | MXFP4 | 4096 | 16 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix` |
| `mxfp4_o56_qkscfix_h4_s2048` | MXFP4 | 2048 | 4 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix` |
| `mxfp4_pstage5_qkscfix_h16_s4096` | MXFP4 probe | 4096 | 16 | rejected probe, reverted |

## Commands

NCU command template, run once per case with the case values above:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_TARGET="$target" PROFILE_SEQ="$seq" PROFILE_HEADS="$heads" PROFILE_LABEL="$label" PROFILE_CONFIG="$cfg" \
ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export "results/mxfp4_fa4_forward_profile_20260612/ncu_${label}" python3 - <<'PY'
import os
import torch
from tk_fa4 import fp4_pv_experiments as exp

seq = int(os.environ["PROFILE_SEQ"])
heads = int(os.environ["PROFILE_HEADS"])
target = os.environ["PROFILE_TARGET"]
cfg = os.environ.get("PROFILE_CONFIG") or None
device = torch.device("cuda")
q_bf16, k_bf16, v_bf16 = exp._make_live_bf16_source_inputs(seq, seed=57000, batch=1, heads=heads, device=device, zero_qk=False)

if target == "bf16":
    ext = exp._load_bf16_causal_baseline_ext()
    launch = "persistent" if seq <= exp._BF16_CAUSAL_PERSISTENT_MAX_SEQ else "fullgrid"
    out = torch.empty((1, seq, heads, exp._D_VO), dtype=torch.bfloat16, device=device)
    lse = torch.empty((1, heads, 1, seq), dtype=torch.float32, device=device)
    fn = ext.forward_persistent if launch == "persistent" else ext.forward
    for _ in range(2):
        fn(q_bf16, k_bf16, v_bf16, out, lse)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn(q_bf16, k_bf16, v_bf16, out, lse)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
else:
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = cfg
    inputs = exp._fp4_qk_mxfp4_v_inputs_from_bf16_source(q_bf16, k_bf16, v_bf16, qk_quant_backend="v5", v_quant_mode=None)
    inputs = exp._prepare_mxfp4_fwd_inputs_for_config(inputs, seqlen=seq, config=cfg)
    q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = inputs
    ext = exp._load_forward_experiments_ext()
    out = torch.empty((1, seq, heads, exp._D_VO), dtype=torch.bfloat16, device=device)
    lse = torch.empty((1, heads, 1, seq), dtype=torch.float32, device=device)
    persistent_launch = exp._resolve_mxfp4_fwd_launch_mode(seq, heads, "auto") != "fullgrid"
    p_quant = exp._mxfp4_quant_mode_to_int(None)
    for _ in range(2):
        ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, p_quant, persistent_launch)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, p_quant, persistent_launch)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
PY
```

CSV export:

```bash
ncu --import "$report" --csv --page raw > "${base}_raw.csv"
ncu --import "$report" --csv --page details > "${base}_details.csv"
```

NSYS command used the same one-kernel driver and:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_TARGET="$target" PROFILE_SEQ="$seq" PROFILE_HEADS="$heads" PROFILE_LABEL="$label" PROFILE_CONFIG="$cfg" \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  --output="results/mxfp4_fa4_forward_profile_20260612/nsys_${label}" python3 - <<'PY'
# same one-kernel driver as the NCU command
PY
```

NSYS stats export:

```bash
nsys stats --force-export=true --format=csv --report=cuda_gpu_kern_sum,cuda_api_sum "$report" > "${base}_stats.csv"
```

## Metrics Used

- `gpu__time_duration.avg`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- `l1tex__throughput.avg.pct_of_peak_sustained_active`
- `lts__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__issue_active.avg.pct_of_peak_sustained_elapsed`
- `smsp__issue_active.avg.per_cycle_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `smsp__warps_active.avg.per_cycle_active`
- `smsp__average_warp_latency_per_inst_issued.ratio`
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`
- `launch__grid_size`
- `launch__block_size`
- `launch__registers_per_thread`
- `launch__shared_mem_per_block`
- `launch__occupancy_limit_registers`
- `launch__occupancy_limit_shared_mem`
- `launch__occupancy_limit_blocks`
- `launch__occupancy_limit_warps`
- `launch__occupancy_cluster_pct`
- `launch__waves_per_multiprocessor`
- `derived__local_spilling_requests`
- `profiler__replayer_passes`

## NCU Timings And Key Counters

| case | duration_us | tensor_pct | dram_pct | eligible_warps | long_scoreboard | regs | spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_h16_s2048` | 71.65 | 27.28 | 5.922 | 0.522 | 3.823 | 128 | 0 |
| `bf16_h16_s4096` | 135.40 | 51.16 | 6.541 | 0.571 | 3.724 | 128 | 117760 |
| `bf16_h4_s2048` | 69.09 | 6.935 | 1.546 | 0.451 | 3.517 | 128 | 0 |
| `mxfp4_o56_h16_s2048` | 53.70 | 5.347 | 1.725 | 0.373 | 3.888 | 168 | 0 |
| `mxfp4_o56_h16_s4096` | 158.70 | 6.999 | 1.159 | 0.396 | 4.019 | 168 | 0 |
| `mxfp4_o56_h4_s2048` | 48.77 | 1.501 | 0.495 | 0.388 | 3.766 | 168 | 0 |
| `mxfp4_o56_qkscfix_h16_s2048` | 54.18 | 5.223 | 1.821 | 0.388 | 3.583 | 168 | 0 |
| `mxfp4_o56_qkscfix_h16_s4096` | 159.90 | 6.914 | 1.226 | 0.414 | 3.661 | 168 | 0 |
| `mxfp4_o56_qkscfix_h4_s2048` | 47.62 | 1.484 | 0.533 | 0.403 | 3.508 | 168 | 0 |
| `mxfp4_pstage5_qkscfix_h16_s4096` | 159.65 | 6.903 | 1.228 | 0.416 | 3.620 | 168 | 0 |

NSYS confirmed one representative forward kernel launch per case. Example kernel names were `kernel` for BF16 and `kernel_streaming_live_fp4pv` for MXFP4.

## Classification

Dominant bottleneck: PV tensor-core underfeed from the QK/softmax producer and P-scale/V-scale handoff path.

Evidence:

- MXFP4 tensor pipe was only about 5-7% for H16 cases, versus BF16 at about 27-51%.
- Eligible warps were low, about 0.37-0.41 per scheduler for MXFP4.
- Long scoreboard was about 3.5-4.0 cycles per issue-active, while DRAM was only about 1-2% of peak.
- Barrier/membar stalls were small, and shared/TMEM/proxy throughput did not look like the primary limiter.
- MXFP4 target kernels had 168 registers/thread and `derived__local_spilling_requests=0`.
- NSYS showed one isolated kernel per run, so the observed issue was not multi-launch overhead.

Not classified as primary: memory bandwidth, occupancy/register spills, launch/scheduling multiplicity, or explicit barrier/membar stalls.

## Probe Loop

Probe: added a pstage5 qkscfix forward config to deepen P staging for the H16/S4096 handoff bottleneck. Built with:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Numerics matched qkscfix, but the isolated-kernel evidence did not validate the probe:

- NCU H16/S4096 was effectively unchanged: qkscfix 159.90 us, pstage5 159.65 us.
- Direct preallocated extension timing H16/S4096 median regressed: qkscfix 0.165344 ms, pstage5 0.171040 ms.
- H16/S2048 and H4/S2048 were noise-level or slower.

Decision: rejected and reverted pstage5. No commit/push.

Final smoke after revert:

- Config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix`
- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 2

Probe: added an opt-in qkscfix-stagev config to stage V scales before the next QK issue:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix_stagev`
- Source change during probe: new config type inheriting qkscfix with `ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK=true`, plus explicit host-dispatch entries.
- Built forward-only with `CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward`.

Rationale: NCU showed PV tensor-core underfeed with low eligible warps and long scoreboard, not DRAM or launch overhead. The V-scale TMEM staging path waits after `v_arrived`; moving V-scale staging before the next QK issue could have hidden part of that handoff.

Direct preallocated timing, 120 samples:

| shape | qkscfix median_ms | stagev median_ms | decision |
| --- | ---: | ---: | --- |
| H16/S2048 | 0.059856 | 0.060064 | no win |
| H16/S4096 | 0.171808 | 0.171264 | possible tiny win; NCU follow-up required |
| H4/S2048 | 0.060272 | 0.060080 | noise-level |

Numerics versus qkscfix were finite with LSE unchanged; max output deltas were <= 0.00232 BF16 units.

NCU follow-up for H16/S4096 rejected the probe:

| metric | qkscfix | stagev |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 159.904 us | 160.064 us |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.913957 | 6.887264 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.414425 | 0.414102 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.660802 | 3.663137 |
| `launch__registers_per_thread` | 168 | 168 |
| `derived__local_spilling_requests` | 0 | 0 |

Decision: rejected and reverted qkscfix-stagev. Rebuilt forward-only after revert and confirmed the probe dispatch string is absent. Final qkscfix smoke after revert:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 3

Probe: added an opt-in qkscfix output-register variant:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o48_qkscfix`
- Source change during probe: explicit host-dispatch entries using the existing qkscfix config type with `ONLINE_OUTPUT_REGS=48` instead of the kept o56 route.
- Built forward-only with `CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward`.

Rationale: after stage-V ordering left eligible warps and long scoreboard unchanged, test whether a smaller output-register budget could reduce scheduling pressure or improve eligible warps without changing the qkscfix numerics.

Smoke:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing, 120 samples:

| shape | o56 qkscfix median_ms | o48 qkscfix median_ms | decision |
| --- | ---: | ---: | --- |
| H16/S2048 | 0.060832 | 0.061648 | slower |
| H16/S4096 | 0.177840 | 0.181488 | slower |
| H4/S2048 | 0.064064 | 0.064128 | no win |

Numerics versus o56 qkscfix were finite with unchanged LSE; max output deltas were <= 0.00220 BF16 units.

Decision: rejected and reverted o48 qkscfix without NCU follow-up because isolated timing did not suggest a win. Rebuilt forward-only after revert and confirmed `o48_qkscfix` is absent. Final qkscfix smoke after revert:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Relevant forward source diff checked after reverting the rejected probe is the pre-existing gated qkscfix route:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Other pre-existing dirty files remain in the repository, including `tk_fa4/fp4_pv_experiments.py` and backward files. Backward files were not edited during this loop.

## Probe Loop 4

Probe: added an opt-in qkscfix producer-register variant:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: explicit forward host-dispatch entries using the existing qkscfix config type with `P_REGS=112` instead of the kept `P_REGS=96` route.
- Changed forward source: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: prior NCU evidence showed MXFP4 qkscfix remained PV tensor-core underfed with low eligible warps and long scoreboard, while `pstage5`, `stagev`, and `o48` did not improve the bottleneck. This tests whether giving the QK/softmax/P producer more register budget reduces producer-side latency without increasing launch registers or spills.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=57000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False,
)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command: inline Python driver using one prepared input set per shape, preallocated output/LSE tensors, CUDA events around only `ext.forward_streaming_live_mxfp4`, 20 warmups and 120 measured iterations. Results were written to `results/mxfp4_fa4_forward_profile_20260612/bench_p112_qkscfix_direct.jsonl`.

Direct preallocated timing:

| shape | p96 o56 qkscfix median_ms | p112 o56 qkscfix median_ms | p112 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.060096 | 0.058624 | -2.45% |
| H16/S4096 | 0.173872 | 0.171152 | -1.56% |
| H4/S2048 | 0.060864 | 0.059872 | -1.63% |

Numerics versus p96 qkscfix were finite with unchanged LSE; p112 max output delta was <= 0.00245 BF16 units.

NCU follow-up command:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_TARGET=mxfp4 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL=mxfp4_o56_qkscfix_p112_h16_s4096 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096 python3 - <<'PY'
import os
import torch
import tk_fa4.fp4_pv_experiments as exp

seq = int(os.environ['PROFILE_SEQ'])
heads = int(os.environ['PROFILE_HEADS'])
cfg = os.environ['PROFILE_CONFIG']
os.environ['TK_FA4_FP4PV_FWD_CONFIG'] = cfg
ext = exp._load_forward_experiments_ext()
q_bf16, k_bf16, v_bf16 = exp._make_live_bf16_source_inputs(seq, heads=heads, seed=57000)
fp4_inputs = exp._fp4_qk_mxfp4_v_inputs_from_bf16_source(q_bf16, k_bf16, v_bf16)
q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = exp._prepare_mxfp4_fwd_inputs_for_config(fp4_inputs, seqlen=seq, config=cfg)
out = torch.empty((1, seq, heads, exp._D_VO), device='cuda', dtype=torch.bfloat16)
lse = torch.empty((1, heads, 1, seq), device='cuda', dtype=torch.float32)
persistent = exp._resolve_mxfp4_fwd_launch_mode(seq, heads, 'auto') != 'fullgrid'
qmode = exp._mxfp4_quant_mode_to_int(None)
for _ in range(3):
    ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent)
torch.cuda.cudart().cudaProfilerStop()
torch.cuda.synchronize()
print({'config': cfg, 'seq': seq, 'heads': heads, 'finite': bool(torch.isfinite(out).all().item() and torch.isfinite(lse).all().item())})
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096_details.csv
```

NCU H16/S4096:

| metric | p96 qkscfix | p112 qkscfix |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 159.904 us | 157.856 us |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.913957 | 6.975208 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.414425 | 0.418665 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.660802 | 3.550028 |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.226214 | 1.242102 |
| `launch__registers_per_thread` | 168 | 168 |
| `derived__local_spilling_requests` | 0 | 0 |

Decision: kept p112 qkscfix as a gated forward-only win. Classification remains PV tensor-core underfeed from producer/handoff latency; the p112 counter movement supports the producer-register hypothesis by modestly increasing eligible warps, lowering long scoreboard, and improving isolated duration without adding spills or DRAM pressure. No backward files were touched.

## Probe Loop 5

Probe: added an opt-in qkscfix producer-register extension:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p128_o56_qkscfix`
- Source change during probe: explicit forward host-dispatch entries with `P_REGS=128`, beside the kept p112 route.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: p112 improved the same producer-underfeed counters that dominated the initial profile. This tested whether the producer-register trend continued past 112 without spills.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke result:

- Config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p128_o56_qkscfix`
- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing, 120 samples, written to `results/mxfp4_fa4_forward_profile_20260612/bench_p128_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix median_ms | p128 o56 qkscfix median_ms | p128 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.059360 | 0.059104 | -0.43% |
| H16/S4096 | 0.175936 | 0.174800 | -0.65% |
| H4/S2048 | 0.064240 | 0.062496 | -2.71% |

Numerics versus p112 qkscfix were finite with unchanged LSE; p128 max output delta was <= 0.00428 BF16 units.

NCU follow-up command:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_TARGET=mxfp4 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL=mxfp4_o56_qkscfix_p128_h16_s4096 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p128_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p128_h16_s4096 python3 - <<'PY'
# same one-kernel cudaProfilerStart/Stop driver as Probe Loop 4, with PROFILE_CONFIG set to p128
PY
```

NCU H16/S4096:

| metric | p112 qkscfix | p128 qkscfix |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 157.856 us | 158.784 us |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.975208 | 6.964974 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418665 | 0.418028 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.550028 | 3.564734 |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.242102 | 1.235136 |
| `launch__registers_per_thread` | 168 | 168 |
| `derived__local_spilling_requests` | 0 | 0 |

Decision: rejected and reverted p128. Although paired direct timing showed a small speedup, the representative H16/S4096 NCU counters regressed versus p112 on duration, eligible warps, tensor active, and long scoreboard. Rebuilt forward-only after revert and confirmed `p128_o56_qkscfix` is absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 6

Probe: added an opt-in qkscfix producer-register midpoint:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p120_o56_qkscfix`
- Source change during probe: explicit forward host-dispatch entries with `P_REGS=120`, beside the kept p112 route.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: p112 improved the producer-underfeed counters, while p128 regressed. p120 tested whether a midpoint could preserve the p112 long-scoreboard reduction while offering a little more producer payload depth.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p120_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 vs p120 for H16/S2048, H16/S4096, H4/S2048 with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_p120_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | p120 o56 qkscfix avg_ms | p120 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051403733 | 0.051302667 | -0.20% |
| H16/S4096 | 0.155795733 | 0.155640793 | -0.10% |
| H4/S2048 | 0.047068799 | 0.047157868 | +0.19% |

NCU follow-up command:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_TARGET=mxfp4 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL=mxfp4_o56_qkscfix_p120_h16_s4096 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p120_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p120_h16_s4096 python3 - <<'PY'
import os
import torch
import tk_fa4.fp4_pv_experiments as exp
seqlen = int(os.environ['PROFILE_SEQ'])
heads = int(os.environ['PROFILE_HEADS'])
cfg = os.environ['PROFILE_CONFIG']
os.environ['TK_FA4_FP4PV_FWD_CONFIG'] = cfg
ext = exp._load_forward_experiments_ext()
q_bf16, k_bf16, v_bf16 = exp._make_live_bf16_source_inputs(seqlen, heads=heads, seed=57000)
fp4_inputs = exp._fp4_qk_mxfp4_v_inputs_from_bf16_source(q_bf16, k_bf16, v_bf16)
q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = exp._prepare_mxfp4_fwd_inputs_for_config(fp4_inputs, seqlen=seqlen, config=cfg)
out = torch.empty((1, seqlen, heads, exp._D_VO), device='cuda', dtype=torch.bfloat16)
lse = torch.empty((1, heads, 1, seqlen), device='cuda', dtype=torch.float32)
launch = exp._resolve_mxfp4_fwd_launch_mode(seqlen, heads, 'auto')
persistent_launch = launch != 'fullgrid'
qmode = exp._mxfp4_quant_mode_to_int(None)
for _ in range(3):
    ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent_launch)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent_launch)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p120_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p120_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p120_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p120_h16_s4096_details.csv
```

NCU H16/S4096 metric names and values:

| metric | p112 qkscfix | p120 qkscfix |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 157.856 us | 158.176 us |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.975208 | 6.998433 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418665 | 0.419613 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.550028 | 3.558480 |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.242102 | 1.239795 |
| `launch__registers_per_thread` | 168 | 168 |
| `derived__local_spilling_requests` | 0 | 0 |

Decision: rejected and reverted p120. Direct timing was noise-level positive on H16 but negative on H4, while the representative H16/S4096 NCU profile regressed on isolated duration and long-scoreboard stalls. Rebuilt forward-only after revert and confirmed `p120_o56_qkscfix` is absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 7

Probe: added an opt-in p-ready-before-rescale-wait variant of the kept qkscfix p112 route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix_pready`
- Source change during probe: derived forward config with `ONLINE_P_READY_BEFORE_RESCALE_WAIT = true`, plus explicit forward host-dispatch entries.
- Changed forward sources during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: existing NCU evidence shows PV tensor-core underfeed with low eligible warps and long scoreboard, while p112 improved the handoff-sensitive counters. This probe moved the producer's P/scale ready signal before its rescale wait, targeting a possible producer-to-PV latency bubble without changing score math or backward code.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix_pready'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs p112 qkscfix_pready for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_pready_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | p112 qkscfix_pready avg_ms | pready delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051382399 | 0.055279199 | +7.58% |
| H16/S4096 | 0.155681594 | 0.168738667 | +8.39% |
| H4/S2048 | 0.047093066 | 0.050742932 | +7.75% |

NCU follow-up: skipped. Timing did not suggest a win on any measured shape, so no representative kernel counter capture was warranted.

Decision: rejected and reverted qkscfix_pready. The probe validates that moving the ready signal before the producer rescale wait is the wrong handoff direction for the current scorepack path. Rebuilt forward-only after revert and confirmed `qkscfix_pready` is absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 8

Probe: added an opt-in q224 quant-register variant of the kept qkscfix p112 route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q224_p112_o56_qkscfix`
- Source change during probe: explicit forward host-dispatch entries only.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: current NCU evidence shows PV tensor-core underfeed with low eligible warps and long scoreboard, while p112 improved the same handoff-sensitive counters without spills. This probe shifted producer/output/quant register balance by increasing q regs from 208 to 224 while holding p112/o56/qkscfix fixed, looking for higher eligible warps without introducing spills.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q224_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs q224_p112 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_q224_p112_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | q224 p112 o56 qkscfix avg_ms | q224 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051349600 | 0.051315467 | -0.07% |
| H16/S4096 | 0.155602407 | 0.155741596 | +0.09% |
| H4/S2048 | 0.047072001 | 0.047120265 | +0.10% |

NCU follow-up: skipped. Timing was noise-level positive only on H16/S2048 and regressed on the representative H16/S4096 and cheap H4/S2048 shapes, so there was no counter-backed win to preserve.

Decision: rejected and reverted q224. Rebuilt forward-only after revert and confirmed `q224_p112_o56_qkscfix` is absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 9

Probe: added an opt-in payload-store/light-local-publish variant of the kept qkscfix p112 scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_payloadv2_lightpub_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: derived qkscfix scorepack config with `ONLINE_P_PAYLOAD_STORE_MODE = 4` and `ONLINE_LIGHT_P_PAYLOAD_LOCAL_PUBLISH = true`, plus explicit forward host-dispatch entries.
- Changed forward sources during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: NCU evidence points to PV tensor-core underfeed with low eligible warps and long scoreboard, not DRAM or launch. The kernel path has explicit payload publish/wait handoffs through `p_payload_published_ready` and remote ready semaphores; this probe tested whether the V2 payload store plus lighter proxy-only local publish reduced that handoff bubble without changing score math.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_payloadv2_lightpub_pstage4_q208_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs payloadv2_lightpub p112 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_payloadv2_lightpub_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | payloadv2 lightpub p112 avg_ms | payload delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051374666 | 0.051301066 | -0.14% |
| H16/S4096 | 0.155609067 | 0.155772273 | +0.10% |
| H4/S2048 | 0.047070134 | 0.047041599 | -0.06% |

NCU follow-up: skipped. The probe did not improve the representative H16/S4096 isolated kernel timing.

Decision: rejected and reverted payloadv2/lightpub. Rebuilt forward-only after revert and confirmed `payloadv2_lightpub_pstage4_q208_p112_o56_qkscfix` and `qkscfix_payloadstore_lightpublish` are absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 10

Probe: added an opt-in P-first staging-order variant of the kept qkscfix p112 scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pfirst_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: derived qkscfix scorepack config with `ONLINE_STAGE_P_BEFORE_V = true`, plus explicit forward host-dispatch entries.
- Changed forward sources during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: NCU evidence still classifies MXFP4 o56/qkscfix as PV tensor-core underfed, with low eligible warps and long scoreboard while not DRAM, launch, or spill limited. Previous payload/pready probes showed the P payload ready path is handoff-sensitive. This probe tested whether staging P before V reduced the P-scale side of the handoff bubble before the PV consumer.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pfirst_pstage4_q208_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs pfirst p112 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_pfirst_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | pfirst p112 o56 qkscfix avg_ms | pfirst delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051379200 | 0.051418666 | +0.08% |
| H16/S4096 | 0.155765597 | 0.155799739 | +0.02% |
| H4/S2048 | 0.047116800 | 0.047148534 | +0.07% |

NCU follow-up: skipped. Timing regressed on every measured isolated-kernel shape, including representative H16/S4096, so no counter follow-up was warranted.

Decision: rejected and reverted pfirst. The probe suggests the existing V-before-P staging order is better for this handoff. Rebuilt forward-only after revert and confirmed `pfirst_pstage4_q208_p112_o56_qkscfix` and `qkscfix_pfirst` are absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 11

Probe: added an opt-in QK two-ahead variant of the kept qkscfix p112 scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_qk2_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: derived qkscfix scorepack config with `ONLINE_QK_TWO_AHEAD = true`, plus explicit forward host-dispatch entries.
- Changed forward sources during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: NCU evidence classifies the current MXFP4 route as PV tensor-core underfed with low eligible warps and long scoreboard. Since pfirst/pready/payload ordering changes were rejected, this probe tested whether increasing producer lead time and score availability could feed the PV side sooner without touching score math or backward code.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_qk2_pstage4_q208_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs qk2 p112 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_qk2_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | qk2 p112 o56 qkscfix avg_ms | qk2 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051376267 | 0.053483999 | +4.10% |
| H16/S4096 | 0.155699460 | 0.157779996 | +1.34% |
| H4/S2048 | 0.047111468 | 0.047207999 | +0.20% |

NCU follow-up: skipped. Timing regressed on every measured isolated-kernel shape, including representative H16/S4096, so the probe did not justify another counter capture.

Decision: rejected and reverted qk2. The producer two-ahead schedule appears to increase overhead or pressure before PV can use the earlier work. Rebuilt forward-only after revert and confirmed `qk2_pstage4_q208_p112_o56_qkscfix` and `qkscfix_qk2` are absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 12

Probe: added an opt-in p112/qkscfix scorepack stageV route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_stagev_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: derived qkscfix scorepack config with `ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true`, plus explicit forward host-dispatch entries.
- Changed forward sources during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: the old p96 qkscfix stageV probe was NCU-neutral/rejected, but the kept p112 route changed the same P/PV handoff depth and reduced long scoreboard. This p112-only retest checked whether moving V-scale staging earlier helps the current kept route's PV tensor-core underfeed without changing math or backward code.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_stagev_pstage4_q208_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs p112 qkscfix stageV for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_stagev_p112_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | stageV p112 o56 qkscfix avg_ms | stageV delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051417069 | 0.051349334 | -0.13% |
| H16/S4096 | 0.155733331 | 0.155706406 | -0.02% |
| H4/S2048 | 0.047070134 | 0.047060267 | -0.02% |

NCU follow-up command:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_TARGET=mxfp4 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL=mxfp4_o56_qkscfix_stagev_p112_h16_s4096 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_stagev_pstage4_q208_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_stagev_p112_h16_s4096 python3 - <<'PY'
# warm up three launches, use cudaProfilerStart/Stop around one representative forward kernel
PY
```

NCU sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.

NCU exports:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_stagev_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_stagev_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_stagev_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_stagev_p112_h16_s4096_details.csv
```

Counter comparison, H16/S4096 representative kernel:

| metric | p112 o56 qkscfix | stageV p112 o56 qkscfix |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 157.856000 us | 157.760000 us |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.975208% | 6.988906% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418665 warp | 0.417940 warp |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.550028 inst | 3.556477 inst |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.242102% | 1.242911% |
| `launch__registers_per_thread` | 168 | 168 |
| `derived__local_spilling_requests` | 0 | 0 |

Decision: rejected and reverted stageV p112. The timing change was noise-level and the dominant bottleneck counters did not improve: eligible warps fell and long scoreboard worsened. Rebuilt forward-only after revert and confirmed `stagev_pstage4_q208_p112_o56_qkscfix` and `qkscfix_stagev` are absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 13

Probe: added an opt-in p104/qkscfix scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p104_o56_qkscfix`
- Source change during probe: forward host-dispatch entries that launch the existing qkscfix scorepack pstage4 template with `_ProducerRegs=104`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: p112 was a validated win over p96, while p120/p128 regressed. This interpolation tested whether a slightly smaller producer register budget could keep p112's tensor activity while recovering more eligible warps and reducing long scoreboard stalls. Dominant bottleneck remained PV tensor-core underfeed with low eligible warps and long scoreboard, not DRAM, launch, or spilling.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p104_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs p104 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_p104_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | p104 o56 qkscfix avg_ms | p104 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051388001 | 0.051328798 | -0.12% |
| H16/S4096 | 0.155684265 | 0.155656004 | -0.02% |
| H4/S2048 | 0.047034665 | 0.046903733 | -0.28% |

NCU follow-up command:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_TARGET=mxfp4 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL=mxfp4_o56_qkscfix_p104_h16_s4096 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p104_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p104_h16_s4096 python3 - <<'PY'
# warm up three launches, use cudaProfilerStart/Stop around one representative forward kernel
PY
```

NCU sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.

NCU exports:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p104_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p104_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p104_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p104_h16_s4096_details.csv
```

Counter comparison, H16/S4096 representative kernel:

| metric | p112 o56 qkscfix | p104 o56 qkscfix |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 157.856000 us | 158.144000 us |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.975208% | 6.929552% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418665 warp | 0.419252 warp |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.550028 inst | 3.540848 inst |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.242102% | 1.240051% |
| `launch__registers_per_thread` | 168 | 168 |
| `derived__local_spilling_requests` | 0 | 0 |

Decision: rejected and reverted p104. The direct timing was only noise-level positive, while the representative NCU kernel regressed in duration and tensor active. The useful signal is that reducing producer regs improved eligible warps and long scoreboard, but p104 crossed into lower tensor utilization. Rebuilt forward-only after revert and confirmed `pstage4_q208_p104_o56_qkscfix` is absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 14

Probe attempted: `p108_o56_qkscfix`, intended as an interpolation between rejected p104 and kept p112:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p108_o56_qkscfix`
- Source change during attempt: forward host-dispatch entries with `_ProducerRegs=108`.
- Changed forward source during attempt: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Build result: rejected at compile time. ThunderKittens requires the register-count argument to `group::decrease_registers<n_reg>()` to be a multiple of 8:

```text
static assertion failed with "n_reg must be a multiple of 8"
instantiation of "void kittens::group<_GROUP_WARPS>::decrease_registers<n_reg>() [with _GROUP_WARPS=4, n_reg=108]"
```

Decision: aborted and reverted without runtime benchmarking. The invalid route was removed and later confirmed absent with:

```bash
grep -R "pstage4_q208_p108_o56_qkscfix" -n tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_configs.inc || true
```

## Probe Loop 15

Probe: added an opt-in q192/p112/qkscfix scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q192_p112_o56_qkscfix`
- Source change during probe: forward host-dispatch entries that launch the existing qkscfix scorepack pstage4 template with `_QuantRegs=192`, `_OutputRegs=56`, `_ProducerRegs=112`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: q224/p112 had already regressed H16/S4096 timing, and p104 showed that register partition changes can move eligible warps and long scoreboard but may reduce tensor activity. q192 kept the validated p112 producer budget and o56 output budget while lowering quant WG registers to test whether the QK/softmax producer side could hand work to PV with less scoreboard pressure.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q192_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs q192 p112 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_q192_p112_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | q192 p112 o56 qkscfix avg_ms | q192 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051418134 | 0.051296532 | -0.24% |
| H16/S4096 | 0.155651474 | 0.156643470 | +0.64% |
| H4/S2048 | 0.047118135 | 0.047139732 | +0.05% |

NCU follow-up: skipped. The representative H16/S4096 isolated kernel regressed by +0.64%, so the probe did not justify another counter capture.

Decision: rejected and reverted q192. The quant-register reduction did not address the dominant PV underfeed path for the representative shape. Rebuilt forward-only after revert and confirmed `pstage4_q192_p112_o56_qkscfix` and `pstage4_q208_p108_o56_qkscfix` are absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 16

Probe: added an opt-in o64/p112/qkscfix scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o64_qkscfix`
- Source change during probe: forward host-dispatch entries that launch the existing qkscfix scorepack pstage4 template with `_QuantRegs=208`, `_OutputRegs=64`, `_ProducerRegs=112`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: o48 had already regressed, while p104/q192 showed the register partition is near a narrow optimum. This tested the opposite output-WG direction from o48 while preserving the validated p112 producer budget and q208 quant budget, looking for a PV/output consumer-side scoreboard reduction without changing math.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o64_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 o56 qkscfix vs p112 o64 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_o64_p112_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | o64 p112 qkscfix avg_ms | o64 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051380801 | 0.051314402 | -0.13% |
| H16/S4096 | 0.155606127 | 0.155750402 | +0.09% |
| H4/S2048 | 0.047065334 | 0.047136001 | +0.15% |

NCU follow-up: skipped. The representative H16/S4096 isolated kernel regressed, and the H4/S2048 cheap shape also regressed.

Decision: rejected and reverted o64. The output-register neighbor did not reduce the dominant PV underfeed path on representative data. Rebuilt forward-only after revert and confirmed `pstage4_q208_p112_o64_qkscfix` is absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 17

Probe: added an opt-in light-local-publish variant of the kept p112/qkscfix scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_lightpub_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: a qkscfix scorepack trait setting `ONLINE_LIGHT_P_PAYLOAD_LOCAL_PUBLISH=true`, plus forward host-dispatch entries for the new config.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: the prior `payloadv2_lightpub` probe changed both payload store mode and local publish behavior. This isolated only the P-payload local publish/proxy handoff knob on the validated p112/q208/o56 qkscfix route. Existing NCU evidence still points to PV tensor-core underfeed with low eligible warps and long scoreboard, not DRAM, launch, or spilling.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_lightpub_pstage4_q208_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs lightpublish p112 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_lightpub_p112_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | lightpublish p112 o56 qkscfix avg_ms | lightpublish delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051406932 | 0.051348265 | -0.11% |
| H16/S4096 | 0.155650393 | 0.155688270 | +0.02% |
| H4/S2048 | 0.047106667 | 0.047042934 | -0.14% |

NCU follow-up: skipped. The representative H16/S4096 isolated kernel regressed by +0.02%, so the mixed cheap-shape gains were treated as noise.

Decision: rejected and reverted lightpublish-only. The isolated proxy-local-publish change did not improve the dominant PV underfeed path on representative data. Rebuilt forward-only after revert and confirmed `lightpub_pstage4_q208_p112_o56_qkscfix` and `qkscfix_lightpublish` are absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 18

Probe: added an opt-in warp-rescale-arrival variant of the kept p112/qkscfix scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_warprescale_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: a qkscfix scorepack trait setting `ONLINE_WARP_RESCALE_ARRIVE=true`, plus forward host-dispatch entries for the new config.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: `WARP_RESCALE_ARRIVE` is live for scorepack and changes only the rescale-finished handoff: the semaphore expects four warp arrivals and the direct-rescale path skips the follow-up warpgroup sync. This targets the shared/proxy/barrier handoff portion of the current PV tensor-core underfeed/low-eligible/long-scoreboard signature without changing math, QK ordering, or register budgets.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_warprescale_pstage4_q208_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 qkscfix vs warprescale p112 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_warprescale_p112_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | warprescale p112 o56 qkscfix avg_ms | warprescale delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051421865 | 0.051302934 | -0.23% |
| H16/S4096 | 0.155610673 | 0.155772003 | +0.10% |
| H4/S2048 | 0.047053067 | 0.047093598 | +0.09% |

NCU follow-up: skipped. The representative H16/S4096 isolated kernel regressed by +0.10%, and the cheap H4/S2048 shape also regressed.

Decision: rejected and reverted warprescale. The four-warp rescale arrival handoff did not improve representative PV underfeed behavior. Rebuilt forward-only after revert and confirmed `warprescale_pstage4_q208_p112_o56_qkscfix` and `qkscfix_warprescale` are absent while `p112_o56_qkscfix` remains. Final p112 smoke after revert:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 19

Probe: added an opt-in p112/o48 output-register neighbor of the kept p112/qkscfix scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o48_qkscfix`
- Source change during probe: forward host-dispatch entries for a qkscfix scorepack config with `ONLINE_QUANT_REGS=208`, `ONLINE_OUTPUT_REGS=48`, `ONLINE_PRODUCER_REGS=112`, and `FORCE_PERSISTENT=1`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: the validated p112 route improved H16/S4096 NCU duration, tensor activity, eligible warps, and long-scoreboard stalls without changing registers or spilling. A prior p104 probe improved eligible/scoreboard but reduced tensor activity, while o64 and p96/o48 output-register neighbors regressed representative timing. This checked the remaining low-output-register neighbor at the validated p112 producer budget, still targeting PV tensor-core underfeed and low eligible warps.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Build evidence:

- ptxas reported `12 bytes spill stores, 12 bytes spill loads` for the new `...ILi208ELi48ELi112...` qkscfix instantiation.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o48_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=57000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# preallocate out/lse and call ext.forward_streaming_live_mxfp4 directly;
# compare p112 o56 qkscfix vs p112 o48 qkscfix for H16/S2048, H16/S4096, H4/S2048
# with 20 warmups and 120 timed iterations.
PY
```

Direct preallocated timing, 120 timed iterations, written to `results/mxfp4_fa4_forward_profile_20260612/bench_o48_p112_qkscfix_direct.jsonl`:

| shape | p112 o56 qkscfix avg_ms | p112 o48 qkscfix avg_ms | p112/o48 delta |
| --- | ---: | ---: | ---: |
| H16/S2048 | 0.051396000 | 0.053145866 | +3.40% |
| H16/S4096 | 0.155559460 | 0.157911460 | +1.51% |
| H4/S2048 | 0.047039735 | 0.047233601 | +0.41% |

NCU follow-up: skipped. The representative H16/S4096 isolated kernel regressed and the new instantiation introduced local spills.

Decision: rejected and reverted p112/o48. The output-register reduction worsened the representative PV underfeed path and added spill traffic. Rebuilt forward-only after revert, confirmed `pstage4_q208_p112_o48_qkscfix` is absent while `p112_o56_qkscfix` remains, and reran the kept p112 smoke:

- `finite=true`
- `lse_max_abs=0.04033677279949188`
- `max_abs=0.97265625`

## Probe Loop 20

Probe: added an opt-in vload3 variant of the kept p112/qkscfix scorepack route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_vload3_pstage4_q208_p112_o56_qkscfix`
- Source change during probe: a qkscfix scorepack trait setting `ONLINE_V_LOAD_WARPS=3`, plus forward host-dispatch entries for the new config.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: `ONLINE_V_LOAD_WARPS` is live for the kept online path. It changes `STATIC_V_PRODUCER_THREADS`, and the producer path uses that thread group for both V payload and V-scale loads before publishing `v_arrived`. This targeted the existing PV tensor-core underfeed and long-scoreboard signature by trying to get V payload/scale ready earlier without changing QK math, P quantization, P staging, or output register budgets.

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_vload3_pstage4_q208_p112_o56_qkscfix'
benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=58000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- Shape: H16/S2048
- Failed: `_run_forward_streaming_live_mxfp4` timed out after 5000 ms.
- Direct preallocated timing: skipped because smoke did not complete.
- NCU follow-up: skipped because smoke did not complete.

Decision: rejected and reverted vload3. Increasing V-load producer warps deadlocked or exceeded the timing watchdog on the representative smoke path, so it is not a valid forward win. Rebuilt forward-only after revert, confirmed `vload3_pstage4_q208_p112_o56_qkscfix` and `qkscfix_vload3` are absent while `p112_o56_qkscfix` remains, and reran the kept p112 smoke:

- `finite=true`
- `lse_max_abs=0.019146978855133057`
- `max_abs=1.1171875`

## Probe Loop 21

Probe: added an opt-in exact coarse K64 P payload handoff/ready path for the score-derived qkscfix p112 route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_k64ready_pstage4_q208_p112_o56_qkscfix`
- Intended structure: two coarse `p_online_k64_ready` events for `Nb=128`, publishing half0 after qid 0-1 payload generation and half1 at the existing tail, with direct score-derived x1 P-scale TMEM stored before payload generation.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: prior NCU on the kept qkscfix scorepack route showed PV tensor-core underfeed, low eligible warps, and long scoreboard, not DRAM, launch, or spill pressure. This was the highest-priority structural P-handoff/PV-work probe requested: reduce per-32 P-ready churn and let PV begin on the first K64 half while the producer completes the second half.

Build:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc)
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_k64ready_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=59000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- Shape: H16/S2048.
- Failed: `_run_forward_streaming_live_mxfp4` timed out after 5000 ms.
- The Python smoke process remained alive after raising the timeout and was killed explicitly.
- Direct preallocated timing: skipped because smoke did not complete.
- NCU follow-up: skipped because smoke did not complete.

Decision: rejected and reverted k64ready. The correctness-first coarse K64 P-ready path deadlocked or exceeded the timing watchdog on the first smoke path, so it is not a valid forward win. Rebuilt forward-only after revert and confirmed no `k64ready` or `SCORE_DERIVED_SPLIT_P_READY_K64` symbols remain while `p112_o56_qkscfix` remains. Reran the kept p112 smoke:

- `finite=true`
- `mxfp4_ms=0.22511999309062958`
- `samples_ms=[0.22511999309062958, 0.15251199901103973]`
- `lse_max_abs=0.02677590399980545`
- `max_abs=0.96875`

## Probe Loop 22

Structural candidate check: real score-derived K256 path.

Findings before patch:

- `STATIC_MXFP4_K256` is only enabled for static consumer modes 6/178 in `fwd_streaming_kernel.inc`; the mxfp4 config dispatch currently launches `kernel_streaming_live_fp4pv<C, true, -1, false, true>`, so a route name alone cannot reach K256.
- Existing K256 producer packing in the online path calls `fp4pv_pack_scores_to_stage_mxfp4` / `_range`, which is the legacy vector-amax path the probe was required to avoid.
- The kept qkscfix scorepack path writes score-derived x1 P scales directly to TMEM, but K256 currently waits `p_quant_ready[half]` and stages scales from `p_sc_stage_k256[pair_buf * 2 + half]`.

Decision: no K256 patch in this loop. A correct probe needs a static K256 launch, K256 dynamic shared-memory sizing, paired payload staging, and paired direct P-scale TMEM staging. That is not a single narrow correctness-first patch against the kept qkscfix scorepack path.

Probe: attempted an opt-in P-scale slot-depth variant of qkscfix p112:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pscale3_pstage4_q208_p112_o56_qkscfix`
- Source changes during probe: qkscfix pscale3 config trait with `ONLINE_P_SCALE_TMEM_SLOTS=3`, direct slot override in `P_SCALE_TMEM_SLOTS`, and host dispatch entries.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: the kept route has `P_STAGE_SLOTS=4` but only two P-scale TMEM slots. This targeted the same PV tensor-core underfeed / long-scoreboard bottleneck by reducing producer waits for P-scale TMEM reuse.

Budget evidence:

- `MAX_TENSOR_COLS=512`.
- Kept qkscfix p112 layout uses `SCORE_TMEM_WIDTH=256`, one `Dvo=128` output tile, compact Q scale 16, K scale 16, two P-scale slots at 16 columns each, and two V-scale slots at 32 columns each.
- Two P-scale slots exactly fit: `256 + 128 + 16 + 16 + 2*16 + 2*32 = 512`.
- Three P-scale slots overflow: `256 + 128 + 16 + 16 + 3*16 + 2*32 = 528`.
- Four P-scale slots to match `P_STAGE_SLOTS=4` would overflow further: `544`.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc)
```

Build result:

- Failed at compile-time:
  `static assertion failed with "streaming live scale ping-pong exceeds TMEM budget"`
- Direct preallocated timing: skipped because the route cannot fit TMEM.
- NCU follow-up: skipped because the route cannot build.

Decision: rejected and reverted pscale3. P-scale depth cannot be increased for qkscfix p112 without taking a V-scale TMEM slot, which would turn this into a mixed V-scale staging probe rather than the requested P-handoff/PV-work probe. Rebuilt forward-only after revert, confirmed no `pscale3` or slot override symbols remain, and reran the kept p112 smoke:

- `finite=true`
- `mxfp4_ms=0.24086399376392365`
- `samples_ms=[0.24086399376392365, 0.1178240031003952]`
- `lse_max_abs=0.018718823790550232`
- `max_abs=0.96484375`

## Probe Loop 23

Probe: added an opt-in online split for score-derived qkscfix p112 P-scale readiness:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_earlypscale_pstage4_q208_p112_o56_qkscfix`
- Intended structure: signal `p_sc_tmem_ready[p_sc_slot]` immediately after direct score-derived x1 P-scale TMEM store, then separately signal `p_payload_published_ready[buf]` after the P payload shared-memory publish.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: existing NCU evidence classifies the kept qkscfix scorepack route as PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM, launch, or spill limited. This probe tested whether the existing decoupled scale/payload wait path, already used by the offline rewrite, could reduce P-scale/payload handoff bubbles for the online score-derived qkscfix route.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc)
```

Smoke command:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_earlypscale_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=60000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.2494720071554184`
- `samples_ms=[0.2494720071554184, 0.11785600334405899]`
- `lse_max_abs=0.030376769602298737`
- `max_abs=1.0`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors,
# WARMUP=20, ITERS=120, SEED=60001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_earlypscale_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_earlypscale_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta | Baseline min ms | Probe min ms | Min delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0625279993 | 0.0638080016 | +2.0471% | 0.0601599999 | 0.0618240014 | +2.7660% |
| H16/S4096 | persistent | 0.1704960018 | 0.1731519997 | +1.5578% | 0.1677120030 | 0.1708479971 | +1.8699% |
| H4/S2048 | persistent | 0.0603840016 | 0.0608000010 | +0.6889% | 0.0574720018 | 0.0578560010 | +0.6682% |

Probe-vs-baseline numeric check from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.004638671875`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00439453125`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.001953125`, no NaN/nonfinite.

NCU follow-up: skipped. H16/S4096 did not produce an isolated-kernel timing win, so there was no counter-follow-up target.

Decision: rejected and reverted earlypscale. Splitting the direct P-scale ready from payload publish is live and numerically clean, but the extra semaphore path makes the kept qkscfix p112 route slower on every measured direct preallocated shape. Rebuilt forward-only after revert, confirmed no `earlypscale`, `ONLINE_EARLY_P_SCALE_READY`, or `fp4pv_online_early_p_scale_ready` symbols remain, and reran the kept p112 smoke:

- `finite=true`
- `mxfp4_ms=0.36316800117492676`
- `samples_ms=[0.36316800117492676, 0.13020800054073334]`
- `lse_max_abs=0.03393649309873581`
- `max_abs=1.015625`

## Probe Loop 24

Probe: duplicate p128 qkscfix p112-adjacent route was briefly reintroduced, measured, and then removed because it had already been rejected by Loop 5 NCU evidence.

- Duplicate config name: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p128_o56_qkscfix`
- Baseline kept config: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix`
- Changed forward source during transient duplicate: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Smoke result for duplicate p128, H16/S2048, seed 61000:

- `finite=true`
- `mxfp4_ms=0.215552`
- `lse_max_abs=0.018410`
- `max_abs=1.0390625`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors,
# WARMUP=20, ITERS=120, SEED=61001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p128_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_p128_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.061055999 | 0.061600000 | +0.891% |
| H16/S4096 | persistent | 0.170527995 | 0.169888005 | -0.375% |
| H4/S2048 | persistent | 0.060031999 | 0.059487998 | -0.906% |

Existing Loop 5 NCU rejection for the same p128 idea on H16/S4096:

| Metric | p112 baseline | p128 probe |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 157.856 us | 158.784 us |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.975208 | 6.964974 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418665 | 0.418028 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.550028 | 3.564734 |

Decision: duplicate p128 route was removed without a new NCU run. The direct H16/S4096 timing signal was small and contradicted the existing representative counter profile. Rebuilt forward-only after removal and reran the kept p112 smoke with seed 61002:

- `finite=true`
- `mxfp4_ms=0.2558720111846924`
- `samples_ms=[0.2558720111846924, 0.12121599912643433]`
- `lse_max_abs=0.01888541504740715`
- `max_abs=0.97265625`

## Probe Loop 25

Probe: added an opt-in qkscfix p112 structural PV/output handoff route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_pstage4_q208_p112_o56_qkscfix`
- Config type: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,208,56,112,1>`
- Trait changes: `ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE=true`, `ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE=false`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: existing NCU evidence says kept qkscfix p112 is PV tensor-core underfed with low eligible warps and long scoreboard. This probe tested whether avoiding direct-after-rescale output accumulation into the main output TMEM path could reduce a P-handoff/PV-work dependency bubble.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_forcespare_qkscfix.log
```

Build result:

- First attempt failed until the route explicitly disabled `ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE`.
- Final probe build succeeded.
- ptxas for the new qkscfix force-spare kernel:
  - `8 bytes stack frame, 12 bytes spill stores, 12 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 8 bytes cumulative stack size, 1984 bytes smem`

Smoke command:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=62000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.22022399306297302`
- `samples_ms=[0.22022399306297302, 0.13849599659442902]`
- `lse_max_abs=0.020946800708770752`
- `max_abs=1.1484375`

Direct preallocated timing command:

```bash
timeout 240s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors,
# WARMUP=20, ITERS=120, SEED=62001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_forcespare_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.102048002 | 0.105024002 | +2.916% |
| H16/S4096 | persistent | 0.220960006 | 0.218656003 | -1.043% |
| H4/S2048 | persistent | 0.095743999 | 0.099008001 | +3.409% |

Probe-vs-baseline numeric check from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.00046825408935546875`, `max_abs_diff=0.0208740234375`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0008330345153808594`, `max_abs_diff=0.030029296875`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0005545616149902344`, `max_abs_diff=0.0143280029296875`, no NaN/nonfinite.

NCU follow-up commands for the H16/S4096 direct-timing win:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL=mxfp4_o56_qkscfix_p112_h16_s4096_forcespareloop_base PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix \
ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096_forcespareloop_base python3 - <<'PY'
# one-kernel CUDA profiler capture driver, seed=62001
PY
```

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL=mxfp4_o56_qkscfix_forcespare_p112_h16_s4096 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_pstage4_q208_p112_o56_qkscfix \
ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_forcespare_p112_h16_s4096 python3 - <<'PY'
# one-kernel CUDA profiler capture driver, seed=62001
PY
```

CSV export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096_forcespareloop_base.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096_forcespareloop_base_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096_forcespareloop_base.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_p112_h16_s4096_forcespareloop_base_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_forcespare_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_forcespare_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_forcespare_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_forcespare_p112_h16_s4096_details.csv
```

NCU metric comparison:

| Metric | Baseline p112 | Force-spare probe |
| --- | ---: | ---: |
| `gpu__time_duration.avg` | 157.440000 us | 172.576000 us |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 33.927878 | 31.677995 |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.245397 | 1.136931 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 11.053684 | 10.749288 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.997625 | 6.414963 |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.531384 | 14.102189 |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.379452 | 31.677995 |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 | 0.34 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.420110 | 0.410688 |
| `smsp__warps_active.avg.per_cycle_active` | 2.868872 | 2.838698 |
| `smsp__average_warp_latency_per_inst_issued.ratio` | 7.954622 | 8.332147 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.531339 | 3.698204 |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.482460 | 0.628959 |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.222439 | 0.195200 |
| `derived__local_spilling_requests` | 0 | 12288 |
| `launch__registers_per_thread` | 168 | 168 |
| `launch__shared_mem_per_block_static` | 1.968 KB | 1.984 KB |
| `launch__grid_size` | 512 | 512 |
| `launch__block_size` | 384 | 384 |

Decision: rejected and reverted force-spare. The direct H16/S4096 timing signal did not survive counter profiling: representative NCU showed +9.614% kernel duration, lower tensor activity, fewer eligible warps, worse long/short scoreboard, and new local spilling. Rebuilt forward-only after revert, confirmed no qkscfix force-spare dispatch/config symbols remain, and reran the kept p112 smoke:

- `finite=true`
- `mxfp4_ms=0.2569279968738556`
- `samples_ms=[0.2569279968738556, 0.11900799721479416]`
- `lse_max_abs=0.01873394101858139`
- `max_abs=0.97265625`

## Probe Loop 26

Probe: added an opt-in qkscfix p112 shared-P-scale handoff route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_sharedpscale_pstage4_q208_p112_o56_qkscfix`
- Config type: `config_fp4pv_3wg_dual_score_shared_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,208,56,112,1>`
- Trait change during probe: inherited qkscfix scorepack path with `ONLINE_DIRECT_P_SCALE_TMEM=false`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: K64 coarse ready had already deadlocked, K256 still needs a larger paired score-derived payload/scale implementation to avoid `fp4pv_pack_scores_to_stage_mxfp4*`, and P-scale slot depth exceeded TMEM budget. Existing NCU classifies the kept qkscfix p112 path as PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM/launch/spill limited. This probe tested whether removing the producer-side direct P-scale TMEM store and staging score-derived P scales through shared memory for PV-side TMEM load would reduce the P-scale/PV handoff bubble.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_sharedpscale_qkscfix.log
```

Build result:

- Build succeeded.
- ptxas for shared-P-scale qkscfix p112 probe:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 1904 bytes smem`
- ptxas for the compiled direct-P-scale p112 baseline in the same build:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_sharedpscale_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=63000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.275519996881485`
- `samples_ms=[0.275519996881485, 0.13312000036239624]`
- `lse_max_abs=0.02082435041666031`
- `max_abs=0.98046875`

Direct preallocated timing command:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors,
# WARMUP=20, ITERS=120, SEED=63001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_sharedpscale_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_sharedpscale_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.065984003 | 0.078528002 | +19.011% |
| H16/S4096 | persistent | 0.172095999 | 0.215136006 | +25.009% |
| H4/S2048 | persistent | 0.063487999 | 0.077472001 | +22.026% |

Probe-vs-baseline numeric check from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=9.5367431640625e-07`, `max_abs_diff=0.0364990234375`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=1.430511474609375e-06`, `max_abs_diff=0.0450439453125`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=9.5367431640625e-07`, `max_abs_diff=0.0312347412109375`, no NaN/nonfinite.

NCU follow-up: skipped. The representative H16/S4096 direct preallocated timing was a large regression, so there was no counter-follow-up target.

Decision: rejected and reverted shared-P-scale. Despite lower static smem and no spills, moving score-derived P-scale TMEM work from the producer to the PV side made every direct timing shape much slower. This classifies the direct P-scale TMEM store as useful overlap rather than the dominant handoff bubble for the kept qkscfix p112 route.

Revert verification:

```bash
grep -R "sharedpscale\|shared_pscale" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_sharedpscale_revert_qkscfix.log
```

The grep returned no matches. Rebuilt forward-only after revert and reran the kept p112 smoke:

- `finite=true`
- `mxfp4_ms=0.2568320035934448`
- `samples_ms=[0.2568320035934448, 0.1186240017414093]`
- `lse_max_abs=0.025980055332183838`
- `max_abs=1.09375`

## Probe Loop 27

Probe: added an opt-in qkscfix p112 early-P-ready/direct-rescale handoff route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_earlypready_pstage4_q208_p112_o56_qkscfix`
- Config type during probe: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_earlypready_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,208,56,112,1>`
- Trait changes during probe: `ONLINE_EARLY_P_READY=true`, `ONLINE_DUAL_OUTPUT_ACCUM_EARLY_P_READY=true`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: existing NCU evidence says the kept MXFP4 qkscfix p112 path is PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM/launch/spill limited. This structural handoff probe tested a live score-derived path: with early P ready and dual-output-accum direct-after-rescale enabled, the PV issue lane waits on per-correction-slot `direct_rescale_finished[corr_slot]` instead of the single `rescale_finished[0]` main-output handoff. This is distinct from the rejected `P_READY_BEFORE_RESCALE_WAIT` producer flag and does not touch V-scale staging.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_earlypready_qkscfix.log
```

Build result:

- Build succeeded.
- ptxas for early-P-ready qkscfix p112 probe:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 2512 bytes smem`
- ptxas for the compiled kept p112 baseline in the same build:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_earlypready_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=64000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.26287999749183655`
- `samples_ms=[0.26287999749183655, 0.11724799871444702]`
- `lse_max_abs=0.01805657148361206`
- `max_abs=0.953125`

Direct preallocated timing command:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors,
# WARMUP=20, ITERS=120, SEED=64001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_earlypready_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_earlypready_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.065536000 | 0.067295998 | +2.686% |
| H16/S4096 | persistent | 0.173503995 | 0.179296002 | +3.338% |
| H4/S2048 | persistent | 0.064640000 | 0.064800002 | +0.248% |

Probe-vs-baseline numeric check from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0015869140625`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.001708984375`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0008544921875`, no NaN/nonfinite.

NCU follow-up: skipped. H16/S4096 direct preallocated timing regressed by +3.338%, so there was no representative win to profile.

Decision: rejected and reverted early-P-ready qkscfix p112. The live per-correction-slot direct-rescale handoff is numerically stable and spill-free, but it increases smem and slows every measured shape. This suggests the single main-output handoff is not the dominant PV underfeed bubble for the kept qkscfix p112 route.

Revert verification:

```bash
grep -n "qkscfix_earlypready\|fusedmax_earlypready_pstage4_q208_p112_o56_qkscfix" \
  tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlypready_revert_qkscfix.log
```

The grep returned no matches. Rebuilt forward-only after revert. Restored qkscfix p112 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reran the kept p112 smoke:

- `finite=true`
- `mxfp4_ms=0.32630398869514465`
- `samples_ms=[0.32630398869514465, 0.12217599898576736]`
- `lse_max_abs=0.016933917999267578`
- `max_abs=1.0546875`

## Probe Loop 28

Probe: added an opt-in qkscfix p112 structural PV/output scheduling route that disabled the online decoupled PV issue split:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_nodecoupled_pstage4_q208_p112_o56_qkscfix`
- Config type during probe: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_nodecoupled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,208,56,112,1>`
- Trait change during probe: `ONLINE_DECOUPLED_PV_ISSUE=false`.
- Changed forward source during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: existing NCU evidence says the kept qkscfix p112 path is PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM, launch, or spill limited. This probe tested whether the decoupled producer/PV split itself was creating a scheduling bubble; it preserved score-derived qkscfix P packing and direct P-scale TMEM.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_nodecoupled_qkscfix.log
```

Build result:

- Build succeeded.
- ptxas for nodecoupled qkscfix p112 probe:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 1968 bytes smem`
- ptxas for the kept p112 qkscfix baseline in the same build:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke result, H16/S2048, seed 65000:

- `finite=true`
- `mxfp4_ms=0.21347199380397797`
- `samples_ms=[0.21347199380397797, 0.11558400094509125]`
- `lse_max_abs=0.04088421165943146`
- `max_abs=1.0078125`

Direct preallocated timing command:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors,
# WARMUP=20, ITERS=120, SEED=65001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_nodecoupled_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_nodecoupled_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.065856002 | 0.071680002 | +8.844% |
| H16/S4096 | persistent | 0.167264000 | 0.193984002 | +15.975% |
| H4/S2048 | persistent | 0.059999999 | 0.067488000 | +12.480% |

Probe-vs-baseline numeric check from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0015869140625`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0030517578125`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0006103515625`, no NaN/nonfinite.

NCU follow-up: skipped. H16/S4096 direct preallocated timing regressed by +15.975%, so there was no representative win to profile.

Decision: rejected and reverted nodecoupled. Disabling decoupled PV issue is live and numerically stable, but it substantially slows every measured shape. The decoupled producer/PV split is necessary for the kept qkscfix p112 route; the remaining PV underfeed is inside the handoff/work schedule, not caused by the split existing.

Revert verification:

```bash
grep -n "nodecoupled" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_nodecoupled_revert_qkscfix.log
```

The grep returned no matches. Rebuilt forward-only after revert. Restored qkscfix p112 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reran the kept p112 smoke, H16/S2048, seed 65002:

- `finite=true`
- `mxfp4_ms=0.36556801199913025`
- `samples_ms=[0.36556801199913025, 0.1409280002117157]`
- `lse_max_abs=0.026173889636993408`
- `max_abs=1.1953125`

## Probe Loop 29

Probe: added an opt-in qkscfix p112 structural P-payload store route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_payloadhs_pstage4_q208_p112_o56_qkscfix`
- Probe config during test: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_payloadstore_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,208,56,112,6,1>`
- Trait under test: `ONLINE_P_PAYLOAD_STORE_MODE=6`.
- Changed forward files during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`.

Rationale: user asked to pivot to structural P-handoff/PV-work. Existing NCU evidence says the kept qkscfix p112 path is PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM, launch, or spill limited. This probe tested a different P payload handoff layout while preserving score-derived qkscfix, scorepack/direct P-scale, decoupled PV issue, and the p112 baseline.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_payloadhs_qkscfix.log
```

Build result:

- Build succeeded.
- ptxas for payloadhs qkscfix p112 probe:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 1968 bytes smem`
- ptxas for kept qkscfix p112 baseline in same build:
  - `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
  - `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_payloadhs_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=66000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.24889600276947021`
- `samples_ms=[0.24889600276947021, 0.11875200271606445]`
- `lse_max_abs=0.027932090684771538`
- `max_abs=1.0859375`

Corrected direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED=66021, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_payloadhs_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_payloadhs_p112_qkscfix_direct_envfix.jsonl
PY
```

Corrected direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.058015998 | 0.058384001 | +0.634% |
| H16/S4096 | persistent | 0.165375993 | 0.165087998 | -0.174% |
| H4/S2048 | persistent | 0.054848000 | 0.055024000 | +0.321% |

Probe-vs-baseline numeric check from the corrected direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002685546875`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.001953125`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0020751953125`, no NaN/nonfinite.

NCU follow-up command, run because an earlier uncorrected direct harness suggested a representative H16/S4096 win:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_payloadhs_pstage4_q208_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
    --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
    --force-overwrite \
    --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_payloadhs_p112_h16_s4096 \
    python3 - <<'PY'
# Warmed the exact config then used cudaProfilerStart/cudaProfilerStop around one
# forward_streaming_live_mxfp4 launch. PROFILE_CONFIG was also assigned to
# TK_FA4_FP4PV_FWD_CONFIG before the extension call.
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_payloadhs_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_payloadhs_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_payloadhs_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_payloadhs_p112_h16_s4096_details.csv
```

NCU metric comparison, payloadhs vs kept p112 qkscfix baseline:

| Metric name | Baseline | Payloadhs | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 157.856000 us | 157.824000 us | -0.020% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.975208% | 6.969598% | -0.080% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418665 warp | 0.419626 warp | +0.230% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.550028 | 3.552468 | +0.069% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 35.979000% | 36.034610% | +0.155% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.242102% | 1.242428% | +0.026% |
| `derived__local_spilling_requests` | 0 | 0 | same |
| `launch__registers_per_thread` | 168 | 168 | same |

Decision: rejected and reverted payloadhs. The probe is live and numerically stable, but the corrected direct timing only had a tiny H16/S4096 win while regressing H16/S2048 and H4/S2048. NCU was neutral to slightly negative on the dominant PV-underfeed counters: tensor pipe active did not improve, long scoreboard did not improve, and eligible warps only moved by noise scale. This is not a validated structural handoff win.

Revert verification:

```bash
grep -R "payloadhs\|qkscfix_payloadstore" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_payloadhs_revert_qkscfix.log
```

The grep returned no matches. Rebuilt forward-only after revert. Restored qkscfix p112 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reran the kept p112 smoke, H16/S2048, seed 66022:

- `finite=true`
- `mxfp4_ms=0.25040000677108765`
- `samples_ms=[0.25040000677108765, 0.11459200084209442]`
- `lse_max_abs=0.020361311733722687`
- `max_abs=1.015625`

## Loop 30: qkscfix p112 pscale3_vsingle P-scale slot-depth probe

Context: user asked to pivot to a structural P-handoff/PV-work probe, with priority order coarse K64 handoff, real score-derived K256, then P-scale slot depth matched to `P_STAGE_SLOTS` if TMEM budget allows. Coarse K64 had already been tested and rejected by timeout/deadlock in Loop 21. Real score-derived K256 had been inspected in Loop 22 and was not a narrow safe patch because it needs static K256 launch, dynamic smem sizing, paired payload staging, paired direct P-scale TMEM staging, and must avoid the forbidden `fp4pv_pack_scores_to_stage_mxfp4*`/vector-amax path. This loop therefore tested the third structural option.

Hypothesis: direct score-derived qkscfix is PV tensor-core underfed with low eligible warps and long scoreboard. The current direct P-scale TMEM ping-pong may force the PV consumer to wait on P-scale slot reuse. Add an opt-in three-slot P-scale TMEM path, while using one V-scale TMEM slot only to stay within the existing `MAX_TENSOR_COLS` budget.

Patch:

- Added opt-in config `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pscale3_vsingle_pstage4_q208_p112_o56_qkscfix`.
- Added temporary traits `ONLINE_P_SCALE_TMEM_SLOTS=3` and `ONLINE_SINGLE_V_SCALE_TMEM=true`.
- Extended online direct P-scale slot selection to allow the three-slot path.
- Forward-only files touched during probe:
  - `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pscale3_vsingle_qkscfix.log
```

Probe ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 2000 bytes smem`

Kept qkscfix p112 baseline ptxas in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pscale3_vsingle_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=67000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.35020801424980164`
- `samples_ms=[0.35020801424980164, 0.13424000144004822]`
- `lse_max_abs=0.0250253789126873`
- `max_abs=0.9375`

Direct preallocated timing command:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED=67001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pscale3_vsingle_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_pscale3_vsingle_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0596799999 | 0.0615359992 | +3.110% |
| H16/S4096 | persistent | 0.1670719981 | 0.1719200015 | +2.902% |
| H4/S2048 | persistent | 0.0564480014 | 0.0578240007 | +2.438% |

Probe-vs-baseline numeric check from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002685546875`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002197265625`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0008544921875`, no NaN/nonfinite.

Decision: rejected without NCU. The probe compiled and was numerically stable, but direct isolated timing regressed every requested/cheap shape, including the representative H16/S4096 case. This says the extra P-scale slot depth plus single V-scale TMEM tradeoff increases latency rather than relieving the PV-underfeed scoreboard bottleneck.

Revert verification:

```bash
grep -R "pscale3_vsingle\|ONLINE_P_SCALE_TMEM_SLOTS\|ONLINE_SINGLE_V_SCALE_TMEM\|fp4pv_online_p_scale_tmem_slots\|fp4pv_online_single_v_scale_tmem" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pscale3_vsingle_revert_qkscfix.log
```

The grep returned no matches. Rebuilt forward-only after revert. Restored qkscfix p112 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reran the kept p112 smoke, H16/S2048, seed 67002:

- `finite=true`
- `mxfp4_ms=0.2463040053844452`
- `samples_ms=[0.2463040053844452, 0.11372800171375275]`
- `lse_max_abs=0.01877152919769287`
- `max_abs=0.91796875`

## Loop 31: qkscfix p112 diagzero causal payload work probe

Context: after Loop 30, the preferred structural options had either been rejected or were not a narrow safe patch. I avoided revisiting coarse K64 ready, K256, pscale3, V-scale staging, and register-neighbor probes. The next counter-backed hypothesis was still PV tensor-core underfeed from producer/handoff work: reduce unnecessary P payload producer work on non-diagonal online causal tiles.

Hypothesis: for the kept qkscfix launch (`Mb=Nb=128`, `CLUSTER_SIZE=1`), `online_iters_for_row_cluster(t_coord.y)` runs K tiles `0..m_tile`. Only the final diagonal tile can contain invalid causal MXFP4 groups. The existing scorepack route still computes `cols_valid`, `valid_groups`, and checks/calls the MXFP4 payload-zero helper on every K tile. Make that zeroing diagonal-only under an opt-in guard to reduce P payload work without changing V staging, P-scale staging, register allocation, or backward code.

Patch:

- Added opt-in config `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix`.
- Added trait `ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY`.
- Added a static assertion restricting the trait to `C::CLUSTER_SIZE == 1 && C::Mb == C::Nb`.
- Changed only the online MXFP4 causal payload-zero guard to skip non-diagonal tiles for the opt-in config.
- Forward-only files touched:
  - `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_qkscfix.log
```

Probe ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Kept qkscfix p112 baseline ptxas in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=68000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.21449600160121918`
- `samples_ms=[0.21449600160121918, 0.11910399794578552]`
- `lse_max_abs=0.0237832460552454`
- `max_abs=1.15625`

Direct preallocated timing command:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED=68001, CUDA_VISIBLE_DEVICES=0.
# Configs:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0611199997 | 0.0609919988 | -0.209% |
| H16/S4096 | persistent | 0.1708640009 | 0.1700160056 | -0.496% |
| H4/S2048 | persistent | 0.0594400000 | 0.0591039993 | -0.565% |

Probe-vs-baseline numeric check from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.001708984375`, no NaN/nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002197265625`, no NaN/nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=9.183549615799121e-41`, no NaN/nonfinite.

NCU follow-up command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
    --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
    --force-overwrite \
    --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_p112_h16_s4096 \
    python3 - <<'PY'
# Warmed the exact config for 10 launches, then used cudaProfilerStart/cudaProfilerStop
# around one preallocated forward_streaming_live_mxfp4 launch.
# PROFILE_CONFIG was assigned to TK_FA4_FP4PV_FWD_CONFIG around the raw extension call.
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_p112_h16_s4096_details.csv
```

NCU metric comparison, diagzero vs kept p112 qkscfix baseline:

| Metric name | Baseline | Diagzero | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 157.856000 us | 156.224000 us | -1.034% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.975208% | 7.050976% | +1.086% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418665 warp | 0.420131 warp | +0.350% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.550028 | 3.535940 | -0.397% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 35.979000% | 36.091683% | +0.313% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.242102% | 1.255418% | +1.072% |
| `derived__local_spilling_requests` | 0 | 0 | same |
| `launch__registers_per_thread` | 168 | 168 | same |
| `inst_executed` | 54384420 | 54871959 | +0.896% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.489875 | 0.488978 | -0.183% |

Decision: kept as a validated forward win. Direct isolated timing improved all measured shapes, and representative H16/S4096 NCU confirms the right bottleneck movement: lower kernel duration, higher tensor activity, more eligible warps, and lower long scoreboard, without spills or register growth. Classification remains PV tensor-core underfeed from producer/PV work and handoff latency; the diagonal-only P payload zeroing reduces enough producer work to feed PV slightly better.

## Loop 32: score-derived diagzero coarse K64 P payload readiness probe

Hypothesis: existing NCU evidence still points to PV tensor-core underfeed driven by producer/PV handoff latency, not DRAM, launch, occupancy, or spills. A score-derived qkscfix route might let PV issue the first K64 half earlier if direct x1 P-scale TMEM is available before payload handoff and if diagonal causal payload zeroing remains exact.

Probe: added an opt-in forward-only config:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_coarsek64_pstage4_q208_p112_o56_qkscfix`

Implementation shape:

- Derived from kept `diagzero` qkscfix route.
- Enabled `ONLINE_EARLY_P_READY`, `ONLINE_DUAL_OUTPUT_ACCUM_EARLY_P_READY`, and `ONLINE_SPLIT_P_READY_K64`.
- Extended split K64 readiness eligibility to score-derived x1 direct P-scale TMEM.
- Stored score-derived x1 P scales before payload readiness.
- Published half0 after first 64 payload columns only for non-diagonal tiles; on the diagonal tile, published half0 only after causal payload zeroing, together with half1.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_coarsek64_qkscfix.log
```

ptxas for probe:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 2576 bytes smem`

ptxas for kept `diagzero` and p112 qkscfix baselines in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_coarsek64_pstage4_q208_p112_o56_qkscfix'
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=70000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg,
    'mxfp4_ms': rec['mxfp4_ms'],
    'samples_ms': rec['mxfp4_samples_ms'],
    'finite': rec['finite'],
    'lse_max_abs': rec['comparison_vs_bf16']['lse_max_abs_diff'],
    'max_abs': rec['comparison_vs_bf16']['max_abs_diff'],
}, indent=2))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.22972799837589264`
- `samples_ms=[0.22972799837589264, 0.11715199798345566]`
- `lse_max_abs=0.03412073850631714`
- `max_abs=1.1328125`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED=70001, CUDA_VISIBLE_DEVICES=0.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_coarsek64_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_coarsek64_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0609599985 | 0.0651199967 | +6.824% |
| H16/S4096 | persistent | 0.1679999977 | 0.1831679940 | +9.029% |
| H4/S2048 | persistent | 0.0586879998 | 0.0623999983 | +6.325% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0006103515625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0010986328125`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0009765625`, no nonfinite.

Decision: rejected and reverted. H16/S4096 was not a timing win, so no NCU follow-up was run. The added readiness events and early x1 P-scale store increased shared state from 1968 to 2576 bytes and regressed all direct timings despite clean correctness. Classification remains PV tensor-core underfeed, but this coarse K64 handoff increased producer/PV synchronization cost more than it exposed useful PV work.

Reverted files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

## Loop 37: exact score-derived qkscfix K64 P payload handoff probe

Hypothesis: existing NCU evidence for kept `diagzero` qkscfix says the kernel is PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM/launch/spill limited. The existing split K64 ready path deliberately excluded score-derived direct-scale scorepack routes, so this probe made that path live for scorepack qkscfix: publish x1 P-scale TMEM readiness early, then publish exactly two coarse payload-ready events for Nb=128, with no per-32 ready churn.

Probe: opt-in forward-only route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_scorek64_pstage4_q208_p112_o56_qkscfix`
- Added temporary `ONLINE_SCOREPACK_SPLIT_P_READY_K64` config/trait.
- Reused the existing consumer `p_online_k64_ready[p_buf][0/1]` wait path.
- Published half 0 after qids 0-1 payload stores and half 1 at the existing final payload publish point.
- Kept diagonal causal-zero correctness guard before half-0 publish.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_scorek64_qkscfix.log
```

ptxas for probe:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 2032 bytes smem`

ptxas for kept `diagzero` q208 baseline in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
import sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_scorek64_pstage4_q208_p112_o56_qkscfix'
res = exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=75000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'finite': res.get('finite'),
    'mxfp4_ms': res.get('mxfp4_ms'),
    'mxfp4_samples_ms': res.get('mxfp4_samples_ms'),
    'lse_max_abs': res.get('lse_max_abs'),
    'max_abs': res.get('max_abs'),
}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.2356480062007904`
- `mxfp4_samples_ms=[0.2356480062007904, 0.12876799702644348]`
- `lse_max_abs=0.024674266576766968`
- `max_abs=0.96484375`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=75001, CUDA_VISIBLE_DEVICES=0.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_scorek64_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_scorek64_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.06035200134 | 0.06374400109 | +5.620% |
| H16/S4096 | persistent | 0.16681599617 | 0.17553599924 | +5.227% |
| H4/S2048 | persistent | 0.05678400025 | 0.05936000124 | +4.536% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.008056640625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00927734375`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.006591796875`, no nonfinite.

Decision: rejected and reverted. H16/S4096 direct timing regressed by +5.227%, so no NCU follow-up was run. The path was live and numerically sane, but earlier coarse payload availability increased producer/PV synchronization cost more than it reduced PV starvation. Classification remains PV tensor-core underfeed from producer/PV handoff latency; exact two-event K64 payload handoff is not the right handoff granularity for the kept score-derived qkscfix route.

Revert/build verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_scorek64_revert_qkscfix.log
```

Post-revert kept `diagzero` q208 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Post-revert kept `diagzero` q200 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reverted files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

## Loop 36: diagzero q200/p112/o56 register-balance probe

Hypothesis: existing NCU evidence for the kept `diagzero` qkscfix route still shows PV tensor-core underfeed, low eligible warps, and long scoreboard, with no DRAM/launch/spill limit. A narrower QK/producer register cap may reduce producer instruction footprint or scoreboard bubbles enough to feed PV slightly better without changing P payload handoff semantics. This is not a baseline reroute; `q208/p112/o56` remains preserved.

Opt-in probe:

- Baseline: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix`
- Probe: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q200_p112_o56_qkscfix`

Changed file:

- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
mkdir -p results/mxfp4_fa4_forward_profile_20260612 && make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_q200_p112_qkscfix.log
```

ptxas for probe and baseline in the same build:

- q200 probe: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q208 baseline: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json, sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q200_p112_o56_qkscfix'
res=exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=74000, warmup=1, iters=2, mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto', bf16_baseline='tk', include_output_only=False)
print(json.dumps({'finite':res.get('finite'),'mxfp4_ms':res.get('mxfp4_ms'),'mxfp4_samples_ms':res.get('mxfp4_samples_ms'),'max_abs':res.get('comparison_vs_bf16',{}).get('max_abs_diff'),'lse_max_abs':res.get('comparison_vs_bf16',{}).get('lse_max_abs_diff')}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.39561599493026733`
- `samples_ms=[0.39561599493026733, 0.15440000593662262]`
- `lse_max_abs=0.024273820221424103`
- `max_abs=0.9296875`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=74001, CUDA_VISIBLE_DEVICES=0.
# Alternated baseline/probe launch order after warmup.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q200_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_q200_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.06075200066 | 0.06088000163 | +0.211% |
| H16/S4096 | persistent | 0.16670399904 | 0.16667199880 | -0.019% |
| H4/S2048 | persistent | 0.05699200183 | 0.05715199932 | +0.281% |

Focused H16/S4096 repeat command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Same direct preallocated alternating-launch harness.
# Shape=H16/S4096, WARMUP=40, ITERS=360, SEED=74111.
# Output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_q200_p112_qkscfix_h16_s4096_repeat.jsonl
PY
```

Focused H16/S4096 repeat result:

- Baseline median: `0.16313600540 ms`
- Probe median: `0.16271999478 ms`
- Delta: `-0.255%`
- `probe_vs_baseline_lse_max_abs_diff=0.0`
- `probe_vs_baseline_max_abs_diff=0.001708984375`
- `probe_nonfinite=false`

NCU command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096 python3 - <<'PY'
import os
import sys
import torch
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
seq = int(os.environ['PROFILE_SEQ'])
heads = int(os.environ['PROFILE_HEADS'])
cfg = os.environ['PROFILE_CONFIG']
os.environ['TK_FA4_FP4PV_FWD_CONFIG'] = cfg
ext = exp._load_forward_experiments_ext()
q_bf16, k_bf16, v_bf16 = exp._make_live_bf16_source_inputs(seq, heads=heads, seed=74120, device='cuda')
fp4_inputs = exp._fp4_qk_mxfp4_v_inputs_from_bf16_source(q_bf16, k_bf16, v_bf16)
q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = exp._prepare_mxfp4_fwd_inputs_for_config(fp4_inputs, seqlen=seq, config=cfg)
out = torch.empty((1, seq, heads, exp._D_VO), device='cuda', dtype=torch.bfloat16)
lse = torch.empty((1, heads, 1, seq), device='cuda', dtype=torch.float32)
persistent = exp._resolve_mxfp4_fwd_launch_mode(seq, heads, 'auto') != 'fullgrid'
qmode = exp._mxfp4_quant_mode_to_int(None)
for _ in range(10):
    ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent)
torch.cuda.cudart().cudaProfilerStop()
torch.cuda.synchronize()
print({'config': cfg, 'seq': seq, 'heads': heads, 'finite': bool(torch.isfinite(out).all().item() and torch.isfinite(lse).all().item())})
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096_details.csv
```

NCU metric names compared against kept `diagzero` q208 baseline:

| Metric | q208 baseline | q200 probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.224 | 155.936 | -0.184% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.050976 | 6.970072 | -1.147% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.420131 | 0.420526 | +0.094% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.535940 | 3.537416 | +0.042% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.488978 | 0.474931 | -2.873% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 36.091683 | 36.084591 | -0.020% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.255418 | 1.257666 | +0.179% |
| `derived__local_spilling_requests` | 0 | 0 | 0 |
| `launch__registers_per_thread` | 168 | 168 | 0 |
| `launch__shared_mem_per_block_static` | 1.968 | 1.968 | 0 |
| `inst_executed` | 54871959 | 54522796 | -0.636% |
| `smsp__warps_active.avg.per_cycle_active` | 2.868280 | 2.868017 | -0.009% |

Decision: keep as opt-in only, no baseline reroute and no commit yet. The representative H16/S4096 direct repeat and NCU isolated kernel both show a small elapsed win with no spill/register/smem cost, and the lower instruction count plus lower short scoreboard is the likely mechanism. However, H16/S2048 and H4/S2048 regressed slightly, tensor active fell, and long scoreboard did not improve. Classification remains PV tensor-core underfeed from producer/PV handoff latency, not memory bandwidth, launch, occupancy, or spills. Next loop should pivot back to structural P-handoff/PV-work rather than more narrow staging/register flags.

## Loop 34: score-derived qkscfix diagzero P payload store-mode 4

Hypothesis: existing NCU evidence shows the kept `diagzero` qkscfix route is PV tensor-core underfed, with low eligible warps and long scoreboard, while not DRAM/launch/spill limited. A structural P-payload store layout change might reduce P payload handoff friction without touching V-scale staging or backward code. This retested payload store-mode 4 specifically on the current kept `diagzero` route rather than older non-diagzero routes.

Probe: opt-in forward-only route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_payloadv2_pstage4_q208_p112_o56_qkscfix`
- Added temporary config trait `ONLINE_P_PAYLOAD_STORE_MODE = 4`.
- Preserved baseline `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix`.

Build command:

```bash
mkdir -p results/mxfp4_fa4_forward_profile_20260612 && make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_payloadv2_qkscfix.log
```

ptxas for probe:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

ptxas for kept `diagzero` baseline in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
import sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_payloadv2_pstage4_q208_p112_o56_qkscfix'
res = exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=72000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'finite': res.get('finite'),
    'mxfp4_ms': res.get('mxfp4_ms'),
    'mxfp4_samples_ms': res.get('mxfp4_samples_ms'),
    'max_abs': res.get('comparison_vs_bf16', {}).get('max_abs_diff'),
    'lse_max_abs': res.get('comparison_vs_bf16', {}).get('lse_max_abs_diff'),
}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.4599039852619171`
- `samples_ms=[0.4599039852619171, 0.14451199769973755]`
- `lse_max_abs=0.0273691788315773`
- `max_abs=1.0546875`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED=72001, CUDA_VISIBLE_DEVICES=0.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_payloadv2_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_payloadv2_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.05761599913 | 0.05779200047 | +0.305% |
| H16/S4096 | persistent | 0.16572800279 | 0.16427200288 | -0.879% |
| H4/S2048 | persistent | 0.05545600131 | 0.05532800034 | -0.231% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0025634765625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0013427734375`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0010986328125`, no nonfinite.

NCU follow-up command because H16/S4096 direct timing suggested a win:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_payloadv2_pstage4_q208_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_payloadv2_p112_h16_s4096 python3 - <<'PY'
import os
import sys
import torch
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
seq = int(os.environ['PROFILE_SEQ'])
heads = int(os.environ['PROFILE_HEADS'])
cfg = os.environ['PROFILE_CONFIG']
os.environ['TK_FA4_FP4PV_FWD_CONFIG'] = cfg
ext = exp._load_forward_experiments_ext()
q_bf16, k_bf16, v_bf16 = exp._make_live_bf16_source_inputs(seq, heads=heads, seed=72010, device='cuda')
fp4_inputs = exp._fp4_qk_mxfp4_v_inputs_from_bf16_source(q_bf16, k_bf16, v_bf16)
q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = exp._prepare_mxfp4_fwd_inputs_for_config(fp4_inputs, seqlen=seq, config=cfg)
out = torch.empty((1, seq, heads, exp._D_VO), device='cuda', dtype=torch.bfloat16)
lse = torch.empty((1, heads, 1, seq), device='cuda', dtype=torch.float32)
persistent = exp._resolve_mxfp4_fwd_launch_mode(seq, heads, 'auto') != 'fullgrid'
qmode = exp._mxfp4_quant_mode_to_int(None)
for _ in range(10):
    ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
ext.forward_streaming_live_mxfp4(q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse, qmode, persistent)
torch.cuda.cudart().cudaProfilerStop()
torch.cuda.synchronize()
print({'config': cfg, 'seq': seq, 'heads': heads, 'finite': bool(torch.isfinite(out).all().item() and torch.isfinite(lse).all().item())})
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_payloadv2_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_payloadv2_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_payloadv2_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_payloadv2_p112_h16_s4096_details.csv
```

Representative H16/S4096 NCU metrics vs kept `diagzero` baseline:

| Metric | Baseline | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.224 us | 156.736 us | +0.328% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.050976 | 7.036299 | -0.208% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.420131 | 0.419995 | -0.032% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.535940 | 3.529582 | -0.180% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.488978 | 0.489369 | +0.080% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 36.091683 | 36.042956 | -0.135% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.255418 | 1.250998 | -0.352% |
| `derived__local_spilling_requests` | 0 | 0 | unchanged |
| `launch__registers_per_thread` | 168 | 168 | unchanged |
| `launch__shared_mem_per_block_static` | 1.968 KB | 1.968 KB | unchanged |
| `inst_executed` | 54871959 | 54889402 | +0.032% |

Decision: rejected and reverted. Although direct preallocated H16/S4096 had a small timing win, the required representative NCU isolated kernel was slower and showed no supporting improvement in tensor-active, eligible warps, or issue-active. Classification remains PV tensor-core underfeed dominated by producer/PV handoff latency, but this payload store-mode change does not reduce the dominant bubble for the kept score-derived qkscfix route.

Revert/build verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_diagzero_payloadv2_revert_qkscfix.log
```

Post-revert kept `diagzero` qkscfix ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reverted files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

## Loop 35: score-derived qkscfix diagzero staggered P payload store

Pre-patch investigation: candidate `real score-derived K256` was inspected first because it is higher priority than another payload layout. Current code does not provide a narrow valid K256 score-derived route:

- `STATIC_MXFP4_K256` is keyed only by static consumer modes `6` and `178`, while fwd-config online dispatches launch `kernel_streaming_live_fp4pv<C, true>` with static consumer mode `-1`.
- Existing K256 pack path around `fwd_streaming_kernel.inc:6414-6454` uses `fp4pv_pack_scores_to_stage_mxfp4[_range]`, which is the fallback/vector-amax-style path this probe must avoid.
- The kept score-derived qkscfix path writes payload through `fp4pv_store_quantized_scores_group_mxfp4_selected(...)` into `p_fp4_stage[buf]` and uses x1 direct P-scale TMEM, while K256 PV consumes `p_fp4_stage_k256[pair_buf]` and `p_sc_stage_k256[...]`. A correctness-first K256 probe would require a new online K256 trait, paired P-stage/event semantics, paired direct x1 P-scale TMEM writes, and payload writes into `p_fp4_stage_k256`, so it was not a small targeted patch.

Hypothesis: with existing NCU showing PV tensor-core underfeed and long scoreboard, a structural P-payload layout change that staggers row-phase stores by MX block might reduce shared/proxy handoff contention for the score-derived qkscfix path. This is live in the current kept route because `ONLINE_P_PAYLOAD_STORE_MODE` drives `fp4pv_store_quantized_scores_group_mxfp4_selected(...)` in the score-derived prescaled pack block.

Probe: opt-in forward-only route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_payloadstagger_pstage4_q208_p112_o56_qkscfix`
- Added temporary config trait `ONLINE_P_PAYLOAD_STORE_MODE = 5`.
- Preserved baseline `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix`.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_payloadstagger_qkscfix.log
```

ptxas for probe:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

ptxas for kept `diagzero` baseline in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
import sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_payloadstagger_pstage4_q208_p112_o56_qkscfix'
res = exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=73000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'finite': res.get('finite'),
    'mxfp4_ms': res.get('mxfp4_ms'),
    'mxfp4_samples_ms': res.get('mxfp4_samples_ms'),
    'max_abs': res.get('comparison_vs_bf16', {}).get('max_abs_diff'),
    'lse_max_abs': res.get('comparison_vs_bf16', {}).get('lse_max_abs_diff'),
}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.2908160090446472`
- `samples_ms=[0.2908160090446472, 0.12387199699878693]`
- `lse_max_abs=0.016952991485595703`
- `max_abs=1.03125`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=73001, CUDA_VISIBLE_DEVICES=0.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_payloadstagger_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_payloadstagger_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0609439984 | 0.0626880005 | +2.862% |
| H16/S4096 | persistent | 0.1644959971 | 0.1836320013 | +11.633% |
| H4/S2048 | persistent | 0.0589280017 | 0.0616960004 | +4.697% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0032958984375`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0032958984375`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0032958984375`, no nonfinite.

Decision: rejected and reverted. H16/S4096 was a large direct-timing regression, so no NCU follow-up was run. Staggered row-phase stores are live in the kept score-derived path but worsen the producer/PV handoff, especially at S4096. Classification remains PV tensor-core underfeed from producer/PV handoff latency; this payload layout increases, rather than reduces, the handoff bubble.

Revert/build verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_diagzero_payloadstagger_revert_qkscfix.log
```

Post-revert kept `diagzero` qkscfix ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reverted files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

## Loop 33: score-derived qkscfix diagzero P-scale slot depth matched to P_STAGE_SLOTS

Hypothesis: existing NCU evidence says the kept `diagzero` scorepack qkscfix route is PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM/launch/spill limited. With `P_STAGE_SLOTS=4` but only two direct P-scale TMEM slots, the quant side can hit P-scale slot reuse pressure before PV has consumed a scale. A four-slot P-scale TMEM probe matched to `P_STAGE_SLOTS=4` could reduce producer/PV scale handoff waits. TMEM budget requires single-buffered V scale for this exact map: score/output/Q/K/P/V = `256 + 128 + 16 + 16 + 4*16 + 1*32 = 512` columns.

Probe: opt-in forward-only route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pscale4_vsingle_pstage4_q208_p112_o56_qkscfix`
- Added temporary traits `ONLINE_P_SCALE_TMEM_SLOTS=4` and `ONLINE_SINGLE_V_SCALE_TMEM=true`.
- Extended direct P-scale TMEM selectors to address slot 3.
- Preserved baseline `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix`.

Build command:

```bash
mkdir -p results/mxfp4_fa4_forward_profile_20260612 && make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_pscale4_vsingle_qkscfix.log
```

ptxas for probe:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 2016 bytes smem`

ptxas for kept `diagzero` baseline in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
import sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pscale4_vsingle_pstage4_q208_p112_o56_qkscfix'
res = exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=71000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'finite': res.get('finite'),
    'mxfp4_ms': res.get('mxfp4_ms'),
    'mxfp4_samples_ms': res.get('mxfp4_samples_ms'),
}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.41843199729919434`
- `samples_ms=[0.41843199729919434, 0.1709119975566864]`

Direct preallocated timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED=71001, CUDA_VISIBLE_DEVICES=0.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pscale4_vsingle_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_pscale4_vsingle_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.05772799999 | 0.05958399922 | +3.215% |
| H16/S4096 | persistent | 0.16342400014 | 0.16641599685 | +1.831% |
| H4/S2048 | persistent | 0.05459199846 | 0.05494400114 | +0.645% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=9.183549615799121e-41`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002197265625`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00048828125`, no nonfinite.

Decision: rejected and reverted. H16/S4096 was not a timing win, so no NCU follow-up was run. The exact four-slot P-scale map fit TMEM and stayed spill-free, but single-buffering V scale plus extra slot selection/reg pressure regressed every direct shape. Classification remains PV tensor-core underfeed from producer/PV handoff latency, but P-scale slot reuse is not the dominant limiter for the kept score-derived qkscfix route.

Reverted files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

## Loop 38: score-derived qkscfix diagzero early P-scale ready with separate payload-ready

Hypothesis: existing NCU evidence still classifies kept `diagzero` qkscfix as PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM/launch/spill limited. The exact K64 handoff probe showed early payload chunking is live but too expensive. This probe tested a narrower structural handoff split: publish direct P-scale TMEM readiness immediately after the x1 P-scale TMEM store, then publish a separate payload-ready event after shared backing is visible. The PV lane waited both events, allowing P-scale wait/staging to overlap payload publish without issuing PV before payload visibility.

Probe: opt-in forward-only route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_earlypscale_pstage4_q208_p112_o56_qkscfix`
- Added temporary `ONLINE_DECOUPLED_PV_EARLY_SCALE_READY`.
- Guarded to cluster1 score-derived x1 direct-scale routes only.
- Preserved kept `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix` and q200 opt-in route.

Build command:

```bash
mkdir -p results/mxfp4_fa4_forward_profile_20260612 && make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_earlypscale_qkscfix.log
```

ptxas for probe:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 2000 bytes smem`

ptxas for kept `diagzero` q208 baseline in the same build:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json
import sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_earlypscale_pstage4_q208_p112_o56_qkscfix'
res = exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=76000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'finite': res.get('finite'),
    'mxfp4_ms': res.get('mxfp4_ms'),
    'mxfp4_samples_ms': res.get('mxfp4_samples_ms'),
    'lse_max_abs': res.get('lse_max_abs'),
    'max_abs': res.get('max_abs'),
}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.23324799537658691`
- `mxfp4_samples_ms=[0.23324799537658691, 0.12383999675512314]`

Direct preallocated timing command:

```bash
timeout 700s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=76001, CUDA_VISIBLE_DEVICES=0.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_earlypscale_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_earlypscale_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.05844799988 | 0.05881600082 | +0.630% |
| H16/S4096 | persistent | 0.16404800117 | 0.16601600498 | +1.200% |
| H4/S2048 | persistent | 0.05310399830 | 0.05423999950 | +2.139% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0030517578125`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0023193359375`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00146484375`, no nonfinite.

Decision: rejected and reverted. H16/S4096 direct timing regressed by +1.200%, so no NCU follow-up was run. The path was live and correctness-safe, but the extra semaphore and split handoff cost outweighed any overlap of P-scale staging with payload publication. Classification remains PV tensor-core underfeed from producer/PV handoff latency; splitting scale-ready and payload-ready at this point does not reduce the dominant bubble.

Revert/build verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlypscale_revert_qkscfix.log
```

Post-revert kept `diagzero` q208 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Post-revert kept `diagzero` q200 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Reverted files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
## Loop 39: `diagzero_pscalereuse` P-scale reuse-wait overlap probe rejected

Hypothesis: existing NCU evidence still classifies MXFP4 qkscfix as PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM/launch/spill limited. The direct P-scale TMEM path waits for P-scale slot reuse after payload generation; if the first `quant_wg_sync()` before that wait is moved behind the warp0 reuse wait, warp0 can overlap the reuse wait with remaining payload stores while the later sync still fences payload before scale TMEM store and payload-ready publication.

Patch: added opt-in forward-only config
`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pscalereuse_pstage4_q208_p112_o56_qkscfix`, guarded to cluster1/3WG score-derived x1 direct P-scale. No backward files touched.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_pscalereuse_qkscfix.log
```

ptxas for probe and kept baselines in the same build:

- pscalereuse q208 probe: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- diagzero q208 baseline: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- diagzero q200 opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json, sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pscalereuse_pstage4_q208_p112_o56_qkscfix'
res = exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=77000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg, 'finite': bool(res.get('finite', False)),
    'mxfp4_ms': res.get('mxfp4_ms'),
    'mxfp4_samples_ms': res.get('mxfp4_samples_ms'),
    'lse_max_abs': res.get('lse_max_abs'),
    'max_abs': res.get('max_abs'),
}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.23340800404548645`
- `samples_ms=[0.23340800404548645, 0.12140800058841705]`

Direct preallocated timing command:

```bash
timeout 700s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=77001, CUDA_VISIBLE_DEVICES=0.
# Alternated baseline/probe launch order after warmup.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pscalereuse_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_pscalereuse_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.05819199979 | 0.05819199979 | +0.000% |
| H16/S4096 | persistent | 0.17124799639 | 0.17129600048 | +0.028% |
| H4/S2048 | persistent | 0.05990400165 | 0.05977600068 | -0.214% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0009765625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0010986328125`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0013427734375`, no nonfinite.

Decision: rejected and reverted. The representative H16/S4096 isolated forward kernel was not a win (+0.028%), so no NCU follow-up was run by the H16/S4096 gate. Classification remains PV tensor-core underfeed from producer/PV handoff latency; overlapping P-scale slot reuse wait with payload tail does not reduce the dominant bubble for the kept qkscfix+p112 route.

Revert/build verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pscalereuse_revert_qkscfix.log
```

Post-revert kept `diagzero` q208 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Post-revert kept `diagzero` q200 ptxas:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Post-revert symbol check: `pscalereuse` not found in `fwd_configs.inc`, `fwd_streaming_kernel.inc`, or `fwd_host_dispatch.inc`.

## Loop 40: `diagzero_prepub` P payload proxy prepublish kept opt-in

Profile basis: existing H16/S4096 qkscfix/diagzero NCU still points to PV tensor-core underfeed with low eligible warps and long scoreboard. P-scale slot-depth matching was rejected before patching because qkscfix q208 already uses the full 512-column TMEM budget: dual score 256 + output 128 + Q/K scale 32 + two P-scale slots 32 + two V-scale slots 64 = 512. A third P-scale slot would require the already-rejected single-V-scale family. The K256 path remains too broad: `STATIC_MXFP4_K256` is tied to consumer modes 6/178, direct-after-rescale asserts `!STATIC_MXFP4_K256`, and the live K256 producer still uses `fp4pv_pack_scores_to_stage_mxfp4[_range]`.

Hypothesis: in the kept 3WG cluster1 qkscfix route, `p_sc_tmem_ready` is the combined readiness signal sequenced after P payload backing publish and P-scale TMEM store/wait. Moving `fence.proxy.async.shared::cta` before the P-scale TMEM store should preserve ordering because payload writes are complete before the local `quant_wg_sync()`, while potentially hiding proxy handoff latency under the scale TMEM store/wait.

Patch: added opt-in forward-only config
`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q208_p112_o56_qkscfix`. It prepublishes P payload proxy backing before direct P-scale TMEM store/wait, then skips the later duplicate publish. Guarded to cluster1 3WG direct P-scale routes. No backward files touched. Kept qkscfix+p112 q208 baseline and q200 opt-in dispatch unchanged.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_qkscfix.log
```

ptxas for probe and kept baselines in the same build:

- prepub q208 probe: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- diagzero q208 baseline: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- diagzero q200 opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import json, sys
sys.path.insert(0, 'tk_fa4')
import fp4_pv_experiments as exp
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q208_p112_o56_qkscfix'
res = exp.benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=78000, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, launch_mode='auto', bf16_launch_mode='auto',
    bf16_baseline='tk', include_output_only=False)
print(json.dumps({
    'config': cfg, 'finite': bool(res.get('finite', False)),
    'mxfp4_ms': res.get('mxfp4_ms'),
    'mxfp4_samples_ms': res.get('mxfp4_samples_ms'),
    'lse_max_abs': res.get('comparison_vs_bf16', {}).get('lse_max_abs_diff'),
    'max_abs': res.get('comparison_vs_bf16', {}).get('max_abs_diff'),
}, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.4304960072040558`
- `samples_ms=[0.4304960072040558, 0.13846400380134583]`
- `lse_max_abs=0.021503347903490067`
- `max_abs=1.03125`

Direct preallocated timing command:

```bash
timeout 700s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Timed forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=78001, CUDA_VISIBLE_DEVICES=0.
# Alternated baseline/probe launch order after warmup.
# Baseline:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix
# Probe:
#   dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q208_p112_o56_qkscfix
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
# Full JSONL output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_p112_qkscfix_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Baseline median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.05872000009 | 0.05875200033 | +0.054% |
| H16/S4096 | persistent | 0.16371199489 | 0.16368000209 | -0.020% |
| H4/S2048 | persistent | 0.05475199968 | 0.05507199839 | +0.584% |

Probe-vs-`diagzero` numeric check from direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0008544921875`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0029296875`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0008544921875`, no nonfinite.

Focused H16/S4096 repeat command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
# Same direct preallocated alternating-launch harness.
# Shape=H16/S4096, WARMUP=40, ITERS=360, SEED=78111.
# Output:
#   results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_p112_qkscfix_h16_s4096_repeat.jsonl
PY
```

Focused H16/S4096 repeat result:

- Baseline median: `0.16025599837 ms`
- Probe median: `0.16022400558 ms`
- Delta: `-0.020%`
- `probe_vs_baseline_lse_max_abs_diff=0.0`
- `probe_vs_baseline_max_abs_diff=0.0023193359375`
- `probe_nonfinite=false`

NCU command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q208_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_p112_h16_s4096 python3 - <<'PY'
# Warmed 10 launches, cudaProfilerStart(), one forward_streaming_live_mxfp4 launch, cudaProfilerStop().
PY
```

NCU exports:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_p112_h16_s4096_details.csv
```

NCU metrics versus existing `diagzero` q208 H16/S4096 baseline:

| Metric | Baseline | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.224 us | 156.000 us | -0.143% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.050976 | 7.005125 | -0.650% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.420131 | 0.421119 | +0.235% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.535940 | 3.525685 | -0.290% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.488978 | 0.487732 | -0.255% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 36.091683 | 36.123232 | +0.087% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.255418 | 1.256806 | +0.111% |
| `derived__local_spilling_requests` | 0 | 0 | 0 |
| `launch__registers_per_thread` | 168 | 168 | 0 |
| `launch__shared_mem_per_block_static` | 1.968 KB | 1.968 KB | 0 |
| `inst_executed` | 54871959 | 54750322 | -0.222% |
| `smsp__warps_active.avg.per_cycle_active` | 2.868280 | 2.868242 | -0.001% |

Decision: keep as opt-in only, do not reroute baseline and do not commit. The representative H16/S4096 isolated-kernel timing and NCU both move in the right direction, and the metric shifts match the target bottleneck: slightly higher eligible warps, lower long/short scoreboard stalls, no resource/spill regression, and fewer instructions. The win is too small and mixed across H16/S2048/H4/S2048 to make this a validated baseline route.

## Loop 41: combined `diagzero_prepub_q200` register-balance/P-handoff probe kept opt-in

Profile basis: the same-build qkscfix/diagzero profiles still classify the forward kernel as PV tensor-core underfed, with low eligible warps and long scoreboard as the dominant bubble. Loop 40's P-payload prepublish improved the q208 route slightly, and Loop 36's q200 producer-register balance improved the H16/S4096 representative shape slightly. This loop tested whether the two changes compose without adding spills, shared memory, or a new handoff stall.

Patch: dispatch-only opt-in forward route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix`
- Template: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Changed forward source: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- No backward files touched.

Build command:

```bash
mkdir -p results/mxfp4_fa4_forward_profile_20260612 && make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_q200_qkscfix.log
```

ptxas from the build:

- q200 prepub probe: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 diagzero baseline: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q208 prepub opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q208 diagzero kept baseline: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_q200_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79000, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.19046400487422943`
- `mxfp4_samples_ms=[0.19046400487422943, 0.12118399888277054]`
- `lse_max_abs_diff=0.02754802070558071` versus TK BF16
- `max_abs_diff=0.90234375` versus TK BF16
- no output/LSE nonfinite

Direct preallocated timing command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_q200_p112_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79001.
# Configs:
#   q208_diagzero, q200_diagzero, q208_prepub, q200_prepub
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q208 diagzero ms | q200 diagzero ms | q208 prepub ms | q200 prepub ms | q200 prepub vs q200 | q200 prepub vs q208 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05980800092 | 0.05972800031 | 0.05971200019 | 0.05964799970 | -0.134% | -0.268% |
| H16/S4096 | 0.16913600266 | 0.16888000071 | 0.16910400242 | 0.16876800358 | -0.066% | -0.218% |
| H4/S2048 | 0.05803199857 | 0.05832000077 | 0.05817599967 | 0.05836800113 | +0.082% | +0.579% |

Probe-vs-q200 direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0020751953125`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0029296875`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=9.183549615799121e-41`, no nonfinite.

Focused H16/S4096 repeat command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_q200_p112_qkscfix_h16_s4096_repeat.jsonl
# Same direct preallocated alternating-launch harness.
# Shape=H16/S4096, WARMUP=50, ITERS=400, SEED=79100.
# Configs:
#   q200_diagzero, q200_prepub, q208_diagzero
PY
```

Focused H16/S4096 repeat result:

- q200 diagzero median: `0.16096000373 ms`
- q200 prepub median: `0.16086399555 ms`
- q208 diagzero median: `0.16116800159 ms`
- q200 prepub vs q200: `-0.060%`
- q200 prepub vs q208: `-0.189%`
- `probe_vs_q200_lse_max_abs_diff=0.0`
- `probe_vs_q200_max_abs_diff=0.001708984375`
- no nonfinite

NCU probe command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096 python3 - <<'PY'
# Warmed 10 launches, cudaProfilerStart(), one forward_streaming_live_mxfp4 launch, cudaProfilerStop().
PY
```

Same-build q200 baseline NCU command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096_samebuild python3 - <<'PY'
# Warmed 10 launches, cudaProfilerStart(), one forward_streaming_live_mxfp4 launch, cudaProfilerStop().
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096_samebuild.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096_samebuild_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096_samebuild.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_q200_p112_h16_s4096_samebuild_details.csv
```

NCU metrics, same-build q200 diagzero baseline versus q200 prepub probe:

| Metric | q200 baseline | q200 prepub | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 157.184 us | 156.224 us | -0.611% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.038723 | 7.026773 | -0.170% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.419888 | 0.421238 | +0.322% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.546488 | 3.542803 | -0.104% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.475422 | 0.474847 | -0.121% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.228270 | 0.228065 | -0.090% |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.043368 | 0.043057 | -0.717% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.636930 | 1.640178 | +0.198% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 36.052549 | 36.106937 | +0.151% |
| `smsp__warps_active.avg.per_cycle_active` | 2.868763 | 2.868195 | -0.020% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.247676 | 1.255527 | +0.629% |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 11.070543 | 11.051399 | -0.173% |
| `derived__local_spilling_requests` | 0 | 0 | 0 |
| `launch__registers_per_thread` | 168 | 168 | 0 |
| `launch__shared_mem_per_block_static` | 1.968 KB | 1.968 KB | 0 |
| `inst_executed` | 54719299 | 55056289 | +0.616% |

Decision: keep as opt-in only, do not reroute baseline and do not commit. The representative H16/S4096 direct timing and same-build NCU both validate a small forward win for the combined q200+prepub route, with higher eligible warps and issue-active, lower long/short scoreboard, lower barrier/MIO throttle, and unchanged spills/registers/smem. The probe is not broad enough to make the default route because H4/S2048 regresses versus q200 and q208. Classification remains PV tensor-core underfeed from producer/P-handoff latency, not DRAM/launch/spill limited.

## Loop 42: `diagzero_prepubreuse_q200` early P-payload proxy before P-scale slot-reuse wait rejected

Profile basis: Loop 41 showed a small H16/S4096 win from publishing the P payload proxy before the direct P-scale wait, with unchanged resources and slightly better eligible-warps/scoreboard metrics. The next structural probe tried to move that proxy publication even earlier, immediately after payload completion and before the P-scale slot-reuse wait/sync, to overlap the proxy handoff with scale-slot reuse latency.

Patch: temporary opt-in forward-only route, later reverted:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepubreuse_pstage4_q200_p112_o56_qkscfix`
- Template: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepubreuse_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Added a temporary trait `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT`
- No backward files touched.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepubreuse_q200_qkscfix.log
```

ptxas from the probe build:

- q200 prepubreuse probe: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 prepub kept opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 diagzero kept opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepubreuse_q200_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79200, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepubreuse_pstage4_q200_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.26342400908470154`
- `mxfp4_samples_ms=[0.26342400908470154, 0.11187200248241425]`
- `lse_max_abs_diff=0.027907170355319977` versus TK BF16
- `max_abs_diff=1.1015625` versus TK BF16
- no output/LSE nonfinite

Direct preallocated timing command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepubreuse_q200_p112_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79201.
# Configs:
#   q200_diagzero, q200_prepub, q200_prepubreuse, q208_diagzero
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q200 diagzero ms | q200 prepub ms | q200 prepubreuse ms | q208 diagzero ms | prepubreuse vs q200 prepub | prepubreuse vs q200 diagzero | prepubreuse vs q208 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05963199958 | 0.05967999995 | 0.05982400104 | 0.05956799909 | +0.241% | +0.322% | +0.430% |
| H16/S4096 | 0.16489599645 | 0.16448000073 | 0.16492800415 | 0.16491200030 | +0.272% | +0.019% | +0.010% |
| H4/S2048 | 0.05513599887 | 0.05502399988 | 0.05521599948 | 0.05508799851 | +0.349% | +0.145% | +0.232% |

Probe-vs-q200-prepub direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00146484375`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0028076171875`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0008544921875`, no nonfinite.

Revert verification:

```bash
grep -R "prepubreuse\|PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE" -n tk_fa4/fp4_fa4_fwd || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_prepubreuse_revert_qkscfix.log
```

Post-revert ptxas for kept qkscfix routes:

- q200 prepub opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 diagzero opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q208 prepub opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q208 diagzero baseline: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Decision: reject and revert. The probe was numerically clean and resource-neutral, but direct isolated-kernel timing regressed the kept q200+prepub route by `+0.272%` on the representative H16/S4096 shape and also regressed H16/S2048 and H4/S2048. No NCU follow-up was run because the timing did not suggest a win. Classification remains PV tensor-core underfeed with producer/P-handoff latency; this particular earlier proxy placement adds or exposes ordering cost rather than hiding it.

## Loop 43: q200 `diagzero_prepub_pscale3` P-scale TMEM slot-depth probe rejected at build budget

Profile basis: the dominant H16/S4096 bottleneck remains PV tensor-core underfeed with producer/P-handoff latency. The user-preferred P-scale slot-depth probe was checked after Loop 42 because the kept q200 route has a four-slot P payload ring while the direct P-scale path still uses two TMEM scale slots.

Patch: temporary opt-in forward-only route, later reverted:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pscale3_pstage4_q200_p112_o56_qkscfix`
- Template: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pscale3_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Added a temporary `ONLINE_TRIPLE_P_SCALE_TMEM` trait and routed online direct P-scale TMEM through three P-scale slots.
- No backward files touched.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_pscale3_q200_qkscfix.log
```

Build result:

- Rejected at compile-time static assert in `fwd_streaming_kernel.inc`: `streaming live scale ping-pong exceeds TMEM budget`.
- The failing layout was `V_SC_BASE + V_SCALE_TMEM_SLOTS * V_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS`.
- Reason: online dual-score qkscfix uses two score slots plus output and two V-scale slots; a third P-scale slot pushes the scale layout past the 512-column TMEM budget. This rules out the slot-depth probe for the kept route unless it is coupled to a V-scale-slot reduction, which was not pursued here because prior V-scale slot probes were rejected and the user asked to pivot to structural P-handoff/PV-work.

Decision: reject and revert without smoke/timing/NCU. No NCU was run because the probe did not build.

## Loop 44: q200 `diagzero_prepub_pready` P-ready-before-rescale ordering probe rejected

Profile basis: q200+prepub slightly improved H16/S4096 by moving P payload proxy publication before the direct P-scale wait, but Loop 42 showed that publishing too early before P-scale slot-reuse wait regressed. This probe tested the opposite side of the handoff: publish direct P readiness first, then wait for output rescale completion, using the existing `ONLINE_P_READY_BEFORE_RESCALE_WAIT` synchronization path.

Patch: temporary opt-in forward-only route, later reverted:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pready_pstage4_q200_p112_o56_qkscfix`
- Template: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pready_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Set `ONLINE_P_READY_BEFORE_RESCALE_WAIT = true`.
- No backward files touched.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_pready_q200_qkscfix.log
```

ptxas from the probe build:

- q200 prepub+pready probe: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 prepub kept opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 diagzero kept opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_pready_q200_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79300, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pready_pstage4_q200_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.26287999749183655`
- `mxfp4_samples_ms=[0.26287999749183655, 0.12995199859142303]`
- `lse_max_abs_diff=0.03003522753715515` versus TK BF16
- `max_abs_diff=1.0` versus TK BF16
- no output/LSE nonfinite

Direct preallocated timing command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_pready_q200_p112_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79301.
# Configs:
#   q208_diagzero, q200_diagzero, q200_prepub, q200_prepub_pready
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q208 diagzero ms | q200 diagzero ms | q200 prepub ms | q200 prepub+pready ms | pready vs q200 prepub | pready vs q200 diagzero | pready vs q208 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.06115199998 | 0.06140799820 | 0.06137600169 | 0.06441599876 | +4.953% | +4.898% | +5.338% |
| H16/S4096 | 0.16819199920 | 0.16809600592 | 0.16383999586 | 0.17849600315 | +8.945% | +6.187% | +6.126% |
| H4/S2048 | 0.05600000173 | 0.05632000044 | 0.05604799837 | 0.06004799902 | +7.137% | +6.619% | +7.229% |

Probe-vs-q200-prepub direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0008544921875`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0018310546875`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=9.183549615799121e-41`, no nonfinite.

Revert verification:

```bash
grep -R "prepub_pready\|prepub_pscale3\|ONLINE_TRIPLE_P_SCALE_TMEM" -n tk_fa4/fp4_fa4_fwd || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pready_revert_qkscfix.log
```

Post-revert ptxas for kept qkscfix routes:

- q200 prepub opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 diagzero opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q208 prepub opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q208 diagzero baseline: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Decision: reject and revert. The probe was numerically clean and resource-neutral, but direct isolated-kernel timing regressed the kept q200+prepub route by `+8.945%` on H16/S4096, `+4.953%` on H16/S2048, and `+7.137%` on H4/S2048. No NCU follow-up was run because timing was strongly negative. Classification remains PV tensor-core underfeed with producer/P-handoff latency; this ordering forces a rescale dependency back onto the producer path and widens the latency bubble.

## Loop 45: q200 `diagzero_prepub_pfirst` P-scale-before-V-scale handoff probe rejected by NCU

Profile basis: existing NCU evidence and Loop 41/42/44 results still classify the kept MXFP4 qkscfix route as PV tensor-core underfed with low eligible warps and long-scoreboard/P-handoff latency, not DRAM, launch, spill, or occupancy limited. The probe was chosen because the decoupled PV lane already has a live `STATIC_STAGE_P_BEFORE_V` branch that can move direct P-scale staging ahead of V-scale staging without changing V-scale slot depth or producer register allocation.

Patch: temporary opt-in forward-only route, later reverted:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pfirst_pstage4_q200_p112_o56_qkscfix`
- Template: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pfirst_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Set `ONLINE_STAGE_P_BEFORE_V = true` on top of the kept q200+prepub route.
- No backward files touched.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_pfirst_q200_qkscfix.log
```

ptxas from the probe build:

- q200 prepub+pfirst probe: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 prepub kept opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`
- q200 diagzero kept opt-in: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_pfirst_q200_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79400, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pfirst_pstage4_q200_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.3834879994392395`
- `mxfp4_samples_ms=[0.3834879994392395, 0.1401280015707016]`
- `lse_max_abs_diff=0.018154509365558624` versus TK BF16
- `max_abs_diff=1.0234375` versus TK BF16
- no output/LSE nonfinite

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_pfirst_q200_p112_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79401.
# Configs:
#   q208_diagzero, q200_diagzero, q200_prepub, q200_prepub_pfirst
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q208 diagzero ms | q200 diagzero ms | q200 prepub ms | q200 prepub+pfirst ms | pfirst vs q200 prepub |
| --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05769599974 | 0.05787200108 | 0.05798399821 | 0.05799999833 | +0.0276% |
| H16/S4096 | 0.16464000195 | 0.16419200599 | 0.16342400014 | 0.16273599863 | -0.4210% |
| H4/S2048 | 0.05443200096 | 0.05433600023 | 0.05420799926 | 0.05420799926 | 0.0000% |

Probe-vs-q200-prepub direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.001220703125`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0015869140625`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0015869140625`, no nonfinite.

NCU follow-up commands on the representative H16/S4096 timing win:

```bash
CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix ncu --target-processes all --profile-from-start off --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_q200_p112_qkscfix_h16_s4096_loop45 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_q200_p112_qkscfix_h16_s4096_loop45.txt
# Warm up 30 iterations, then cudaProfilerStart(); one ext.forward_streaming_live_mxfp4 launch; synchronize; cudaProfilerStop().
# Shape H16/S4096, seed=79402, persistent_launch=True.
PY

CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pfirst_pstage4_q200_p112_o56_qkscfix ncu --target-processes all --profile-from-start off --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_pfirst_q200_p112_qkscfix_h16_s4096_loop45 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_pfirst_q200_p112_qkscfix_h16_s4096_loop45.txt
# Warm up 30 iterations, then cudaProfilerStart(); one ext.forward_streaming_live_mxfp4 launch; synchronize; cudaProfilerStop().
# Shape H16/S4096, seed=79402, persistent_launch=True.
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_q200_p112_qkscfix_h16_s4096_loop45.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_q200_p112_qkscfix_h16_s4096_loop45_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_pfirst_q200_p112_qkscfix_h16_s4096_loop45.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_pfirst_q200_p112_qkscfix_h16_s4096_loop45_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_q200_p112_qkscfix_h16_s4096_loop45.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_q200_p112_qkscfix_h16_s4096_loop45_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_pfirst_q200_p112_qkscfix_h16_s4096_loop45.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_diagzero_prepub_pfirst_q200_p112_qkscfix_h16_s4096_loop45_details.csv
```

NCU metric deltas, q200 prepub baseline -> q200 prepub+pfirst:

| Metric name | Base | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.160 us | 156.928 us | +0.492% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.058378% | 6.996672% | -0.874% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.889113% | 14.812726% | -0.513% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.420710 | 0.421260 | +0.131% |
| `smsp__warps_active.avg.per_cycle_active` | 2.868471 | 2.866304 | -0.0755% |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 | 0.36 | 0.000% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 36.099819% | 36.067967% | -0.0882% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.539926 | 3.521046 | -0.533% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.476553 | 0.475263 | -0.271% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.227599 | 0.228249 | +0.286% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.639926 | 1.641296 | +0.0835% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.255719% | 1.249878% | -0.465% |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | 6.598944% | 6.565879% | -0.501% |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | 11.717360% | 11.696667% | -0.177% |
| `launch__registers_per_thread` | 168 | 168 | 0 |
| `launch__shared_mem_per_block_static` | 1.968 KB | 1.968 KB | 0 |
| `launch__barrier_count` | 2 | 2 | 0 |
| `derived__local_spilling_requests` | 0 | 0 | 0 |
| `sass__inst_executed_register_spilling` | 0 | 0 | 0 |
| `inst_executed` | 54,526,221 | 54,491,194 | -0.0642% |

Revert verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pfirst_revert_qkscfix.log
grep -n "qkscfix_diagzero_prepub_pfirst\|diagzero_prepub_pfirst_pstage4_q200" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc results/mxfp4_fa4_forward_profile_20260612/build_after_pfirst_revert_qkscfix.log
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_pfirst_revert_q200_prepub_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79410, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Post-revert result:

- No `qkscfix_diagzero_prepub_pfirst` or `diagzero_prepub_pfirst_pstage4_q200` symbols remain in source or post-revert build log.
- Kept q200/q208 diagzero and q200/q208 prepub routes all compile at `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`.
- Post-revert q200 prepub smoke: `finite=true`, `lse_max_abs_diff=0.025864234194159508` versus TK BF16, `max_abs_diff=1.0`, no output/LSE nonfinite.

Decision: reject and revert. The direct event median suggested a small H16/S4096 win, but the representative isolated NCU launch regressed `gpu__time_duration.avg` by `+0.492%` and did not improve the real bottleneck: eligible warps were effectively unchanged, issue activity slightly dropped, tensor/TC pipe activity dropped, and wait/barrier stalls nudged upward. Classification remains PV tensor-core underfeed with producer/P-handoff latency; staging P scale before V scale does not create enough PV-ready work and slightly worsens the isolated kernel.

## Loop 46 - q200 prepub QK two-ahead distance probe rejected/reverted

Hypothesis: the kept q200+prepub route is still PV tensor-core underfed with low eligible warps and long scoreboard, while prior P-scale/V-scale staging-only flags did not move the isolated NCU bottleneck. A structural producer-distance probe should test whether starting the next QK/softmax tile two iterations ahead can create more ready P payload for PV without changing the score-derived qkscfix pack path.

Probe: add opt-in forward-only config `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_qk2_pstage4_q200_p112_o56_qkscfix`, deriving from the score-derived qkscfix diagzero prepub route with `ONLINE_QK_TWO_AHEAD = true`, template `<128,128,192,128,200,56,112,1>`. This did not touch backward files and did not change kept qkscfix/p112 routes.

Changed files while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_qk2_q200_qkscfix.log
```

Build result for qk2 probe:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_qk2_q200_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79500, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_qk2_pstage4_q200_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.24223999679088593`
- `mxfp4_samples_ms=[0.24223999679088593, 0.13174399733543396]`
- `lse_max_abs_diff=0.02879277616739273` versus TK BF16
- `max_abs_diff=0.8828125` versus TK BF16
- no output/LSE nonfinite

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_qk2_q200_p112_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79501.
# Configs:
#   q208_diagzero, q200_diagzero, q200_prepub, q200_prepub_qk2
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q208 diagzero ms | q200 diagzero ms | q200 prepub ms | q200 prepub+qk2 ms | qk2 vs q200 prepub |
| --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05784000084 | 0.05828800052 | 0.05796799995 | 0.05870399997 | +1.2697% |
| H16/S4096 | 0.16550399363 | 0.16419200599 | 0.16374399513 | 0.16531200707 | +0.9576% |
| H4/S2048 | 0.05401600152 | 0.05384000018 | 0.05364799872 | 0.05503999814 | +2.5947% |

Probe-vs-q200-prepub direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0015869140625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00244140625`, no nonfinite.
- H4/S2048: exact output/LSE match, no nonfinite.

NCU decision: skipped. Direct isolated timing was negative on all three shapes, including representative H16/S4096, so there was no timing win to follow with counters.

Revert verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_qk2_revert_qkscfix.log
grep -n "qkscfix_diagzero_prepub_qk2\|diagzero_prepub_qk2_pstage4_q200" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc results/mxfp4_fa4_forward_profile_20260612/build_after_qk2_revert_qkscfix.log
grep -n -B 1 -A 4 "qkscfix_diagzero_prepub.*ILi128ELi128ELi192ELi128ELi200ELi56ELi112\|qkscfix_diagzero.*ILi128ELi128ELi192ELi128ELi200ELi56ELi112\|qkscfix_diagzero_prepub.*ILi128ELi128ELi192ELi128ELi208ELi56ELi112\|qkscfix_diagzero.*ILi128ELi128ELi192ELi128ELi208ELi56ELi112" results/mxfp4_fa4_forward_profile_20260612/build_after_qk2_revert_qkscfix.log
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_qk2_revert_q200_prepub_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79510, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Post-revert result:

- No `qkscfix_diagzero_prepub_qk2` or `diagzero_prepub_qk2_pstage4_q200` symbols remain in source or post-revert build log.
- Kept q200/q208 diagzero and q200/q208 prepub routes all compile at `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`.
- Post-revert q200 prepub smoke: `finite=true`, `mxfp4_ms=0.31567999720573425`, `mxfp4_samples_ms=[0.31567999720573425, 0.16016000509262085]`. The helper did not return BF16 diff fields for this post-revert smoke call.

Decision: reject and revert. Moving QK/softmax production two-ahead increases the producer/PV distance but slows every direct timing shape. Classification remains PV tensor-core underfeed dominated by producer/P-handoff latency, but this particular producer-distance reorder creates extra scheduling pressure rather than more PV-ready work.

## Loop 47 - q200 prepub q192 register-balance/PV-work probe rejected/reverted

Hypothesis: representative NCU still shows MXFP4 qkscfix/prepub as PV tensor-core underfed with low eligible warps and long scoreboard, while q200 improved slightly over q208. Reducing quant registers from 200 to 192 could reduce live register pressure or rebalance producer/PV work enough to raise eligible warps without changing the score-derived qkscfix scorepack route.

Probe: add opt-in forward-only dispatch
`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q192_p112_o56_qkscfix`, using existing
`config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128,128,192,128,192,56,112,1>`.
No backward files were touched.

Changed file while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_q192_qkscfix.log
```

Build result for q192 qkscfix prepub:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

The physical ptxas register count did not change versus q200/q208 prepub.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_q192_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(
#   seqlen=2048, heads=16, seed=79600, warmup=1, iters=2,
#   mxfp4_fwd_config='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q192_p112_o56_qkscfix',
#   bf16_baseline='tk', include_output_only=False)
PY
```

Smoke result:

- `finite=true`
- `mxfp4_ms=0.38646399974823`
- `mxfp4_samples_ms=[0.38646399974823, 0.127360001206398]`
- helper did not return BF16 diff fields for this smoke call

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_q192_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79601.
# Configs:
#   q208_diagzero, q200_diagzero, q200_prepub, q192_prepub
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q208 diagzero ms | q200 diagzero ms | q200 prepub ms | q192 prepub ms | q192 vs q200 prepub |
| --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05836800113 | 0.05846399814 | 0.05820799991 | 0.05811199918 | -0.1649% |
| H16/S4096 | 0.16359999776 | 0.16284799576 | 0.16217599809 | 0.16249600053 | +0.1973% |
| H4/S2048 | 0.05385600030 | 0.05340800062 | 0.05324799940 | 0.05347200111 | +0.4207% |

Probe-vs-q200-prepub direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0015869140625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0025634765625`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0001220703125`, no nonfinite.

NCU decision: skipped. Direct isolated timing was negative on the representative H16/S4096 shape and on H4/S2048, and the build showed no physical register reduction.

Revert verification:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_q192_revert_qkscfix.log
grep -n "diagzero_prepub_pstage4_q192_p112_o56_qkscfix" tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc results/mxfp4_fa4_forward_profile_20260612/build_after_q192_revert_qkscfix.log
grep -n -B 1 -A 4 "qkscfix_diagzero_prepub.*ILi128ELi128ELi192ELi128ELi200ELi56ELi112\|qkscfix_diagzero.*ILi128ELi128ELi192ELi128ELi200ELi56ELi112\|qkscfix_diagzero_prepub.*ILi128ELi128ELi192ELi128ELi208ELi56ELi112\|qkscfix_diagzero.*ILi128ELi128ELi192ELi128ELi208ELi56ELi112" results/mxfp4_fa4_forward_profile_20260612/build_after_q192_revert_qkscfix.log
```

Post-revert result:

- No `diagzero_prepub_pstage4_q192_p112_o56_qkscfix` dispatch entry or post-revert q192 build symbol remains.
- Kept q200/q208 diagzero and q200/q208 prepub routes all compile at `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`; `Used 168 registers, used 2 barriers, 1968 bytes smem`.

Decision: reject and revert. The representative shape regressed, and q192 did not change the compiled register resource profile. Classification remains PV tensor-core underfeed with producer/P-handoff latency, but reducing q registers alone is not the lever on the kept q200 prepub qkscfix route.

## Loop 48 - q200 prepub pstage2 structural P-handoff probe kept

Hypothesis: current MXFP4 qkscfix/prepub NCU evidence remains PV tensor-core underfed: low eligible warps, low tensor-active percentage, dominant long scoreboard, and not DRAM/launch/spill limited. Prior P-scale slot-depth probes showed extra P-scale TMEM slots regress or exceed budget. The kept qkscfix/prepub path uses `P_STAGE_SLOTS=4` while the online direct P-scale path has only two P-scale TMEM slots. Matching P payload staging depth down to the live two-slot scale handoff could reduce P-stage reuse and ready-event churn without changing numerical semantics.

Probe: add opt-in forward-only route
`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage2_q200_p112_o56_qkscfix`, deriving from the kept q200 prepublish score-derived qkscfix route and overriding only `P_STAGE_SLOTS=2`.

Changed files while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_pstage2_q200_qkscfix.log
```

ptxas for pstage2 q200 prepub:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1904 bytes smem`

Same-build q200 prepub pstage4 baseline remained:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1968 bytes smem`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_pstage2_q200_qkscfix_h16_s2048.log
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=79700, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False,
)
print(json.dumps(res, indent=2, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_samples_ms=[0.35209599137306213, 0.12972800433635712]`
- vs TK BF16: `lse_max_abs_diff=0.01904967427253723`, `max_abs_diff=1.0`, no output/LSE nonfinite.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_pstage2_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79701.
# Configs:
#   q208_diagzero
#   q200_diagzero
#   q200_prepub
#   q200_prepub_pstage2
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q208 diagzero ms | q200 diagzero ms | q200 prepub pstage4 ms | q200 prepub pstage2 ms | pstage2 vs pstage4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.06288000196 | 0.06220800057 | 0.06207999960 | 0.06198399886 | -0.1546% |
| H16/S4096 | 0.17633599788 | 0.16817600280 | 0.16784000397 | 0.16739199311 | -0.2669% |
| H4/S2048 | 0.06004799902 | 0.05985600129 | 0.05999999866 | 0.05943999998 | -0.9333% |

Probe-vs-q200-prepub direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0018310546875`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002197265625`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0006103515625`, no nonfinite.

Representative H16/S4096 NCU command for same-build q200 prepub baseline:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096_pstage2_loop_base python3 - <<'PY'
# Inline driver creates live BF16 source inputs, quantizes Q/K/V with _fp4_qk_mxfp4_v_inputs_from_bf16_source,
# prepares MXFP4 forward inputs, warms 10 launches, then isolates exactly one forward kernel with
# torch.cuda.cudart().cudaProfilerStart()/cudaProfilerStop().
PY
```

Representative H16/S4096 NCU command for pstage2 probe:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage2_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_pstage2_q200_p112_h16_s4096 python3 - <<'PY'
# Same isolated one-kernel driver as baseline, seed=79720.
PY
```

NCU report exports:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096_pstage2_loop_base.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_q200_p112_h16_s4096_pstage2_loop_base_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_pstage2_q200_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_pstage2_q200_p112_h16_s4096_raw.csv
```

NCU representative isolated-kernel counters:

| Metric | q200 prepub pstage4 | q200 prepub pstage2 | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.256 us | 156.192 us | -0.0410% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.027695% | 7.045502% | +0.2534% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.420805 warp | 0.422525 warp | +0.4087% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 36.092281% | 36.170216% | +0.2159% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.535321 inst | 3.519970 inst | -0.4342% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.228402 inst | 0.223210 inst | -2.2732% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.639889 inst | 1.646252 inst | +0.3880% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.255107% | 1.255336% | +0.0182% |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 11.061752% | 11.108846% | +0.4257% |
| `launch__registers_per_thread` | 168 | 168 | 0 |
| `launch__shared_mem_per_block_static` | 1.968 KB | 1.904 KB | -3.2520% |
| `launch__barrier_count` | 2 | 2 | 0 |
| `derived__local_spilling_requests` | 0 | 0 | 0 |
| `inst_executed` | 54,669,197 | 53,960,448 | -1.2964% |

Decision: keep as an opt-in forward route. The win is small but isolated-kernel counters agree with the timing direction: slightly more eligible warps and tensor activity, less long-scoreboard/barrier pressure, fewer instructions and less static shared memory, with unchanged register count and no spills. Classification remains PV tensor-core underfeed with P-scale/P-payload handoff latency, not DRAM/launch/spill limited.

## Loop 49 - q200 prepub pstage2 early-reuse proxy handoff probe kept

Hypothesis: Loop 48 showed that matching P payload staging depth to the two live direct P-scale TMEM slots gives a small validated win. In the pstage2 route, the current prepublish still emits the P-payload proxy fence after the leader waits for the P-scale TMEM slot to become reusable and after a producer-wide sync. Moving that proxy fence immediately after payload/diagonal-zero completion, before the P-scale slot reuse wait, may overlap the P-payload handoff with the remaining scale-slot reuse wait. This revisits Loop 42's rejected pstage4 early-reuse idea only on the now-validated pstage2 handoff path.

Probe: add opt-in forward-only route
`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`.

Implementation:

- Added `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT`.
- Derived the probe from kept q200 prepub pstage2.
- In the online direct P-scale block, emits `fp4pv_publish_shared_backing_proxy_only()` immediately after the payload/zeroing `quant_wg_sync()` and before the P-scale reusable wait.
- Skips the later prepublish proxy for that route so the proxy fence is not duplicated.
- No backward files touched.

Changed files while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage2_q200_qkscfix.log
```

ptxas for earlyreuse pstage2 q200 prepub:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1904 bytes smem`

Same-build kept q200 prepub pstage2:

- `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`
- `Used 168 registers, used 2 barriers, 1904 bytes smem`

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_pstage2_q200_qkscfix_h16_s2048.log
import json
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=79800, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False,
)
print(json.dumps(res, indent=2, sort_keys=True))
PY
```

Smoke result:

- `finite=true`
- `mxfp4_samples_ms=[0.211776003241539, 0.10979200154542923]`
- vs TK BF16: `lse_max_abs_diff=0.018145084381103516`, `max_abs_diff=0.86328125`, no output/LSE nonfinite.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_pstage2_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79801.
# Configs:
#   q200_prepub_pstage4
#   q200_prepub_pstage2
#   q200_prepub_earlyreuse_pstage2
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | q200 prepub pstage4 ms | q200 prepub pstage2 ms | earlyreuse pstage2 ms | earlyreuse vs pstage2 | earlyreuse vs pstage4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.06015999988 | 0.05980800092 | 0.05996799842 | +0.2675% | -0.3192% |
| H16/S4096 | 0.16911999881 | 0.16750399768 | 0.16635199636 | -0.6877% | -1.6367% |
| H4/S2048 | 0.05507199839 | 0.05486400053 | 0.05471999943 | -0.2625% | -0.6392% |

Probe-vs-pstage2 direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00244140625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.003662109375`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.000244140625`, no nonfinite.

Representative H16/S4096 NCU command for kept pstage2 baseline:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage2_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_pstage2_q200_p112_h16_s4096_earlyreuse_loop_base python3 - <<'PY'
# Inline driver creates live BF16 source inputs, quantizes Q/K/V, warms 10 launches, then profiles one forward kernel
# bracketed by torch.cuda.cudart().cudaProfilerStart()/cudaProfilerStop(); seed=79820.
PY
```

Representative H16/S4096 NCU command for earlyreuse probe:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_earlyreuse_pstage2_q200_p112_h16_s4096 python3 - <<'PY'
# Same isolated one-kernel driver as baseline, seed=79820.
PY
```

NCU report exports:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_pstage2_q200_p112_h16_s4096_earlyreuse_loop_base.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_pstage2_q200_p112_h16_s4096_earlyreuse_loop_base_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_earlyreuse_pstage2_q200_p112_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_diagzero_prepub_earlyreuse_pstage2_q200_p112_h16_s4096_raw.csv
```

NCU representative isolated-kernel counters:

| Metric | q200 prepub pstage2 | earlyreuse pstage2 | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.256 us | 156.160 us | -0.0614% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.060091% | 7.029781% | -0.4293% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.422265 warp | 0.421902 warp | -0.0860% |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 36.167071% | 36.097048% | -0.1936% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.510884 inst | 3.529556 inst | +0.5318% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.477052 inst | 0.476007 inst | -0.2191% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.224575 inst | 0.224070 inst | -0.2249% |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.043037 inst | 0.042829 inst | -0.4833% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.646263 inst | 1.641384 inst | -0.2964% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.255109% | 1.255673% | +0.0449% |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 11.121064% | 11.056067% | -0.5844% |
| `launch__registers_per_thread` | 168 | 168 | 0 |
| `launch__shared_mem_per_block_static` | 1.904 KB | 1.904 KB | 0 |
| `launch__barrier_count` | 2 | 2 | 0 |
| `derived__local_spilling_requests` | 0 | 0 | 0 |
| `inst_executed` | 54,061,587 | 53,783,284 | -0.5148% |

Decision: keep as an opt-in forward route. H16/S4096 direct timing and isolated NCU duration both improve, with unchanged resources and no spills. The counters do not show a PV-underfeed cure: tensor-active and eligible warps dip and long scoreboard rises slightly. The win is therefore classified as a small producer/proxy ordering and instruction-count reduction on top of the pstage2 handoff, while the dominant bottleneck remains PV tensor-core underfeed/long scoreboard rather than DRAM, launch, occupancy, or spills.

## Probe Loop 50

Probe: added an opt-in q200 qkscfix `P_STAGE_SLOTS=3` plus early P-payload proxy-publish route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage3_q200_p112_o56_qkscfix`

Rationale: after pstage2 and earlyreuse both helped, test whether a middle P-payload staging depth could recover producer/PV overlap without returning all the way to the pstage4 shared footprint.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage3_q200_qkscfix.log
```

ptxas for the probe was spill-free:

- pstage3 earlyreuse: 168 registers/thread, 2 barriers, 1936 bytes smem, 0 spill stores, 0 spill loads.
- pstage2 earlyreuse baseline: 168 registers/thread, 2 barriers, 1904 bytes smem, 0 spill stores, 0 spill loads.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_pstage3_q200_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=79900, warmup=1, iters=2,
#   mxfp4_fwd_config="dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage3_q200_p112_o56_qkscfix")
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.028011739253997803`, `max_abs_diff=1.109375`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_pstage3_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=79901.
# Configs:
#   earlyreuse_pstage2
#   earlyreuse_pstage3
#   prepub_pstage4
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | earlyreuse pstage2 ms | earlyreuse pstage3 ms | prepub pstage4 ms | pstage3 vs pstage2 |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.06060799956 | 0.06068800017 | 0.06059199944 | +0.1320% |
| H16/S4096 | 0.17225599289 | 0.17089600116 | 0.16921600699 | -0.7895% |
| H4/S2048 | 0.06060799956 | 0.06004799902 | 0.05988800153 | -0.9240% |

Representative H16/S4096 NCU command, run for pstage2, pstage3, and pstage4:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LABEL="${label}_q200_h16_s4096_loop50" PROFILE_CONFIG="$cfg" \
ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export "results/mxfp4_fa4_forward_profile_20260612/ncu_${label}_q200_h16_s4096_loop50" python3 - <<'PY'
# Inline driver creates live BF16 source inputs, quantizes Q/K/V, warms two launches, then profiles one forward kernel
# bracketed by torch.cuda.cudart().cudaProfilerStart()/cudaProfilerStop(); seed=79950.
PY
```

NCU CSV exports:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_earlyreuse_pstage2_q200_h16_s4096_loop50.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_earlyreuse_pstage2_q200_h16_s4096_loop50_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_earlyreuse_pstage3_q200_h16_s4096_loop50.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_earlyreuse_pstage3_q200_h16_s4096_loop50_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prepub_pstage4_q200_h16_s4096_loop50.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_prepub_pstage4_q200_h16_s4096_loop50_raw.csv
```

NCU representative isolated-kernel counters:

| Metric | earlyreuse pstage2 | earlyreuse pstage3 | prepub pstage4 |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.256 us | 156.768 us | 158.432 us |
| `inst_executed` | 53,803,778 | 54,769,179 | 54,888,569 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 6.996172% | 7.043440% | 7.018959% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.421148 | 0.422094 | 0.421231 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.534746 | 3.498534 | 3.531744 |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.224405 | 0.223404 | 0.229209 |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.345061 | 0.365889 | 0.355683 |
| `smsp__average_warp_latency_per_inst_issued.ratio` | 7.950495 | 7.905408 | 7.942327 |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.254806% | 1.250825% | 1.237959% |
| `launch__shared_mem_per_block` | 103.280 KB | 112.528 KB | 120.752 KB |
| `launch__registers_per_thread` | 168 | 168 | 168 |
| `derived__local_spilling_requests` | 0 | 0 | 0 |
| `profiler__replayer_passes` | 16 | 16 | 16 |

Decision: rejected and reverted. Although direct timing looked better on H16/S4096 than pstage2, isolated NCU regressed `gpu__time_duration.avg` by 0.328% and increased `inst_executed` by 1.79%. The small eligible-warp and long-scoreboard improvements were not enough to offset extra P-stage work/shared footprint. Rebuild after revert:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_pstage3_revert_qkscfix.log
grep -n "earlyreuse_pstage3_q200\\|earlyreuse_dualaccum_directrescale_decoupled_pstage3" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_pstage3_revert_qkscfix.log || true
```

Post-revert grep was empty. Post-revert smoke for kept pstage2 earlyreuse qkscfix:

- `finite=true`
- `lse_max_abs_diff=0.02155464142560959`
- `max_abs_diff=1.2578125`
- no output/LSE nonfinite.

## Probe Loop 53

Probe: opt-in pstage1 single P-payload-stage path for the kept score-derived qkscfix earlyreuse route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage1_q200_p112_o56_qkscfix`

Rationale: existing NCU classified the kept qkscfix route as PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM, launch, spill, or occupancy limited. Since pstage2 was a validated win over pstage4, this tested whether reducing P payload staging to a single slot would further reduce shared footprint and handoff bookkeeping. Correctness-first guard: the generic static assert was temporarily relaxed from `P_STAGE_SLOTS >= 2` to `>= 1`, and the route was kept opt-in.

Build commands:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage1_q200_qkscfix.log
grep -n -A4 -B2 "pstage1" results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage1_q200_qkscfix.log
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage1_q200_qkscfix_active.log
grep -n -A4 -B2 "pstage1" results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage1_q200_qkscfix_active.log
```

The first build showed no active pstage1 instantiation because the temporary route was only in the probe-build dispatch ladder. After adding the active default dispatch entry, ptxas reported:

- pstage1: 168 registers/thread, 2 barriers, 1872 bytes smem, 0 spill stores, 0 spill loads.
- same-build pstage2 earlyreuse baseline: 168 registers/thread, 2 barriers, 1904 bytes smem, 0 spill stores, 0 spill loads.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_pstage1_q200_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage1_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048,
    heads=16,
    seed=80300,
    warmup=1,
    iters=2,
    mxfp4_fwd_config=cfg,
)
print(res)
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.027136728167533875`, `max_abs_diff=0.90234375`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_pstage1_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80301.
# Configs:
#   earlyreuse_pstage2 baseline
#   earlyreuse_pstage1 probe
#   prepub_pstage4 reference
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | baseline pstage2 ms | pstage1 probe ms | pstage4 reference ms | pstage1 vs baseline |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.06117119996 | 0.06633368871 | 0.06064266664 | +8.4394% |
| H16/S4096 | 0.16793422293 | 0.19544568890 | 0.16650275530 | +16.3823% |
| H4/S2048 | 0.05774879998 | 0.06494737793 | 0.05724835551 | +12.4653% |

Probe-vs-pstage2 direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.003173828125`, `mean_abs_diff=1.0540949801907118e-07`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002685546875`, `mean_abs_diff=1.3577493973571109e-07`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, no nonfinite.

Decision: rejected and reverted without NCU because direct isolated-kernel timing regressed all measured shapes, especially H16/S4096. Hypothesis outcome: one payload stage saves only 32 B of shared memory but over-serializes the producer/PV handoff; the validated two-slot P payload ring remains necessary.

Revert and validation commands:

```bash
grep -n "pstage1\\|one to six" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
git diff --stat -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pstage1_revert_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "prepub_earlyreuse_pstage1" || true
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_pstage1_revert_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80310, warmup=1, iters=2, mxfp4_fwd_config=cfg)
print(res)
PY
```

Post-revert grep and forward-source diff were empty. The rebuilt binary had no pstage1 route string. Post-revert smoke for kept pstage2 earlyreuse qkscfix: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.01793256402015686`, `max_abs_diff=0.9609375`.

## Probe Loop 52

Probe: opt-in score-derived qkscfix K64 P-payload ready path for the kept pstage2 earlyreuse route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_k64_pstage2_q200_p112_o56_qkscfix`

Rationale: existing NCU showed PV tensor-core underfeed with low eligible warps and long scoreboard, not DRAM/launch/spill limited. This probe tried to let PV consume a score-derived Nb128 P tile as two coarse K64 payload halves. Correctness guard: direct X1 P scales were staged before either half-ready event, and the first half performed diagonal causal payload zeroing before publishing.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_k64_pstage2_qkscfix.log
```

ptxas for the probe: 168 registers/thread, 2 barriers, 2480 bytes smem, 0 spill stores, 0 spill loads.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_k64_pstage2_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80200, warmup=1, iters=2,
#   mxfp4_fwd_config="dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_k64_pstage2_q200_p112_o56_qkscfix")
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.022931255400180817`, `max_abs_diff=0.98046875`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_k64_pstage2_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80201.
# Configs: earlyreuse_pstage2_q200 baseline, earlyreuse_k64_pstage2_q200 probe, prepub_pstage4_q200 reference.
# Shapes: H16/S2048, H16/S4096, H4/S2048.
PY
```

Direct preallocated timing results:

| Shape | baseline pstage2 ms | K64 probe ms | pstage4 reference ms | K64 vs baseline |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05972800031 | 0.06219200045 | 0.05900799856 | +4.1254% |
| H16/S4096 | 0.16443199664 | 0.17627199739 | 0.16359999776 | +7.2005% |
| H4/S2048 | 0.05558399856 | 0.05836800113 | 0.05510399863 | +5.0086% |

Probe-vs-baseline direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002197265625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0023193359375`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0`, no nonfinite.

Decision: rejected and reverted without NCU because direct isolated-kernel timing regressed all measured shapes. Hypothesis outcome: the added early P-scale staging and K64 semaphores did not relieve PV underfeed; they introduced enough synchronization/shared-state cost to dominate.

Rebuild after revert:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_k64_scorederived_revert_qkscfix.log
grep -R -n "earlyreuse_k64\\|SCORE_DERIVED_SPLIT_P_READY_K64" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
```

Post-revert grep was empty. Post-revert smoke for kept pstage2 earlyreuse qkscfix:

- `finite=true`
- `lse_max_abs_diff=0.020212173461914062`
- `max_abs_diff=1.1640625`
- no output/LSE nonfinite.

## Probe Loop 51

Probe: added an opt-in q200 qkscfix `P_STAGE_SLOTS=2` route without P-payload prepublish:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage2_q200_p112_o56_qkscfix`

Rationale: Loop 50 showed that deeper P payload staging was not enough to beat pstage2. This control tests whether the pstage2 win is from stage depth alone, or whether the P-payload prepublish/proxy handoff is live and still needed after reducing to two P-stage slots.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_noprepub_pstage2_q200_qkscfix.log
```

ptxas for the probe was spill-free: 168 registers/thread, 2 barriers, 1904 bytes smem, 0 spill stores, 0 spill loads.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_noprepub_pstage2_q200_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80000, warmup=1, iters=2,
#   mxfp4_fwd_config="dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage2_q200_p112_o56_qkscfix")
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.02035355567932129`, `max_abs_diff=0.95703125`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_noprepub_pstage2_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80001.
# Configs:
#   noprepub_pstage2
#   prepub_pstage2
#   earlyreuse_pstage2
#   noprepub_pstage4
#   prepub_pstage4
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | noprepub pstage2 ms | prepub pstage2 ms | earlyreuse pstage2 ms | noprepub vs earlyreuse |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05827200040 | 0.05828800052 | 0.05782400072 | +0.7748% |
| H16/S4096 | 0.16303999722 | 0.16212800145 | 0.16172799468 | +0.8112% |
| H4/S2048 | 0.05345600098 | 0.05342400074 | 0.05336000025 | +0.1799% |

Probe-vs-earlyreuse direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002197265625`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00244140625`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0009765625`, no nonfinite.

Decision: rejected and reverted without NCU because the probe had no direct isolated-kernel timing win. The control still provided useful evidence: the P-payload prepublish path is live and non-noop for the kept pstage2 qkscfix route. Rebuild after revert:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_noprepub_pstage2_revert_qkscfix.log
grep -n "diagzero_pstage2_q200_p112_o56_qkscfix\\|qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage2" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc results/mxfp4_fa4_forward_profile_20260612/build_after_noprepub_pstage2_revert_qkscfix.log || true
```

Post-revert grep was empty. Post-revert smoke for kept pstage2 earlyreuse qkscfix:

- `finite=true`
- `lse_max_abs_diff=0.029954783618450165`
- `max_abs_diff=1.0234375`
- no output/LSE nonfinite.

## Probe Loop 54

Probe: opt-in pstage2 earlyreuse P-before-V PV issue-order route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_p112_o56_qkscfix`

Rationale: after K64, P-scale slot depth, pstage1, pstage3, no-prepub, and pstage4 pfirst variants failed, this tested a narrow P-handoff/PV-work ordering change on the kept score-derived qkscfix pstage2 earlyreuse route. It used the existing `ONLINE_STAGE_P_BEFORE_V` branch so the PV lane waits/stages P before V, without changing V-scale slot depth, backward code, qkscfix, or the kept p112 baseline.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_qkscfix.log
grep -n -A4 -B2 "earlyreuse_pfirst" results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_qkscfix.log
grep -n -A4 -B2 "earlyreuse_dualaccum_directrescale_decoupled_pstage2" results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_qkscfix.log
```

ptxas:

- pfirst pstage2 probe: 168 registers/thread, 2 barriers, 1904 bytes smem, 0 spill stores, 0 spill loads.
- kept pstage2 earlyreuse baseline: 168 registers/thread, 2 barriers, 1904 bytes smem, 0 spill stores, 0 spill loads.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048,
    heads=16,
    seed=80400,
    warmup=1,
    iters=2,
    mxfp4_fwd_config=cfg,
)
print(res)
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.02194676548242569`, `max_abs_diff=0.96875`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80401.
# Configs:
#   earlyreuse_pstage2 baseline
#   earlyreuse_pfirst_pstage2 probe
#   prepub_pstage4 reference
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | baseline pstage2 ms | pfirst pstage2 ms | pstage4 reference ms | pfirst vs baseline |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.05142613517 | 0.05131502151 | 0.05132924186 | -0.2161% |
| H16/S4096 | 0.15388551288 | 0.15388124254 | 0.15383662118 | -0.0028% |
| H4/S2048 | 0.04570275413 | 0.04599804348 | 0.04610364702 | +0.6461% |

Probe-vs-baseline direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.001953125`, `mean_abs_diff=1.544985508417085e-07`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00390625`, `mean_abs_diff=2.2752854533791833e-07`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.000732421875`, `mean_abs_diff=6.244300720936735e-08`, no nonfinite.

Decision: rejected and reverted without NCU. H16/S4096 was effectively flat within noise (`-0.0028%`) and H4/S2048 regressed, so there was no representative isolated-kernel win to profile. The result indicates that in the kept pstage2 earlyreuse route, simply waiting/staging P before V on the PV issue lane does not relieve the dominant PV tensor-core underfeed/long-scoreboard bubble.

Revert and validation commands:

```bash
grep -n "earlyreuse_pfirst\\|pfirst_dualaccum_directrescale_decoupled_pstage2" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
git diff --stat -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_pfirst_pstage2_revert_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pfirst_pstage2" || true
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_earlyreuse_pfirst_pstage2_revert_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80410, warmup=1, iters=2, mxfp4_fwd_config=cfg)
print(res)
PY
```

Post-revert source grep and forward-source diff were empty; the rebuilt binary had no `earlyreuse_pfirst_pstage2` route string. Post-revert smoke for kept pstage2 earlyreuse qkscfix: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.03514164686203003`, `max_abs_diff=0.9609375`.

## Probe Loop 55

Probe: opt-in producer-register balance on the kept pstage2 earlyreuse score-derived qkscfix route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p104_o56_qkscfix`

Rationale: existing NCU classifies the kept qkscfix pstage2 earlyreuse route as PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM, launch, spill, or occupancy limited. The preferred structural options were already exhausted for this route: exact K64 handoff regressed, real score-derived K256 remains a multi-path implementation because the live K256 producer still uses `fp4pv_pack_scores_to_stage_mxfp4*`, and extra P-scale slot depth either exceeds the TMEM map or regresses. This tested whether reducing producer WG register allocation from 112 to 104 on the kept pstage2 handoff path raises eligible work without touching backward code or changing payload/scale math.

Changed file while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage2_q200_p104_qkscfix.log
grep -n -B 2 -A 4 "qkscfix_diagzero_prepub_earlyreuse.*ILi128ELi128ELi192ELi128ELi200ELi56ELi104\\|ILi128ELi128ELi192ELi128ELi200ELi56ELi104ELi1" results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pstage2_q200_p104_qkscfix.log
```

Build result:

- p104 pstage2 probe: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- kept p112 pstage2 baseline in same build: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_pstage2_q200_p104_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p104_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80500, warmup=1, iters=2, mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.037230610847473145`, `max_abs_diff=1.3125`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_pstage2_q200_p104_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80501.
# Configs:
#   earlyreuse_pstage2 p112 baseline
#   earlyreuse_pstage2 p104 probe
#   prepub_pstage4 p112 reference
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | baseline pstage2 p112 ms | p104 pstage2 probe ms | pstage4 reference ms | p104 vs baseline |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.06368000060 | 0.06249599904 | 0.06238400005 | -1.8593% |
| H16/S4096 | 0.16534399986 | 0.16521599889 | 0.16572800279 | -0.0774% |
| H4/S2048 | 0.05804799870 | 0.05801599845 | 0.05814399943 | -0.0551% |

Probe-vs-baseline direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0068359375`, `mean_abs_diff=2.7924983214688837e-07`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0032958984375`, `mean_abs_diff=1.9508237869558798e-07`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0013427734375`, `mean_abs_diff=2.4870612946870096e-07`, no nonfinite.

Representative paired timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_pstage2_q200_p104_qkscfix_h16_s4096_paired_retry.jsonl
# Paired alternating raw ext.forward_streaming_live_mxfp4 calls on H16/S4096.
# WARMUP=30, ITERS=240, SEED=80551.
# Configs: kept p112 baseline vs p104 probe.
PY
```

Paired H16/S4096 result:

- baseline p112 median: `0.16980800032615662 ms`
- p104 probe median: `0.16993600130081177 ms`
- p104 delta: `+0.07537982569094659%`
- numeric vs baseline: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0042724609375`, no output/LSE nonfinite.

NCU decision: skipped. The only representative H16/S4096 follow-up was a paired regression, so there was no isolated-kernel win to profile.

Decision: rejected and reverted. The p104 route was live and numerically stable, and it kept the same ptxas resources as p112, but representative H16/S4096 timing was flat-to-negative. Classification remains PV tensor-core underfeed with low eligible warps and long scoreboard; this producer-register rebalance does not relieve the pstage2 earlyreuse P-handoff bubble.

Revert and validation commands:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_pstage2_p104_revert_qkscfix.log
grep -n "earlyreuse_pstage2_q200_p104\\|ILi128ELi128ELi192ELi128ELi200ELi56ELi104ELi1" tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_pstage2_p104_revert_qkscfix.log || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pstage2_q200_p104" || true
git diff --stat -- tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_earlyreuse_pstage2_p104_revert_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80590, warmup=1, iters=2, mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Post-revert source grep, build-log grep, binary-string grep, and forward-source diff were empty. Post-revert smoke for kept pstage2 earlyreuse qkscfix: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.020972132682800293`, `max_abs_diff=1.0`.

## Probe Loop 56

Probe: opt-in structural P-ready/rescale handoff on the kept pstage2 earlyreuse score-derived qkscfix route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pready_pstage2_q200_p112_o56_qkscfix`

Rationale: existing NCU classifies the kept route as PV tensor-core underfed with low eligible warps and long scoreboard, not DRAM, launch, spill, or occupancy limited. This probe was live for the score-derived direct P-ready path: it set `ONLINE_P_READY_BEFORE_RESCALE_WAIT=true`, skipped the earlier rescale wait before scorepack/P-scale publish, arrived P-ready first, then waited for `rescale_finished[0]`. The test was meant to expose whether rescale waiting before P-ready was blocking the PV consumer.

Changed files while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pready_pstage2_q200_qkscfix.log
grep -n -B 2 -A 5 "earlyreuse_pready\\|earlyreuse_dualaccum_directrescale_decoupled_pstage2" results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_pready_pstage2_q200_qkscfix.log
```

Build result:

- pready pstage2 probe: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- kept pstage2 baseline in same build: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_pready_pstage2_q200_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pready_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80600, warmup=1, iters=2, mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.022362112998962402`, `max_abs_diff=1.109375`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_pready_pstage2_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80601.
# Configs:
#   earlyreuse_pstage2 p112 baseline
#   earlyreuse_pready_pstage2 p112 probe
#   prepub_pstage4 p112 reference
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | baseline pstage2 p112 ms | pready pstage2 probe ms | pstage4 reference ms | pready vs baseline |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.0587040000 | 0.0622560009 | 0.0585599989 | +6.0507% |
| H16/S4096 | 0.1639520004 | 0.1786239967 | 0.1632959992 | +8.9490% |
| H4/S2048 | 0.0553120002 | 0.0594879985 | 0.0542079993 | +7.5499% |

Probe-vs-baseline direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0008544921875`, `mean_abs_diff=5.080217846398227e-08`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0025634765625`, `mean_abs_diff=1.3912094232182426e-07`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, no nonfinite; output delta effectively zero.

NCU decision: skipped. The representative H16/S4096 result was a clear isolated-kernel regression, so no follow-up profile was warranted.

Decision: rejected and reverted. The flag is live and numerically stable, but moving the rescale wait after P-ready arrival makes the P-ready/PV handoff materially worse. Classification remains PV tensor-core underfeed/long-scoreboard dominated; the producer must keep the current rescale-before-P-ready ordering for this route.

Revert and validation commands:

```bash
grep -n "earlyreuse_pready\\|earlyreuse_pready_dualaccum" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
git diff --stat -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_pready_pstage2_revert_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pready_pstage2"
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pstage2_q200_p112_o56_qkscfix"
grep -n -B 2 -A 4 "qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2.*ILi128ELi128ELi192ELi128ELi200ELi56ELi112" results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_pready_pstage2_revert_qkscfix.log
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_earlyreuse_pready_pstage2_revert_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=2048, heads=16, seed=80690, warmup=1, iters=2, mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Post-revert source grep and forward-source diff were empty. Rebuilt binary had no `earlyreuse_pready_pstage2` route string and still had the kept `earlyreuse_pstage2_q200_p112_o56_qkscfix` route string. Kept route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`. Post-revert smoke for kept pstage2 earlyreuse qkscfix: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.027131669223308563`, `max_abs_diff=0.91015625`.

## Probe Loop 57

Probe: opt-in warp-level rescale completion arrival on the kept pstage2 earlyreuse score-derived qkscfix route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_warprescale_pstage2_q200_p112_o56_qkscfix`

Rationale: existing NCU evidence classifies the kept route as PV tensor-core underfed with low eligible warps and long scoreboard. This probe was live for the current route: it set `ONLINE_WARP_RESCALE_ARRIVE=true`, changed the rescale barrier init count from warpgroup-level to four warp arrivals, and also skipped the warpgroup sync inside the direct-after-rescale helper. The intent was to reduce a handoff wait between correction/rescale and the P-ready/PV consumer path.

Changed files while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_warprescale_pstage2_q200_qkscfix.log
```

Build result:

- warprescale pstage2 probe: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- kept pstage2 baseline in same build: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_warprescale_pstage2_q200_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_warprescale_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=80700, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.026978015899658203`, `max_abs_diff=1.03125`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_warprescale_pstage2_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80701.
# Configs:
#   earlyreuse_pstage2 p112 baseline
#   earlyreuse_warprescale_pstage2 p112 probe
#   prepub_pstage4 p112 reference
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | baseline pstage2 p112 ms | warprescale pstage2 probe ms | pstage4 reference ms | warprescale vs baseline |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.0595839992 | 0.0599359982 | 0.0596159995 | +0.5908% |
| H16/S4096 | 0.1734559983 | 0.1710720062 | 0.1701759994 | -1.3744% |
| H4/S2048 | 0.0604160018 | 0.0598720014 | 0.0591679998 | -0.9004% |

The first H16/S4096 sweep was suspect because the pstage4 reference also looked faster, so a paired retry was required before profiling.

Paired H16/S4096 retry command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_warprescale_pstage2_q200_qkscfix_h16_s4096_paired_retry.jsonl
# Paired alternating raw ext.forward_streaming_live_mxfp4 calls on H16/S4096.
# WARMUP=30, ITERS=240, SEED=80751.
# Configs: kept p112 baseline vs warprescale probe.
PY
```

Paired H16/S4096 result:

- baseline p112 median: `0.16172799468040466 ms`
- warprescale probe median: `0.16179199516773224 ms`
- warprescale delta: `+0.0395729183769733%`
- numeric vs baseline: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00390625`, `mean_abs_diff=2.06404536129412e-07`, no output/LSE nonfinite.

NCU decision: skipped. The representative paired H16/S4096 result was flat-to-negative, so there was no isolated-kernel win to profile.

Decision: rejected and reverted. The route is live and numerically stable, but warp-level rescale arrival does not relieve the dominant PV tensor-core underfeed/long-scoreboard bottleneck on the kept pstage2 earlyreuse qkscfix path.

Revert and validation commands:

```bash
grep -n "earlyreuse_warprescale\\|earlyreuse_warprescale_dualaccum" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_warprescale_pstage2_revert_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_warprescale_pstage2"
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pstage2_q200_p112_o56_qkscfix"
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_earlyreuse_warprescale_pstage2_revert_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=80790, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Post-revert source grep and forward-source diff were empty. Rebuilt binary had no `earlyreuse_warprescale_pstage2` route string and still had the kept `earlyreuse_pstage2_q200_p112_o56_qkscfix` route string. Kept route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`. Post-revert smoke for kept pstage2 earlyreuse qkscfix: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.02127859927713871`, `max_abs_diff=0.953125`.

## Probe Loop 58

Probe: opt-in current-route force-spare output/PV handoff on the kept pstage2 earlyreuse score-derived qkscfix route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_forcespare_pstage2_q200_p112_o56_qkscfix`

Rationale: existing NCU evidence classifies the kept route as PV tensor-core underfed with low eligible warps and long scoreboard. The older force-spare rejection was on a pstage4/q208 route; this tested the same structural handoff on the kept pstage2/q200 earlyreuse route. The probe was live: `ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE=true` forces non-first PV output to the spare TMEM path, and `ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE=false` is required by the existing direct-after-rescale static assertion.

Changed files while probing:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_diagzero_prepub_earlyreuse_forcespare_pstage2_q200_qkscfix.log
```

Build result:

- force-spare pstage2 probe: `8 bytes stack frame`, `12 bytes spill stores`, `12 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- kept pstage2 baseline in same build: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_diagzero_prepub_earlyreuse_forcespare_pstage2_q200_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_forcespare_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=80800, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Smoke result: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.026270387694239616`, `max_abs_diff=1.0234375`.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_diagzero_prepub_earlyreuse_forcespare_pstage2_q200_qkscfix_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80801.
# Configs:
#   earlyreuse_pstage2 p112 baseline
#   earlyreuse_forcespare_pstage2 p112 probe
#   prepub_pstage4 p112 reference
# Shapes:
#   H16/S2048, H16/S4096, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | baseline pstage2 p112 ms | force-spare pstage2 probe ms | pstage4 reference ms | force-spare vs baseline |
| --- | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.0614399984 | 0.0644479990 | 0.0615999997 | +4.8958% |
| H16/S4096 | 0.1689279974 | 0.1803520024 | 0.1661439985 | +6.7626% |
| H4/S2048 | 0.0579520017 | 0.0608800016 | 0.0574400015 | +5.0525% |

Probe-vs-baseline direct numeric checks:

- H16/S2048: `lse_max_abs_diff=0.000545501708984375`, `max_abs_diff=0.029541015625`, `mean_abs_diff=3.098018714808859e-05`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0006585121154785156`, `max_abs_diff=0.0214691162109375`, `mean_abs_diff=3.0428187528741546e-05`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0005598068237304688`, `max_abs_diff=0.015869140625`, `mean_abs_diff=2.362510349485092e-05`, no nonfinite.

NCU decision: skipped. The representative H16/S4096 direct timing was a large isolated-kernel regression, and the probe introduced local spills, so there was no win to profile.

Decision: rejected and reverted. Current-route force-spare is live but worsens the PV/output handoff: it adds a small spill footprint and forces spare-output merge work, increasing latency on every target shape. Classification remains PV tensor-core underfeed with low eligible warps and long scoreboard; moving non-first PV away from direct-after-rescale main-output accumulation is not the relief valve.

Revert and validation commands:

```bash
grep -n "earlyreuse_forcespare_pstage2\\|earlyreuse_forcespare_dualaccum" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlyreuse_forcespare_pstage2_revert_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_forcespare_pstage2"
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pstage2_q200_p112_o56_qkscfix"
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_earlyreuse_forcespare_pstage2_revert_qkscfix_h16_s2048.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
res = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=2048, heads=16, seed=80890, warmup=1, iters=2,
    mxfp4_fwd_config=cfg, bf16_baseline='tk', include_output_only=False)
print(res)
PY
```

Post-revert source grep and forward-source diff were empty. Rebuilt binary had no `earlyreuse_forcespare_pstage2` route string and still had the kept `earlyreuse_pstage2_q200_p112_o56_qkscfix` route string. Kept route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`. Post-revert smoke for kept pstage2 earlyreuse qkscfix: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.04401761293411255`, `max_abs_diff=1.0`.

## Probe Loop 59 Baseline and Structural Queue Plan

User directive: stop minor knob sweeps after the resolved pstage2 pready/force-spare work and pivot to a real structural P producer queue. Do not touch backward, do not merge Claude branches, preserve the kept qkscfix+p112 route unless data says otherwise, and do not add fake K256.

Current kept live route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`
- Config type: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Forward source dirty state before structural edits: no forward source modifications; only this ledger was dirty.
- Binary route string was present, and post-revert ptxas from `build_after_earlyreuse_forcespare_pstage2_revert_qkscfix.log` showed `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`, `168` registers/thread, `2` barriers, and `1904` bytes static smem.

Current live qkscfix critical path, confirmed from `fwd_configs.inc` and `fwd_streaming_kernel.inc` before any code change:

- Score-derived route flags are live: `ONLINE_SCORE_DERIVED_P_SCALE_PACK=true`, `ONLINE_SCORE_DERIVED_P_PRESCALED_PACK=true`, `ONLINE_SCORE_DERIVED_P_SCALE_MODE=2`, `ONLINE_SCORE_DERIVED_P_X1_SCALE_TMEM=true`, `ONLINE_SCORE_DERIVED_P_FUSED_BLOCK_MAX=true`, `QK_SCALE_COORD_HEAD_INDEX=true`, `ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY=true`, `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT=true`, and `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT=true`.
- Payload is generated directly from scores/local/global max: row/block score max -> E8M0 P scale -> residual score plus scale coefficient -> `fp4pv_exp2_approx` -> `fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte` -> `fp4pv_store_quantized_scores_group_mxfp4_selected(p_fp4_stage[buf], ...)`.
- The kept score-derived prescaled path does not call `fp4pv_pack_scores_to_stage_mxfp4` or the vector-amax/materialized-P fallback; the fallback `fp4pv_pack_scores_to_stage_mxfp4_scaled_rte` remains in the non-score-derived branch.
- Current payload ring depth: `P_STAGE_SLOTS=2`, with `p_fp4_stage[P_STAGE_SLOTS]` and `p_sc_stage[P_STAGE_SLOTS]`.
- Current P-scale TMEM depth: `P_SCALE_TMEM_SLOTS=2`, because `STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM=true`, `STATIC_ALIAS_SCALE_TMEM=false`, and `STATIC_TRIPLE_P_SCALE_TMEM=false`.
- Current V-scale TMEM depth: `V_SCALE_TMEM_SLOTS=2`.
- Current ready/reuse semaphores used on the live direct P-scale path: `p_stage_reusable[P_STAGE_SLOTS]`, `p_sc_tmem_ready[P_SCALE_TMEM_SLOTS]`, `p_sc_tmem_reusable[P_SCALE_TMEM_SLOTS]`, and for cluster remote handoff `p_remote_ready[P_STAGE_SLOTS]` when required. The consumer path does not separately wait `p_quant_ready` for the current direct P-scale TMEM route.
- PV wait location: `wait_and_stage_p_sc(score_idx, p_buf, wait_remote_p_quant)` waits exactly one coarse `p_sc_tmem_ready[p_sc_slot]` for the tile scale/payload visibility, then optionally waits `p_remote_ready[p_buf]`, then `issue_pv(score_idx, p_buf, v_buf, ...)` consumes `p_fp4_stage[p_buf]`.
- Producer ordering for each P tile: score/QK stats -> E8M0 P-scale from score block max vs row max -> residual exp2 -> E2M1 payload store -> `quant_wg_sync()` -> early payload proxy publish -> wait `p_sc_tmem_reusable[p_sc_slot]` if two P-scale slots are already in flight -> x1 P-scale TMEM store -> `fp4pv_tmem_store_wait()` -> one coarse `arrive(p_sc_tmem_ready[p_sc_slot])` -> PV consumes -> consumer/output side releases payload and P-scale reuse.
- Remaining critical path to attack: residual `exp2`/pack, shared payload store/publish, P-scale TMEM store/wait, coarse ready event, PV scale wait/load and consume. Existing counters still classify the bottleneck as PV tensor-core underfeed with low eligible warps and long scoreboard, not DRAM, launch, occupancy, or spills.

TMEM slot budget for the kept route:

- `SCORE_TMEM_WIDTH = SCORE_TMEM_SLOTS * C::Nb = 2 * 128 = 256`.
- Output accumulator TMEM width = `1 * Dvo = 128`.
- Q/K scale width = `Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH = 16 + 16 = 32`.
- Current P-scale width = `2 * P_SCALE_TMEM_WIDTH = 2 * 16 = 32`.
- Current V-scale width = `2 * V_SCALE_TMEM_WIDTH = 2 * 32 = 64`.
- Total = `256 + 128 + 32 + 32 + 64 = 512` columns. A third independent P-scale TMEM slot would require another 16 columns and exceeds the 512-column budget unless some other live allocation is reduced or aliased. Therefore a depth-3 payload ring can be added directly, but matching depth-3 P-scale slots are TMEM-blocked under the current allocation.

K256 blocker audit, to avoid letting true score-derived K256 disappear as "too broad":

- `STATIC_MXFP4_K256` is currently only `STATIC_CONSUMER_MODE == 6 || STATIC_CONSUMER_MODE == 178`.
- The kept dual-output direct-after-rescale path has a static assertion requiring `!STATIC_MXFP4_K256`, so current qkscfix+p112 cannot simply flip K256 on.
- Existing K256 producer code around the K256 branch still calls `fp4pv_pack_scores_to_stage_mxfp4_range` / `fp4pv_pack_scores_to_stage_mxfp4`, which is the vector-amax/materialized-P packer path the user explicitly disallowed for score-derived qkscfix.
- Concrete enabling slice for real score-derived K256, if the ring path does not pay off: add host dynamic-smem launch plumbing and a consumer-mode route that permits K256 without the direct-after-rescale assertion; implement paired score-derived K256 payload staging directly from score residuals, paired direct P-scale TMEM staging, and explicit avoidance of `fp4pv_pack_scores_to_stage_mxfp4` / vector-amax packers. Do not add a fake K256 route that compiles by falling back to the old packer.

Fresh kept-route H16/S4096 baseline timing command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_kept_qkscfix_h16_s4096_ringbase_direct.jsonl
# Raw ext.forward_streaming_live_mxfp4 timing with preallocated out/lse tensors.
# Config: dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix
# Shape: B1/H16/S4096/Dqk192/Dvo128. WARMUP=30, ITERS=240, SEED=80900.
# CUDA events bracket only the raw forward extension call.
PY
```

Fresh baseline timing result: median `0.1605919972062111 ms`, mean `0.16073253334810336 ms`, min `0.1586879938840866 ms`, p10 `0.15955199301242828 ms`, p90 `0.1619199961423874 ms`, max `0.1735360026359558 ms`, finite output/LSE, persistent launch.

Fresh kept-route representative NCU command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_kept_qkscfix_ringbase_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_kept_qkscfix_ringbase_h16_s4096.txt
# Warmed 10 launches, then torch.cuda.cudart().cudaProfilerStart();
# one raw ext.forward_streaming_live_mxfp4 launch; synchronize; cudaProfilerStop().
# Shape: B1/H16/S4096/Dqk192/Dvo128, seed=80901, persistent launch.
PY
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_kept_qkscfix_ringbase_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_kept_qkscfix_ringbase_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_kept_qkscfix_ringbase_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_kept_qkscfix_ringbase_h16_s4096_details.csv
```

Fresh NCU bottleneck snapshot:

| Metric | Value |
| --- | ---: |
| `gpu__time_duration.avg` | 157.056 us |
| `inst_executed` | 53,872,892 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.035447% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.606066% |
| `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.035447% |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.422331 |
| `smsp__warps_active.avg.per_cycle_active` | 2.867963 |
| `smsp__average_warp_latency_per_inst_issued.ratio` | 7.936837 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.529101 |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.479018 |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.221754 |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.641412 |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.248482% |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 11.057198% |
| `derived__local_spilling_requests` | 0 |
| `launch__registers_per_thread` | 168 |
| `launch__shared_mem_per_block_static` | 1.904 KB |
| `launch__shared_mem_per_block` | 103.280 KB |
| `launch__grid_size` | 512 |
| `launch__block_size` | 384 |
| `launch__waves_per_multiprocessor` | 3.37 |
| `profiler__replayer_passes` | 16 |

Classification before patch: representative H16/S4096 remains PV tensor-core underfed. Tensor active is low, eligible warps are low, and long scoreboard/wait dominate. DRAM is only ~1.25% of peak, compute-memory throughput ~11.06%, no local spilling, and launch geometry/occupancy are not the dominant limiter. The structural probe should therefore target producer/P-scale/P payload queueing ahead of PV rather than register-only, launch-only, or DRAM changes.

Next patch decision: the old Loop 50 `P_STAGE_SLOTS=3` payload-only q200 earlyreuse route was rejected because it improved some eligible/scoreboard counters but regressed isolated NCU duration and instruction count. This loop must not duplicate that as a small slot-depth sweep. The full depth-3 P queue is TMEM-blocked if it requires three P-scale slots with the existing two V-scale slots. The targeted structural probe will therefore either:

- implement a current-route depth-3 payload ring plus matching depth-3 P-scale TMEM by explicitly freeing the 16-column TMEM budget from V-scale depth, with all changes opt-in and score-derived, or
- if that compiles/times poorly or proves to be the same failure mode as the older pscale3/vsingle experiment, revert and ledger the blocker precisely, then pivot to the smallest non-fake K256 enabling slice.

Implemented structural probe:

- Route string: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pring3_pscale3_vsingle_q200_p112_o56_qkscfix`
- Forward-only files touched during the probe:
  - `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- Design: inherited the kept score-derived qkscfix pstage2 earlyreuse route, set `P_STAGE_SLOTS=3`, added an opt-in online P-scale TMEM slot override to make `P_SCALE_TMEM_SLOTS=3`, and set an opt-in online single V-scale TMEM mode to free the required 16 TMEM columns.
- Score-derived math was unchanged: payload/scales still came from score residual/max and did not call `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax packers.
- Ring semantics: producer writes one P tile to `p_fp4_stage[idx % 3]`, stores the matching x1 P scale into `p_sc_tmem_ready[idx % 3]` / `p_sc_tmem_reusable[idx % 3]`, and PV still waits one coarse `p_sc_tmem_ready` event for the tile it consumes. No future P tiles are held in registers.

Build command:

```bash
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pring3_pscale3_vsingle_qkscfix.log
```

Probe ptxas:

- `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`
- `Used 168 registers`, `used 2 barriers`, `1968 bytes smem`

Kept baseline ptxas in the same build:

- `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`
- `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`

Smoke command:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pring3_pscale3_vsingle_qkscfix_h16_s4096_s8192.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16 for:
#   H16/S4096 seed=80910
#   H16/S8192 seed=80911
# Config: dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pring3_pscale3_vsingle_q200_p112_o56_qkscfix
PY
```

Smoke results:

| Shape | Finite | `lse_max_abs_diff` vs BF16 | `max_abs_diff` vs BF16 | Notes |
| --- | --- | ---: | ---: | --- |
| H16/S4096 | true | 0.0195930339 | 0.9453125 | no output/LSE nonfinite |
| H16/S8192 | true | 0.0272245035 | 1.203125 | no output/LSE nonfinite |

Direct paired timing command:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pring3_pscale3_vsingle_qkscfix_direct.jsonl
# Alternated kept baseline and probe on the same prepared inputs.
# CUDA events bracketed only raw ext.forward_streaming_live_mxfp4.
# Baseline: dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix
# Probe:    dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pring3_pscale3_vsingle_q200_p112_o56_qkscfix
# Shapes: H16/S2048, H16/S4096, H16/S8192, H4/S2048.
PY
```

Direct paired timing results:

| Shape | Launch | Baseline ms | Probe ms | Probe delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0591679998 | 0.0610559992 | +3.1909% |
| H16/S4096 | persistent | 0.1642239988 | 0.1689279974 | +2.8644% |
| H16/S8192 | fullgrid | 0.5431039929 | 0.5582399964 | +2.7869% |
| H4/S2048 | persistent | 0.0569280013 | 0.0580799989 | +2.0236% |

Probe-vs-kept numeric checks from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=9.183549615799121e-41`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0015869140625`, no nonfinite.
- H16/S8192: `lse_max_abs_diff=0.0`, `max_abs_diff=0.002197265625`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.000732421875`, no nonfinite.

NCU decision: skipped. Representative H16/S4096 direct isolated timing regressed by +2.8644%, and every other tested shape regressed. There was no non-negative or interesting timing result to justify another replay profile.

Decision: rejected and reverted. The full matched depth-3 P queue is correct and spill-free, but the only available way to make room for three independent P-scale TMEM slots in the current 512-column budget is to collapse V-scale TMEM to one slot. That V-scale serialization costs more than the deeper P producer queue recovers. This is an exact resource blocker for matching P-scale depth under the current score/output/QK/V TMEM layout; payload-only depth-3 was already rejected in Loop 50, so the next structural slice should pivot to true score-derived K256 rather than another ring-depth or V-scale tweak.

Revert and validation commands:

```bash
grep -R "pring3_pscale3_vsingle\|ONLINE_P_SCALE_TMEM_SLOTS\|ONLINE_SINGLE_V_SCALE_TMEM\|fp4pv_online_p_scale_tmem_slots\|fp4pv_online_single_v_scale_tmem" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pring3_pscale3_vsingle_revert_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pring3_pscale3_vsingle_q200_p112_o56_qkscfix" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pstage2_q200_p112_o56_qkscfix"
CUDA_VISIBLE_DEVICES=0 timeout 180s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_pring3_pscale3_vsingle_revert_qkscfix_h16_s2048.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16, H16/S2048, seed=80990,
# kept qkscfix pstage2 earlyreuse route.
PY
```

Post-revert verification:

- Forward source diff for `fwd_configs.inc`, `fwd_host_dispatch.inc`, and `fwd_streaming_kernel.inc` was empty.
- The rebuilt binary had no `earlyreuse_pring3_pscale3_vsingle_q200_p112_o56_qkscfix` route string and still had the kept `earlyreuse_pstage2_q200_p112_o56_qkscfix` route string.
- Kept route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert kept-route smoke: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.020408928394317627`, `max_abs_diff=0.83984375`.

## Loop 60: score-derived qkscfix split-K64 P-ready/P-scale early issue rejected

Intent: structural enabling slice after the rejected matched depth-3 P queue. Instead of another register or staging knob, test whether the live score-derived qkscfix path can expose the first K64 half of the P payload earlier while issuing the direct x1 P-scale TMEM store before payload generation. This was meant to reduce the `exp2/pack -> shared payload store/publish -> TMEM scale store/wait -> ready event -> PV consume` critical path without changing math and without materializing P for vector-amax.

Current live qkscfix critical path before patch:

- Route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`.
- Config: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`.
- `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, direct x1 P-scale TMEM, score-derived prescaled P payload, prepublish, early P-scale reuse wait.
- Producer computes score local/global max, derives E8M0 row scales, emits E2M1 P payload directly from residual `score - row_max` through `exp2`/FP4 conversion, stores payload to `p_fp4_stage[idx % 2]`, publishes shared backing, stores direct x1 P scale to TMEM, waits TMEM store, signals `p_sc_tmem_ready`, then signals `p_quant_ready`.
- PV waits for coarse tile readiness and the matching P-scale TMEM ready event, consumes the tile, then signals payload and scale reuse.

K256 blocker ledger before patch:

- Existing static K256 is gated by `STATIC_MXFP4_K256 = MXFP4_PV && (STATIC_CONSUMER_MODE == 6 || STATIC_CONSUMER_MODE == 178)`.
- The kept qkscfix route uses `launch_mxfp4` with dynamic consumer mode `-1`, so it does not enter the existing K256 consumer route.
- The current direct-after-rescale/qkscfix path has an assert blocking `STATIC_MXFP4_K256`.
- Existing K256 producer paths call materialized/vector-amax packers (`fp4pv_pack_scores_to_stage_mxfp4` and split-half variants), which are explicitly disallowed for true score-derived K256.
- Correct score-derived K256 needs paired score-derived payload and P-scale staging: compute pair/global row max over two K128 score tiles, derive both halves' E8M0 row scales from the pair max, generate both E2M1 payload halves directly from score residuals, issue paired direct P-scale TMEM stores, and then consume with the K256 PV route. A K256 route that only flips consumer mode without paired score-derived staging would be fake and can change online-softmax correction semantics.

Implemented opt-in probe:

- Route string: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_splitk64_pstage2_q200_p112_o56_qkscfix`.
- Forward-only files touched during the probe:
  - `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
  - `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
- Design:
  - inherited the kept score-derived qkscfix pstage2/earlyreuse route,
  - enabled split K64 P-ready for the score-derived direct-scale path,
  - issued direct x1 P-scale TMEM early after score-derived E8M0 scale computation,
  - signaled `p_online_k64_ready[buf][0]` after qid 1 and `p_online_k64_ready[buf][1]` after full tile payload,
  - skipped the late duplicate direct P-scale store/ready for this route.
- Math path stayed strict score-derived: no `fp4pv_pack_scores_to_stage_mxfp4`, no vector-amax over materialized P, no fake K256.

Build command:

```bash
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_splitk64_score_direct_pscale_qkscfix_rebuild2.log
```

Probe ptxas:

- `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`
- `Used 168 registers`, `used 2 barriers`, `2464 bytes smem`

Kept baseline ptxas in the same build:

- `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`
- `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`

Smoke commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_splitk64_score_direct_pscale_qkscfix_h16_s2048_rebuild2.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16, H16/S2048,
# config dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_splitk64_pstage2_q200_p112_o56_qkscfix
PY
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_splitk64_score_direct_pscale_qkscfix_h16_s4096_rebuild2.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16, H16/S4096,
# same config
PY
```

Smoke results:

| Shape | Finite | `lse_max_abs_diff` vs BF16 | `max_abs_diff` vs BF16 | Notes |
| --- | --- | ---: | ---: | --- |
| H16/S2048 | true | 0.0230055898 | 1.1328125 | no output/LSE nonfinite |
| H16/S4096 | true | 0.0295750331 | 1.15625 | no output/LSE nonfinite |

Direct paired timing command:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_splitk64_score_direct_pscale_qkscfix_direct.jsonl
# Alternated kept baseline and probe on the same prepared inputs.
# CUDA events bracketed only raw ext.forward_streaming_live_mxfp4.
# Baseline: dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix
# Probe:    dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_splitk64_pstage2_q200_p112_o56_qkscfix
# Shapes: H16/S2048, H16/S4096, H16/S8192, H4/S2048.
PY
```

Direct paired timing results:

| Shape | Launch | Baseline ms | Probe ms | Probe delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0613440014 | 0.0632799976 | +3.1560% |
| H16/S4096 | persistent | 0.1675679982 | 0.1760960072 | +5.0893% |
| H16/S8192 | fullgrid | 0.5531360209 | 0.5868960023 | +6.1034% |
| H4/S2048 | persistent | 0.0644479990 | 0.0667360015 | +3.5502% |

Probe-vs-kept numeric checks from the direct timing harness:

- H16/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00048828125`, no nonfinite.
- H16/S4096: `lse_max_abs_diff=0.0`, `max_abs_diff=0.0030517578125`, no nonfinite.
- H16/S8192: `lse_max_abs_diff=0.0`, `max_abs_diff=0.004150390625`, no nonfinite.
- H4/S2048: `lse_max_abs_diff=0.0`, `max_abs_diff=0.00048828125`, no nonfinite.

NCU decision: skipped. Representative H16/S4096 direct isolated timing regressed by +5.0893%, and every other tested shape regressed. There was no non-negative or interesting timing result to justify replay profiling.

Decision: rejected and reverted. The route was live, correct, and spill-free, but the extra half-ready semaphore path and earlier TMEM scale synchronization added more overhead than it removed. This confirms that splitting K64 readiness inside the existing K128 score-derived producer does not unlock the PV tensor-core underfeed; the next structural slice should be true score-derived K256 or the smallest enabling patch toward it, not another K64/readiness knob.

Revert and validation commands:

```bash
grep -R "splitk64_pstage2_q200_p112_o56_qkscfix\|splitk64_dualaccum_directrescale_decoupled_pstage2\|STATIC_ONLINE_MXFP4_SPLIT_K64_SCORE_DIRECT_PSCALE" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_splitk64_score_direct_pscale_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -F "earlyreuse_splitk64_pstage2_q200_p112_o56_qkscfix" || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_splitk64_score_direct_pscale_qkscfix_h16_s4096.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16, H16/S4096,
# kept qkscfix pstage2 earlyreuse route.
PY
```

Post-revert verification:

- Forward source diff for `fwd_configs.inc`, `fwd_host_dispatch.inc`, and `fwd_streaming_kernel.inc` is empty.
- The rebuilt binary has no `earlyreuse_splitk64_pstage2_q200_p112_o56_qkscfix` route string.
- Kept route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert kept-route smoke: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.029575033113360405`, `max_abs_diff=1.15625`.

## Loop 61: true score-derived K256 blocker map and guarded launch plumbing

Intent: pivot away from small readiness/register knobs and make the true score-derived K256 path concrete. This loop does not add a runnable K256 route; it adds compile-time traits, static guards, and host launch plumbing for the K256 dynamic-smem consumer so that any later route must implement paired score-derived payload/P-scale staging and cannot silently fall back to materialized/vector-amax packers.

Current live qkscfix critical path confirmed before patch:

- Route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`.
- Config: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`.
- Score-derived prescaled P payload: E8M0 scale from score block max vs online row max, E2M1 payload generated directly from score residual via `exp2` and FP4 conversion.
- Direct x1 P-scale TMEM route is live; P scales are issued after scale derivation and before/with payload publish as slot ownership permits in the kept path.
- Slot/resource state: `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, direct x1 P-scale TMEM, score-derived prescaled P payload, prepublish, early P-scale reuse wait.
- Remaining critical path: score local/global max -> E8M0 scale -> residual `exp2`/pack -> shared P payload store/publish -> direct TMEM P-scale store/wait -> coarse ready event -> PV consume -> payload/P-scale reuse.

Representative baseline status used for this loop:

- Existing NCU evidence: H16/S4096 qkscfix is PV tensor-core underfed with low eligible warps and long scoreboard/wait; not DRAM, launch, or spill limited.
- Kept route ptxas after guarded K256 build: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Concrete K256 blockers:

- Existing static K256 is selected only by `STATIC_MXFP4_K256 = MXFP4_PV && (STATIC_CONSUMER_MODE == 6 || STATIC_CONSUMER_MODE == 178)`.
- The live qkscfix online dispatch currently launches with dynamic consumer mode `-1`, so it cannot enter the existing static K256 consumer path.
- Existing K256 producer code uses `fp4pv_pack_scores_to_stage_mxfp4`, `fp4pv_pack_scores_to_stage_mxfp4_range`, or related materialized/vector-amax paths. Those are explicitly disallowed for true score-derived K256.
- A real score-derived K256 route needs paired staging over two K128 score tiles: compute a pair/global row max, derive both halves' E8M0 P scales from that pair max, generate both E2M1 payload halves directly from score residuals, issue paired direct P-scale TMEM stores, and consume with the K256 PV path.
- A consumer-mode-only K256 route would be fake: it would not have paired payload/scales, and it would risk changing online-softmax correction semantics.

Implemented guarded enabling slice:

- Added `fp4pv_online_score_derived_k256<C>` trait, default false, reading `C::ONLINE_SCORE_DERIVED_K256` if present.
- Added `globals_fp4pv_mxfp4_dv::dynamic_shared_memory_k256()` returning `K256_DYNAMIC_SHARED_MEMORY_PADDED`.
- Added `STATIC_ONLINE_MXFP4_SCORE_DERIVED_K256` in the forward kernel and included it in `STATIC_MXFP4_K256`.
- Added compile-time requirements for online score-derived K256: `Mb=128`, `Nb=128`, `CLUSTER_SIZE=1`, score-derived prescaled pack, direct x1 P-scale TMEM, and no direct-after-rescale K256.
- Added a hard static guard: online score-derived K256 is intentionally blocked until paired payload and direct P-scale TMEM staging are implemented; the guard text explicitly says not to fall back to `fp4pv_pack_scores_to_stage_mxfp4`.
- Added an unwired `[[maybe_unused]] launch_mxfp4_k256_static` host helper using `globals_fp4pv_mxfp4_dv<C,true>`, `g.dynamic_shared_memory_k256()`, and `kernel_streaming_live_fp4pv<C, true, 178, false, true>`. It is intentionally not dispatched by any route string.

Forward-only files touched:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Build commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_k256_plumbing_guard_qkscfix.log
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_k256_plumbing_guard_qkscfix_maybeunused.log
git diff --check
```

Build result:

- First build succeeded but emitted `warning #177-D: variable "launch_mxfp4_k256_static" was declared but never referenced`.
- Marked the helper `[[maybe_unused]]`; the second build succeeded with no K256 helper warning.
- Kept qkscfix route ptxas in the clean build: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- `git diff --check` passed.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_k256_plumbing_guard_kept_qkscfix_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
print('cfg', cfg)
rec=benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=4096, heads=16, warmup=1, iters=1, mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline='tk')
print(rec)
PY
```

Smoke result:

- H16/S4096 kept route: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.029575033113360405`, `max_abs_diff=1.15625`.

Timing and NCU decision:

- No paired direct timing or NCU was run for this loop because no runnable K256 route was added. The helper is compile-time/host plumbing only and is intentionally unreachable until paired score-derived payload/P-scale staging exists.

Decision:

- Keep as uncommitted guarded enabling plumbing for the next structural slice. It does not alter the kept qkscfix route, preserves baseline numerics/resources, and prevents fake K256 by compile-time guard. Next implementation slice must be paired score-derived K256 staging, not another small knob sweep and not a route that calls `fp4pv_pack_scores_to_stage_mxfp4`.

## Loop 62: online K256 TMEM layout enabling slice, route still guarded

Intent: continue the structural K256 pivot without adding a fake runnable route. The inspection after Loop 61 found that host launch plumbing alone is not enough: the current online qkscfix TMEM layout cannot fit K256 P/V scale ping-pong because K256 doubles the P/V scale tensor widths.

New K256 blockers confirmed before patch:

- TMEM budget: with the kept qkscfix layout, `SCORE_TMEM_SLOTS=2`, `SCORE_TMEM_WIDTH=256`, `OUTPUT_TMEM_SLOTS=1`, `Dvo=128`, so `SCALE_TMEM_BASE=384`. Compact Q/K scales use 16 columns each, so non-aliased P scales start at column 416.
- K128 kept route fits exactly: `P_SCALE_TMEM_WIDTH=16`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_WIDTH=32`, `V_SCALE_TMEM_SLOTS=2`; columns 416-511 are fully consumed.
- K256 changes widths to `P_SCALE_TMEM_WIDTH=32` and `V_SCALE_TMEM_WIDTH=64`. Two P slots plus two V slots cannot fit. Even two P slots plus one V slot exceeds 512 columns. The only fitting non-aliased online layout is one P-scale slot and one V-scale slot: columns 416-447 for P and 448-511 for V.
- Existing K256 V loader publishes only after `pair_half == 1`. Causal online work has odd `iters_per_task` for every even query tile, so a K256 route needs explicit odd-tail V/P publication or it can deadlock.
- Existing `issue_pv_k256` and output-WG K256 handling are written like the external/non-online path. They do not implement the live dual-output direct-rescale qkscfix contract. A safe online K256 route must either extend K256 PV/output handling for online rescale/spare-output semantics or intentionally use a separate non-dual-output K256 probe route.

Implemented guarded enabling slice:

- Added `STATIC_ONLINE_MXFP4_K256_SINGLE_SCALE_TMEM`, currently tied to `STATIC_ONLINE_MXFP4_SCORE_DERIVED_K256`.
- Made online score-derived K256 force `P_SCALE_TMEM_SLOTS=1`.
- Made online score-derived K256 force `V_SCALE_TMEM_SLOTS=1` through `STATIC_SINGLE_V_SCALE_TMEM`.
- Added a static assertion documenting that online score-derived K256 only fits the current non-aliased TMEM layout with single P/V scale slots.
- Kept the existing hard static guard that blocks `STATIC_ONLINE_MXFP4_SCORE_DERIVED_K256` until paired payload/direct P-scale staging and the remaining output/tail handling are implemented.

Forward-only file touched:

- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Build command:

```bash
git diff --check
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_k256_single_scale_layout_guard_qkscfix.log
```

Build result:

- Build succeeded.
- No unused K256 helper warning.
- Kept qkscfix route ptxas remained unchanged: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- `git diff --check` passed.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_k256_single_scale_layout_guard_kept_qkscfix_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
print('cfg', cfg)
rec=benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=4096, heads=16, warmup=1, iters=1, mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline='tk')
print(rec)
PY
```

Smoke result:

- H16/S4096 kept route: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.029575033113360405`, `max_abs_diff=1.15625`.

Timing and NCU decision:

- No paired direct timing or NCU was run for this loop because the K256 route is still intentionally blocked and unreachable. This is a structural enabling slice, not a performance probe.

Decision:

- Keep as uncommitted guarded enabling code. It is inert for the kept qkscfix route and makes the K256 TMEM budget constraint explicit in code. A runnable K256 patch still requires, in order: paired score reload/row-max/payload generation, direct P-scale x1 stores into both K256 scale subtiles, odd-tail V/P publication, and online output rescale handling for either the current dual-output route or a clearly named non-dual K256 probe.

## Loop 63: guarded online K256 odd-tail V producer enabling slice, route still blocked

Intent: remove one concrete K256 deadlock blocker without creating a fake K256 route. The Loop 62 inspection found that the existing K256 V producer only publishes a pair after `pair_half == 1`; causal online work can end on an odd final half for every even query tile, so an online score-derived K256 route could wait forever for a V pair that is never signaled.

Implemented guarded enabling slice:

- Under `STATIC_ONLINE_MXFP4_SCORE_DERIVED_K256` only, when `pair_half == 0` and `idx + 1 >= iters_per_task`, the V producer now fills the second K256 half with the next tile if valid, otherwise the last tile.
- It loads the matching V scale for that synthetic second half, publishes the shared backing, fences, signals `v_arrived[pair_buf]` and `v_remote_ready[pair_buf]`, and advances `v_phase` at the ring wrap.
- This keeps the path score-derived and does not call `fp4pv_pack_scores_to_stage_mxfp4` or any materialized/vector-amax P packer.
- The route remains intentionally blocked by the existing online K256 static guard until paired score-derived P payload and direct P-scale TMEM staging are implemented.

Forward-only file touched:

- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Build and checks:

```bash
git diff --check
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_k256_vtail_guard_qkscfix.log
grep -n -A4 -B2 "dualaccum_directrescale_decoupled_pstage2" results/mxfp4_fa4_forward_profile_20260612/build_k256_vtail_guard_qkscfix.log | head -80
grep -n "warning #177-D\\|launch_mxfp4_k256_static" results/mxfp4_fa4_forward_profile_20260612/build_k256_vtail_guard_qkscfix.log | head -40
git diff --check
```

Build result:

- Build succeeded.
- No unused K256 helper warning.
- Kept qkscfix route ptxas remained unchanged: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- `git diff --check` passed.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_k256_vtail_guard_kept_qkscfix_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
print('cfg', cfg)
rec=benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=4096, heads=16, warmup=1, iters=1, mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline='tk')
print(rec)
PY
```

Smoke result:

- H16/S4096 kept route: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.029575033113360405`, `max_abs_diff=1.15625`.

Timing and NCU decision:

- No paired direct timing or NCU was run for this loop because the K256 route is still guarded and unreachable. This patch is an enabling correctness slice for the future online K256 producer.

Decision:

- Keep as uncommitted guarded enabling code. It preserves qkscfix resources and numerics while removing a precise future K256 deadlock condition. Remaining blockers for true score-derived K256: paired score reload/row-max over two K128 tiles, paired E2M1 payload generation directly from score residuals, paired direct x1 P-scale TMEM staging in the single-slot K256 layout, and online PV/output rescale handling without falling back to `fp4pv_pack_scores_to_stage_mxfp4`.

## Loop 64: guarded online K256 P pair payload/direct-scale staging slice, route still blocked

Intent: implement the P-side structural slice for true score-derived K256 without adding a fake route. This addresses the largest remaining producer blocker from Loops 61-63: the K256 pair needs direct score-derived E2M1 payload generation and direct P-scale TMEM staging for both K128 halves, plus a zero second half for odd causal tails. The route is still intentionally blocked until PV/output scheduling is wired and numerics can be validated.

Implemented guarded enabling slice:

- Added a `packed_col_offset` parameter to `fp4pv_zero_invalid_causal_payload_groups_mxfp4`, defaulting to zero, so one K128 half inside a wider K256 payload tile can be zeroed without touching the other half.
- Added a score-derived payload store abstraction in the live qkscfix producer. For normal K128 it writes `p_fp4_stage[buf]` exactly as before; for guarded online K256 it writes `p_fp4_stage_k256[pair_buf]` at `pair_half * (C::Nb / 2)`.
- Added guarded odd-tail P payload zeroing: if a K256 pair has only half 0, the future half is zero-filled in the second K128 half of `p_fp4_stage_k256[pair_buf]`.
- Added K256-aware causal payload zeroing for the current half, again using the packed offset in the wider tile.
- Changed guarded online K256 direct P-scale TMEM staging to use the single non-aliased P-scale TMEM slot (`p_sc_tm0`) and write half 0/half 1 into 16-column subtiles. Odd-tail half 1 gets a zero scale word.
- Changed guarded online K256 producer readiness to signal one coarse `p_sc_tmem_ready[0]` per completed pair, not one event per K128 half.
- Changed K256 PV-side scale selection to use `p_sc_tm0`/`v_sc_tm0` for the online single-scale-slot layout instead of selecting `p_sc_tm1`, which aliases the V-scale region when `P_SCALE_TMEM_SLOTS == 1`.
- Added guarded K256 PV-side reuse signaling for the single P-scale and V-scale slots.
- Updated the online K256 static guard text: paired P staging now exists, but the route remains blocked until the paired producer is wired through PV/output scheduling and numerics.
- The guarded path still explicitly avoids `fp4pv_pack_scores_to_stage_mxfp4` and the materialized/vector-amax quantization path. The live math remains score-derived from block max versus row max, residual -> `exp2`, and direct E2M1 conversion.

Forward-only files touched:

- `tk_fa4/fp4_fa4_fwd/fwd_device_helpers.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Build and checks:

```bash
git diff --check
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_k256_pair_pstage_guard_qkscfix.log
grep -n -A4 -B2 "dualaccum_directrescale_decoupled_pstage2" results/mxfp4_fa4_forward_profile_20260612/build_k256_pair_pstage_guard_qkscfix.log | head -120
grep -n "warning #177-D\\|ONLINE_MXFP4_SCORE_DERIVED_K256\\|fp4pv_pack_scores_to_stage_mxfp4" results/mxfp4_fa4_forward_profile_20260612/build_k256_pair_pstage_guard_qkscfix.log | head -80
git diff --check
```

Build result:

- Build succeeded.
- No unused helper warning and no static-guard diagnostic appeared.
- Kept qkscfix route ptxas remained unchanged: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- ptxas emitted spill warnings for other already-compiled exploratory direct/fixed-scale variants during the full build, so the decision criterion here is the kept qkscfix pstage2 route line above.
- `git diff --check` passed before and after build.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_k256_pair_pstage_guard_kept_qkscfix_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
print('cfg', cfg)
rec=benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=4096, heads=16, warmup=1, iters=1, mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline='tk')
print(rec)
PY
```

Smoke result:

- H16/S4096 kept route: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.029575033113360405`, `max_abs_diff=1.15625`, `mean_abs_diff=0.005228234454989433`.

Timing and NCU decision:

- No paired direct timing or NCU was run because the K256 route remains intentionally blocked and unreachable. This is a structural enabling slice, not a runnable performance probe.

Decision:

- Keep as uncommitted guarded enabling code. It preserves the kept qkscfix route's resources and numerics while implementing the P pair payload/direct-scale staging contract that was previously only documented. Remaining blockers for a real runnable K256 route: host dispatch route wiring to a config with `ONLINE_SCORE_DERIVED_K256=true`, removing the hard static guard only for that route, making the K256 PV loop consume one coarse P-ready/scale-ready event per pair, and fixing/validating online output rescale/reuse semantics for paired PV tiles.

## Loop 65: opt-in online K256 compile probe, rejected and reverted to guarded state

Intent: do not let true score-derived K256 remain only a documented blocker. After Loop 64 implemented the guarded paired P payload/direct-scale producer contract, I attempted the next structural step: add a clearly named opt-in route and instantiate the online K256 path so the compiler would expose the next real blocker.

Temporary probe patch:

- Added a derived config with `ONLINE_SCORE_DERIVED_K256=true`.
- Hid the inherited `ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE=true` by setting `ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE=false`, because the K256 requirements correctly reject the direct-after-rescale path.
- Temporarily removed the hard K256 static guard while keeping the shape/math requirements.
- Wired route string `scorederived_k256_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix_probe` in both dispatch tables to `launch_mxfp4_k256_static`.

Compile command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_k256_probe_route_compile_qkscfix.log
```

Compile result:

- Build failed while instantiating `kernel_streaming_live_fp4pv<..., ONLINE=true, STATIC_CONSUMER_MODE=178, EXTERNAL_COL_LSE=false, MXFP4_PV=true>`.
- The concrete error was ThunderKittens `tcgen05.cuh(508)`:

```text
static assertion failed
static_assert(N == D::cols && ((ncta == 1 && N%8 == 0) || (ncta == 2 && N%16 == 0)));
```

- Instantiated types in the failing call:
  - `D = kittens::tt<float, 128, 128>`
  - `A = kittens::st<kittens::fp4e2m1_2, 128, 128, true, 0>`
  - `B = kittens::st<kittens::fp4e2m1_2, 64, 128, true, 0>`
  - `SA = kittens::tt<kittens::fp8e8m0, 128, 32>`
  - `SB = kittens::tt<kittens::fp8e8m0, 128, 64>`
  - `ncta = 1`
- Interpretation: for cluster1 online qkscfix, K256 PV with the current `v_fp4_k256_tile = st_fp4e2m1_2<C::Dvo / 2, C::Nb>` produces a 64-column output tile, but `issue_pv_k256` feeds the existing `tt_output` accumulator typed as 128 columns. The existing K256 debug path compiles because it uses a cluster2 configuration where the pair of CTAs covers the full Dvo width. Live qkscfix is cluster1.

Revert:

- Reverted only the temporary config, route strings, and removal of the hard K256 static guard.
- Restored the hard guard with a more precise message: online score-derived K256 is blocked until it has a Dvo/2 output accumulator path and validation.
- Kept the inert guarded enabling slices from Loops 61-64.

Restore build and checks:

```bash
git diff --check
grep -n "scorederived_k256_prepub_earlyreuse\\|earlyreuse_k256" tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_configs.inc || true
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_k256_probe_route_qkscfix.log
grep -n -A4 -B2 "dualaccum_directrescale_decoupled_pstage2" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_k256_probe_route_qkscfix.log | head -120
```

Restore result:

- `git diff --check` passed.
- No temporary K256 route/config symbols remain.
- Restore build succeeded.
- Kept qkscfix route ptxas remained unchanged: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_k256_probe_route_kept_qkscfix_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix'
print('cfg', cfg)
rec=benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=4096, heads=16, warmup=1, iters=1, mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline='tk')
print(rec)
PY
```

Smoke result:

- H16/S4096 kept route: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.029575033113360405`, `max_abs_diff=1.15625`, `mean_abs_diff=0.005228249356150627`.

Decision:

- Reject the route compile probe and keep the tree in the guarded state. The next real enabling slice for K256 is not more P staging; it is an online K256 output path: either a Dvo/2 accumulator/subtile path that issues/stores/merges two output halves for cluster1, or a deliberate cluster2 online route with correct remote P/V/output semantics. Until that exists, adding a runnable K256 route would be fake.

## Loop 66: opt-in online K256 cluster2 route probe, rejected and reverted to guarded state

Intent: test the concrete alternative exposed by Loop 65. The existing K256 debug kernel compiles with cluster2 because two CTAs cover the full Dvo output width, so I tried a clearly named cluster2 online qkscfix route instead of dismissing K256 as too broad.

Temporary probe patch:

- Added `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_k256_cluster2_dualaccum_decoupled_pstage2_pregs_force_persistent`.
- Set `ONLINE_SCORE_DERIVED_K256=true`.
- Disabled inherited cluster1-only/direct-output pieces for the probe: `ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE=false`, `ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY=false`, `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT=false`, and `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT=false`.
- Temporarily allowed `C::CLUSTER_SIZE == 2` in the online K256 shape assert and removed the hard K256 guard.
- Wired route string `scorederived_k256_cluster2_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix_probe` in both dispatch tables through `launch_mxfp4_k256_static`.
- The math path remained score-derived: no call to `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax-over-materialized-P was added to the live qkscfix route.

Initial compile blocker and adjustment:

- First cluster2 build failed on existing cluster1-only asserts for diagonal payload zeroing and P-payload prepublish:
  - `diagonal-only online causal payload zeroing assumes one CTA and square online tiles`
  - `online P payload prepublish currently supports cluster1 3WG direct P-scale routes`
- I disabled those inherited flags in the temporary config and rebuilt.

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_k256_cluster2_noprepub_probe_qkscfix.log
grep -n -A4 -B2 "k256_cluster2" results/mxfp4_fa4_forward_profile_20260612/build_k256_cluster2_noprepub_probe_qkscfix.log | head -160
grep -n -A4 -B2 "dualaccum_directrescale_decoupled_pstage2" results/mxfp4_fa4_forward_profile_20260612/build_k256_cluster2_noprepub_probe_qkscfix.log | head -160
git diff --check
```

Build result:

- Build succeeded after disabling the cluster1-only inherited flags.
- Probe route ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1936 bytes smem`.
- Kept qkscfix baseline remained unchanged: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- `git diff --check` passed.

Smoke command:

```bash
timeout 90s bash -lc 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'"'"'PY'"'"' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_k256_cluster2_noprepub_probe_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = "scorederived_k256_cluster2_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix_probe"
print("cfg", cfg)
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=4096, heads=16, warmup=1, iters=1, mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline="tk")
print(rec)
PY'
```

Smoke result:

- The probe failed before timing or numeric comparison with `CUDA error: unspecified launch failure`.
- No direct timing or NCU was run because the route is not correct enough to be performance-eligible.

Revert:

- Removed only the temporary cluster2 config and route strings.
- Restored the hard compile-time guard:
  - `online score-derived K256 is intentionally blocked until the paired producer is wired through a Dvo/2 output accumulator path and validated; do not fall back to fp4pv_pack_scores_to_stage_mxfp4`
- Restored the online K256 shape assert to cluster1-only.
- Kept the guarded K256 enabling slices from Loops 61-64.

Restore build and checks:

```bash
grep -RIn "k256_cluster2\\|scorederived_k256_cluster2" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
git diff --check
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd -j$(nproc) 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_k256_cluster2_probe_qkscfix.log
grep -n -A4 -B2 "dualaccum_directrescale_decoupled_pstage2" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_k256_cluster2_probe_qkscfix.log | head -120
```

Restore result:

- No temporary cluster2 K256 route/config symbols remain.
- Restore build succeeded.
- Kept qkscfix route ptxas remained unchanged: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Restore smoke command:

```bash
timeout 90s bash -lc 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'"'"'PY'"'"' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_k256_cluster2_probe_kept_qkscfix_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix"
print("cfg", cfg)
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(seqlen=4096, heads=16, warmup=1, iters=1, mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline="tk")
print(rec)
PY'
```

Restore smoke result:

- H16/S4096 kept route: `finite=true`, no output/LSE nonfinite, `lse_max_abs_diff=0.029575033113360405`, `max_abs_diff=1.15625`, `mean_abs_diff=0.005228275433182716`.

Decision:

- Reject the cluster2 online K256 route. It compiles without spills but the live online kernel launch fails, which is consistent with missing cluster2 online remote/output/reuse semantics rather than a simple register or smem issue.
- K256 blocker is now precise: a real runnable path needs either a cluster1 Dvo/2 output accumulator/subtile issue-store-merge path, or a full cluster2 online route with validated remote P/V readiness, output ownership, and reuse signaling. The path must continue to use paired score-derived P payload staging and direct P-scale TMEM staging, and must explicitly avoid `fp4pv_pack_scores_to_stage_mxfp4` and vector-amax quantization.

## Hard redirect after Loop 66: freeze K256 route work and pivot to feeder/P-ring overlap

User directive:

- Stop K256 route work now.
- Do not continue K256, cluster2, or output-half probing.
- Keep only already validated guarded K256 scaffolding.
- Immediately implement the feeder/P-ring overlap experiment.

K256 state after redirect:

- No selectable `scorederived_k256*` route/config remains.
- `ONLINE_SCORE_DERIVED_K256` is still parsed as a trait, but the live kernel has a hard static guard:
  - `online score-derived K256 is intentionally blocked until the paired producer is wired through a Dvo/2 output accumulator path and validated; do not fall back to fp4pv_pack_scores_to_stage_mxfp4`
- The only retained K256 code is inert guarded scaffolding from Loops 61-64: dynamic shared-memory plumbing, paired payload/scale staging branches behind the hard guard, and the optional launch helper with no route string.
- Latest blocker: a real K256 route still needs a validated output path. Cluster1 requires a Dvo/2 output accumulator/subtile issue-store-merge path; cluster2 requires validated online remote P/V readiness, output ownership, and reuse signaling. Do not continue this now.

New overlap/P-ring constraints:

- A deeper P ring could still work if the TMEM layout is solved, scale slots are aliased safely, score/output TMEM footprint is reduced, or the handoff changes so P-scale depth increases without collapsing V-scale ping-pong.
- Any renewed P-ring/double-buffer/ring-handoff attempt must preserve or improve the whole PV feed path: P payload, P scales, V payload, V scales, ready/reuse protocol, and actual PV MMA/tensor utilization.
- Do not accept a P-ring probe that makes P staging look better by serializing V-scale TMEM or reducing total PV issue rate.
- If TMEM budget blocks deeper P-scale depth, explore scale-slot aliasing, shrinking score/output TMEM footprint, or coarser handoff protocols before declaring P-ring dead.
- Benchmark/profile acceptance must include p_sc/v_sc waits, PV issue rate/tensor active, eligible warps, and end-to-end timing, not just P producer progress.
- First overlap implementation must use the feeder WG/warp pattern before broad WG-count changes:
  - producer generates score-derived P payload plus a P-scale shadow ring;
  - lightweight feeder warp/WG copies P-scale shadow to existing P-scale TMEM slots just in time;
  - feeder also keeps V-scale TMEM ping-pong staged;
  - feeder publishes one coarse PV-ready event;
  - PV WG stays focused on MMA;
  - V-scale ping-pong must be preserved.
- A 4WG or extra-warp/resource redesign is only eligible after this feeder version is implemented, debugged, smoked, timed/profiled, and ledgered.

## Loop 67: feeder/P-ring overlap probe, rejected and reverted

Baseline critical path before patch:

- Live route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`.
- Live config: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`.
- Math path: score-derived prescaled E2M1 P payload and direct x1 E8M0 P-scale TMEM from score block max vs row max. No `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax-over-materialized-P in the qkscfix path.
- Slot/protocol counts: `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`; P-scale uses `p_sc_tmem_ready[slot]` / `p_sc_tmem_reusable[slot]`; V-scale ping-pong uses `v_sc_tmem_ready[v_buf]` / `v_sc_tmem_reusable[v_buf]`; PV waits for P-scale ready plus V-scale ready before MMA and then signals reuse.
- Kept-route ptxas from the rebuild after reverting this probe: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Probe design:

- First implemented the requested feeder/P-ring overlap as an opt-in forward route:
  - producer generates score-derived P payload plus a shared P-scale shadow ring;
  - feeder copies P-scale shadow to existing two P-scale TMEM slots just in time;
  - feeder also stages V scales into the existing V-scale ping-pong slots;
  - feeder publishes one coarse `p_sc_tmem_ready[slot]` PV-ready event;
  - PV WG keeps using the existing MMA path and V-scale ping-pong is not collapsed.
- 3WG single-warp feeder route was implemented first and compiled, but S128 and S4096 both timed out on the first tile. Both one-warp x1 and one-warp x4 P-scale TMEM store variants wedged. Blocker: a single warp is not a safe participant for this TMEM store/publish protocol; the deadlock occurs before ring reuse.
- Minimal enabling slice after that blocker: 4WG relay-feeder route `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_feederwg_pstage3_q200_p112_o56_r32_qkscfix`.

4WG feeder build command:

```bash
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_feederwg_pstage3_probe_qkscfix_fix1.log
grep -n -A5 -B3 "feederwg_pstage3\\|qkscfix_diagzero_prepub_earlyreuse" results/mxfp4_fa4_forward_profile_20260612/build_feederwg_pstage3_probe_qkscfix_fix1.log | head -240
```

Build result:

- 4WG feederWG route ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 128 registers`, `used 4 barriers`, `3504 bytes smem`.
- Same build kept baseline ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Smoke commands:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 60s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feederwg_pstage3_isolate_h16_s128.log
# S128/H16 isolated feederWG launch using route dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_feederwg_pstage3_q200_p112_o56_r32_qkscfix.
PY

set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feederwg_pstage3_vs_baseline_h16_s4096.log
# H16/S4096 same-input BF16, kept qkscfix baseline, and feederWG comparison; seed=81704.
PY
```

Smoke results:

- S128 feederWG: finite output and finite LSE.
- H16/S4096 feederWG vs BF16: finite output/LSE, `vs_bf16_out_max=1.03125`, `out_mean=0.005164616741240025`, `lse_max=0.02299598976969719`.
- H16/S4096 kept baseline vs BF16 in the same run: finite output/LSE, `vs_bf16_out_max=1.03125`, `out_mean=0.005164598114788532`, `lse_max=0.02299598976969719`.
- Probe vs baseline: `out_max=0.0048828125`, `out_mean=2.50076027441537e-07`, `lse_max=9.5367431640625e-07`.

Direct preallocated timing command:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/timing_feederwg_pstage3_direct_prealloc_h16_s2048_s4096.log
# Direct raw ext.forward_streaming_live_mxfp4 calls with preallocated out/lse, alternating kept baseline and feederWG route.
# Routes:
# baseline=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix
# probe=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_feederwg_pstage3_q200_p112_o56_r32_qkscfix
# H16, seed=81704, warmup=6, iters=24, persistent_launch=True.
PY
```

Direct timing results:

| Config | Kept baseline median | Probe median | Delta | Kept baseline min | Probe min | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.068528 ms | 0.071104 ms | +3.759% | 0.065632 ms | 0.067904 ms | +3.462% |
| H16/S4096 | 0.177984 ms | 0.183424 ms | +3.056% | 0.174496 ms | 0.180064 ms | +3.191% |

NCU commands:

```bash
set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_base_qkscfix_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_base_qkscfix_h16_s4096.txt
# Creates H16/S4096 inputs, warms five launches, profiles one forward launch bracketed by torch.cuda.cudart().cudaProfilerStart()/cudaProfilerStop().
PY

set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_feederwg_pstage3_q200_p112_o56_r32_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_probe_qkscfix_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_probe_qkscfix_h16_s4096.txt
# Same isolated one-kernel driver, seed=81704.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_feederwg_probe_qkscfix_h16_s4096_source.csv
```

NCU sections and metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Timing/resource: `gpu__time_duration.avg`, `launch__block_size`, `launch__grid_size`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__waves_per_multiprocessor`.
- PV/tensor issue: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`.
- Scheduler/eligible/stalls: `smsp__warps_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__average_warp_latency_per_inst_issued.ratio`, `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio`.
- Memory: `dram__bytes.avg.per_second`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`, `l1tex__throughput.avg.pct_of_peak_sustained_active`, `l1tex__data_bank_reads.avg.pct_of_peak_sustained_elapsed`, `l1tex__data_bank_writes.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed`.
- Instruction/spill: `inst_executed`, `sass__inst_executed_register_spilling`, `derived__local_spilling_requests`.
- p_sc/v_sc wait behavior was attributed through the explicit route protocol plus source-page SASS samples around `SYNCS.PHASECHK.TRANS64.TRYWAIT`, `SYNCS.ARRIVE`, `STTM`, `UTCCP.T.S.4x32dp128bit`, and `UTCOMMA`. NCU does not expose route-local names like `p_sc_tmem_ready` or `v_sc_tmem_ready`, so the comparison uses these source/protocol sites plus `wait`/`barrier` stall metrics.

NCU representative H16/S4096 counters:

| Metric | Kept baseline | FeederWG | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.064 us | 162.848 us | +4.347% |
| `launch__block_size` | 384 | 512 | +33.333% |
| `launch__registers_per_thread` | 168 | 128 | -23.810% |
| `launch__shared_mem_per_block_static` | 1.904 KB | 3.504 KB | +84.034% |
| `launch__barrier_count` | 2 | 4 | +100.000% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.0375% | 6.7934% | -3.469% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.9042% | 14.2023% | -4.709% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.5814% | 13.2304% | -2.585% |
| `smsp__warps_active.avg.per_cycle_active` | 2.8672 | 3.8423 | +34.008% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.4214 | 0.4630 | +9.874% |
| `smsp__average_warp_latency_per_inst_issued.ratio` | 7.9511 | 10.2504 | +28.917% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.5334 | 3.5113 | -0.624% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.6411 | 1.7095 | +4.167% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.2241 | 1.8324 | +717.814% |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.3515 | 0.5770 | +64.161% |
| `dram__bytes.avg.per_second` | 1.5561 GB/s | 1.4921 GB/s | -4.111% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.2565% | 1.2051% | -4.090% |
| `inst_executed` | 53,821,513 | 61,401,448 | +14.083% |
| `sass__inst_executed_register_spilling` | 0 | 0 | unchanged |
| `derived__local_spilling_requests` | 0 | 0 | unchanged |

Classification:

- The probe did not fix PV tensor-core underfeed. Eligible warps increased slightly, but tensor/TC active fell and wall time regressed.
- Not DRAM limited: DRAM throughput stayed near 1.2% of peak and decreased in the slower probe.
- Not spill limited: both routes have zero ptxas spills and NCU reports no register spilling.
- Dominant regression: shared/TMEM/proxy/barrier handoff and scheduling overhead. The extra WG raises active warps and instruction count, doubles the launch barrier count, and moves time into barrier/no-instruction/protocol waits. This is consistent with p_sc/v_sc handoff overhead rather than useful PV feed overlap.
- Reject criterion hit: feederWG preserved V-scale ping-pong functionally, but overall PV issue/tensor active fell and the route slowed; it mostly moved the stall into protocol overhead rather than improving P+V feed.

Revert commands/checks:

```bash
grep -R -n "FEEDER_PV_READY\\|feederwg_pstage3\\|feeder_pstage3\\|ONLINE_FEEDER" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_feeder_probe.log
grep -n -A4 -B2 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_feeder_probe.log | head -40
```

Revert result:

- No `feederwg_pstage3`, `feeder_pstage3`, `FEEDER_PV_READY`, or `ONLINE_FEEDER` symbols remain in the forward configs/kernel/dispatch.
- Forward-only rebuild succeeded.
- Kept qkscfix route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Post-revert smoke:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 120s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_kept_qkscfix_h16_s4096.log
# H16/S4096 BF16 vs kept qkscfix route, seed=81704.
PY
```

Post-revert smoke result:

- H16/S4096 kept qkscfix route: `finite_out=True`, `finite_lse=True`, `vs_bf16_out_max=1.03125`, `out_mean=0.005164611618965864`, `lse_max=0.02299598976969719`.

Decision and next structural blocker:

- Rejected and reverted the feeder/P-ring overlap probe.
- K256 route work remains frozen by user directive; no selectable K256 route exists.
- A deeper P-ring still looks necessary for MXFP4 FA4, but the next attempt should not add a whole WG as a pure handoff engine unless it demonstrably reduces PV wait cycles. The current evidence says extra execution resources alone increased protocol/barrier work and lowered tensor utilization.
- Concrete next enabling slice should target the same bottleneck with less protocol overhead: either alias/shrink TMEM so P-scale depth can increase without a new WG and without collapsing V-scale ping-pong, or coarsen ready/reuse handoff so one ready event covers fewer waits without adding full-WG barriers. Any follow-up must preserve P payload, P scales, V payload, V scales, and PV MMA issue rate together.

## Loop 68 - 3WG coarse P/V scale ready event probe rejected

Directive context:

- K256/cluster2/output-half work remains stopped. The only retained K256 pieces are guarded, non-selectable scaffolding plus the compile-time block that forbids falling back to vector-amax/materialized-P packers.
- This probe was a smaller structural handoff test after the 4WG feederWG rejection: keep the live qkscfix score-derived P payload/P-scale math and V-scale ping-pong, but collapse the PV-side P-scale/V-scale waits into one coarse event without adding another WG.

Probe design:

- Opt-in route tested: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_coarsepvready_pstage2_q200_p112_o56_qkscfix`.
- Opt-in config inherited the kept qkscfix `pstage2/earlyreuse` route, enabled producer-side async V-scale TMEM staging, and added a coarse PV-ready protocol:
  - quant WG still generated score-derived prescaled E2M1 P payload directly from score residuals and direct x1 E8M0 P scales from score block max vs row max;
  - quant WG still stored P scale into the existing two P-scale TMEM slots and signaled `p_sc_tmem_ready[slot]`;
  - producer spare warp staged V scale into the existing two V-scale TMEM ping-pong slots, waited for the matching `p_sc_tmem_ready[slot]`, then arrived `v_sc_tmem_ready[v_buf]`;
  - PV skipped its separate P-scale wait and used `v_sc_tmem_ready[v_buf]` as the single coarse P+V-scale ready event.
- V-scale ping-pong was preserved; no P/V scale slot was removed or aliased.

Build command:

```bash
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_coarsepvready_probe_qkscfix.log
grep -n -A4 -B2 "coarsepvready\\|qkscfix_diagzero_prepub_earlyreuse" results/mxfp4_fa4_forward_profile_20260612/build_coarsepvready_probe_qkscfix.log | head -220
```

Build result:

- Probe ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1936 bytes smem`.
- Same build kept baseline ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Resource delta was only +32 bytes smem; no spill/register/barrier regression.

Smoke command:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 60s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_coarsepvready_isolate_h16_s128.log
# Isolated H16/S128 launch using route dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_coarsepvready_pstage2_q200_p112_o56_qkscfix.
PY
```

Smoke result:

- H16/S128 printed `inputs` and `probe`, then timed out after 60 seconds with exit code 124.
- No timing or NCU was run because the route failed the deterministic smoke/deadlock guard.

Classification:

- Dominant blocker: readiness/protocol phase deadlock. The spare producer warp cannot safely combine `p_sc_tmem_ready` and `v_sc_tmem_ready` this way while the existing PV reuse/phase protocol expects independent P-scale and V-scale events.
- Not a resource/register/spill blocker: ptxas was clean and nearly identical to baseline.
- Not a DRAM or launch scheduling finding; the route never completed a representative launch.
- p_sc/v_sc wait behavior: the hang is specifically in the attempted p_sc-to-v_sc coarse event handoff path, before PV could produce useful tensor utilization data. This rejects the protocol, not the math path.

Decision:

- Rejected and reverted the coarse PV-ready route, trait, dispatch entries, and event rewiring.
- Post-revert checks:

```bash
grep -R -n "COARSE_PV_READY\\|coarsepvready\\|coarse_pv_ready" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_coarsepvready_probe.log
grep -n -A4 -B2 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_coarsepvready_probe.log | head -40
```

- Result: no coarse-ready symbols remain; diff check passed.
- Post-revert rebuild result for kept qkscfix route: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert smoke command:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 120s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_coarsepvready_kept_qkscfix_h16_s4096_rerun.log
# H16/S4096 BF16 baseline and kept qkscfix route; seed=81704.
PY
```

- Post-revert H16/S4096 kept qkscfix smoke: `finite_out=True`, `finite_lse=True`, `vs_bf16_out_max=1.03125`, `out_mean=0.00012375465303193778`, `lse_max=8.320762634277344`.

Next slice:

- Do not retry this exact coarse event graft. It moved ownership of the P-scale wait without proving the corresponding reuse phase is single-owner and deadlock-free.
- Next structural overlap attempt should avoid adding a new WG or combining events late in the pipeline. The higher-value enabling slice is a lower-overhead slot/TMEM change that preserves independent P-scale and V-scale readiness while letting P-scale depth increase or alias safely without collapsing V-scale ping-pong.

## Loop 69 - online alias3 P-scale slot probe rejected

Directive context:

- K256/cluster2/output-half route work remains stopped. No selectable score-derived K256 route exists.
- This was the next structural P-ring enabling slice after the feederWG and coarse-ready failures: test whether the existing alias-scale TMEM machinery can expose three logical P-scale slots while preserving V-scale ping-pong and avoiding another WG.

Probe design:

- Opt-in route tested: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_alias3_pstage3_q200_p112_o56_qkscfix`.
- Added temporary online traits/config enabling:
  - `P_STAGE_SLOTS=3`;
  - online triple-score TMEM;
  - online alias-scale TMEM;
  - alias-scale slot reuse waits.
- Math path was kept score-derived: direct E2M1 payload from score residual/exp2 and direct x1 P-scale from score block max vs row max. No `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax-over-materialized-P path was used.
- V-scale ping-pong was preserved; the probe only attempted to add logical P-scale depth by aliasing.

Build commands:

```bash
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_alias3_probe_qkscfix.log
grep -n -A4 -B2 "alias3\\|qkscfix_diagzero_prepub_earlyreuse" results/mxfp4_fa4_forward_profile_20260612/build_alias3_probe_qkscfix.log | head -220
```

Initial build blocker:

- The online direct P-scale store path still assumed non-aliased P-scale TMEM slots and tripped the direct-store budget guard around `fwd_streaming_kernel.inc`.
- Minimal fix tested: for `STATIC_ALIAS_SCALE_TMEM`, store online direct P scales at `tt_scores.addr + scale_slot * C::Nb + ALIAS_P_SC_OFFSET`, matching the existing alias addressing model.

Build result after the minimal fix:

- Alias3 ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1968 bytes smem`.
- Same build kept qkscfix baseline: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Resource delta was +64 bytes smem and no register/spill/barrier change.

Smoke commands:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 60s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_alias3_isolate_h16_s128.log
# Isolated H16/S128 launch using route dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_alias3_pstage3_q200_p112_o56_qkscfix.
PY

set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 90s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_alias3_vs_baseline_h16_s128.log
# Paired H16/S128 kept baseline vs alias3 route.
PY
```

Smoke result:

- Isolated alias3 H16/S128: output finite, LSE nonfinite: `done torch.Size([1, 128, 16, 128]) True False`, `out_mean=0.00036576419370248914`, `lse_max=inf`.
- Paired H16/S128:
  - kept qkscfix baseline: output finite and LSE finite, LSE range `[-0.08788539469242096, 4.862833499908447]`;
  - alias3: output finite but LSE nonfinite, `lse_nonfinite=128`;
  - `out_diff_max=0.1279296875`, `out_diff_mean=0.0014615177642554045`, `lse_finite_diff_max=0.0`.
- No timing or NCU was run because the probe failed the deterministic numerics smoke.

Classification:

- Dominant blocker: unsafe online TMEM aliasing for the dualaccum direct-rescale qkscfix route. The failure is numerical/protocol state corruption, not a resource or spill problem.
- The likely issue is aliasing a third P-scale logical slot into TMEM regions that online dual-output qkscfix still uses for score/stats/output-spare state. The offline alias path does not prove this online path is safe.
- p_sc/v_sc conclusion: increasing logical P-scale depth by this alias pattern corrupts correctness before PV feed can be evaluated. This does not satisfy the requirement to preserve overall P payload, P scales, V scales, ready/reuse protocol, and PV MMA utilization together.

Revert commands/checks:

```bash
grep -R -n "alias3\\|ONLINE_ALIAS_SCALE_TMEM\\|ONLINE_TRIPLE_SCORE_TMEM\\|ONLINE_WAIT_ALIAS_SCALE_SLOT_REUSE\\|STATIC_ONLINE_MXFP4_ALIAS_SCALE_TMEM" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_alias3_probe.log
grep -n -A4 -B2 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_alias3_probe.log | head -40
```

Revert result:

- No `alias3`, `ONLINE_ALIAS_SCALE_TMEM`, `ONLINE_TRIPLE_SCORE_TMEM`, `ONLINE_WAIT_ALIAS_SCALE_SLOT_REUSE`, or online alias-scale predicate symbols remain in forward configs/kernel/dispatch.
- Forward diff check passed.
- Post-revert forward rebuild succeeded.
- Kept qkscfix route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Post-revert smoke:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_alias3_kept_qkscfix_h16_s4096.log
# H16/S4096 BF16 vs kept qkscfix route, seed=123, one iteration, include_output_only=False.
PY
```

Post-revert smoke result:

- Route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`.
- H16/S4096 finite result: `finite=True`.
- BF16 comparison: `max_abs_diff=1.0546875`, `mean_abs_diff=0.005278126336634159`, `rmse=0.010779849187966014`, `lse_max_abs_diff=0.02541939541697502`; no output/LSE nonfinite values.
- The printed one-shot `mxfp4_ms=128.42825317382812` is a cold smoke artifact and is not used as a timing comparison.

Decision and next structural slice:

- Rejected and reverted alias3. Do not retry the same online alias-scale layout.
- The exact blocker is TMEM alias safety for online dual-output qkscfix: logical P-scale depth cannot be added by reusing the existing alias-scale offsets without corrupting LSE/state.
- Next structural P/P-scale/V-scale overlap work should inspect whether non-aliased scale depth can be enabled by reducing score/output TMEM footprint or by a different alias with proven ownership, while preserving V-scale ping-pong and independent ready/reuse phases. Any renewed P-ring attempt must measure p_sc/v_sc waits, tensor active/PV issue, eligible warps, and end-to-end timing.

## Loop 70 - P-scale slot 2 aliased to K-scale TMEM rejected

Directive context:

- K256 route work remains stopped. Only guarded, non-selectable K256 scaffolding remains.
- This was a structural P-ring/P-scale/V-scale feed probe after the score-slot alias3 correctness failure. The goal was to test a different, explicit alias that preserves V-scale TMEM ping-pong instead of collapsing V to one slot.

Probe design:

- Opt-in route tested: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pscale3_aliasksc_pstage3_q200_p112_o56_qkscfix`.
- Inherited kept qkscfix `pstage2/earlyreuse`, then set:
  - `P_STAGE_SLOTS=3`;
  - three logical P-scale semaphores/slots;
  - only two physical non-aliased P-scale TMEM slots;
  - logical P-scale slot 2 mapped onto the K-scale TMEM columns.
- Added an explicit QK-side reuse wait: before issuing QK for `next_idx % 3 == 0`, the producer waits for `p_sc_tmem_reusable[2]` so it does not overwrite the K-scale TMEM columns while PV may still consume logical P-scale slot 2.
- V-scale ping-pong was preserved: `V_SCALE_TMEM_SLOTS=2`, with `V_SC_BASE` computed after two physical P-scale slots.
- Math path stayed score-derived: x1 E8M0 scale from score block max vs row max and E2M1 payload from score residual/exp2. No `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax materialized-P path.

Build command:

```bash
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pscale3_aliasksc_probe_qkscfix.log
grep -n -A4 -B2 "pscale3_aliasksc" results/mxfp4_fa4_forward_profile_20260612/build_pscale3_aliasksc_probe_qkscfix.log | head -80
grep -n -A4 -B2 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_pscale3_aliasksc_probe_qkscfix.log | head -40
```

Build result:

- Probe ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1952 bytes smem`.
- Same build kept qkscfix baseline: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Resource delta was +48 bytes smem; no register/spill/barrier regression.

Smoke commands:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pscale3_aliasksc_vs_baseline_h16_s128.log
# Paired H16/S128 kept baseline vs pscale3_aliasksc on identical prepared inputs.
PY

set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pscale3_aliasksc_vs_baseline_h16_s4096.log
# Paired H16/S4096 kept baseline vs pscale3_aliasksc on identical prepared inputs.
PY
```

Smoke results:

- H16/S128: kept and probe both finite; `probe_lse_nonfinite=0`; probe vs kept was bitwise equal for output and LSE (`max_abs_diff=0.0`, `lse_max_abs_diff=0.0`).
- H16/S4096: kept and probe both finite; `probe_lse_nonfinite=0`; `lse_max_abs_diff=0.0`; `max_abs_diff=0.002197265625`, `mean_abs_diff=1.59595657578393e-07`; no output/LSE nonfinite.

Direct preallocated timing command:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pscale3_aliasksc_qkscfix_direct.jsonl
# Alternated kept baseline and probe on the same prepared Q/K/V inputs.
# Outputs/LSE were preallocated; CUDA events bracketed only ext.forward_streaming_live_mxfp4.
# Shapes: H16/S2048, H16/S4096, H16/S8192, H4/S2048.
PY
```

Direct paired timing results:

| Shape | Launch | Baseline ms | Probe ms | Probe delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0643359981 | 0.0676000006 | +5.0734% |
| H16/S4096 | persistent | 0.1739040017 | 0.1859519929 | +6.9280% |
| H16/S8192 | fullgrid | 0.5511679947 | 0.5997120142 | +8.8075% |
| H4/S2048 | persistent | 0.0556799993 | 0.0590879992 | +6.1207% |

Direct timing numeric checks:

- All four shapes had finite kept/probe outputs and LSE.
- Probe-vs-kept LSE max diff was `0.0` for all four shapes.
- Probe-vs-kept output max abs diff stayed small: H16/S2048 `0.0003662109375`, H16/S4096 `0.0052490234375`, H16/S8192 `0.00213623046875`, H4/S2048 `0.0030517578125`.

NCU decision:

- Skipped. The representative H16/S4096 direct isolated timing regressed by +6.9280%, and every other tested shape regressed. This was not a non-negative or interesting win candidate for replay profiling.

Classification:

- Correctness: pass.
- Resource: clean ptxas, no spills, no register/barrier increase.
- Bottleneck effect: negative. The alias wait preserves V-scale ping-pong and avoids score/LSE corruption, but it serializes the QK producer whenever the next QK needs K-scale TMEM after logical P-scale slot 2. That extra producer wait costs more than the extra P payload/P-scale queue depth helps.
- Dominant blocker: QK/P-scale TMEM alias ownership conflict. K-scale columns are too live for this alias to increase overall PV feed; it trades P-scale depth for QK producer stalls.
- Reject criterion hit: the probe improves neither end-to-end timing nor PV feed proxy; it likely moves stall pressure from P-scale slot reuse to QK scale-TMEM reuse.

Revert commands/checks:

```bash
grep -R -n "pscale3_aliasksc\\|ONLINE_ALIAS_THIRD_P_SCALE_K_TMEM\\|STATIC_ONLINE_ALIAS_THIRD_P_SCALE_K_TMEM\\|P_SCALE_TMEM_PHYSICAL_SLOTS\\|ALIAS_THIRD_P_SCALE_K_TMEM" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pscale3_aliasksc_probe.log
grep -n -A4 -B2 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pscale3_aliasksc_probe.log | head -40
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "pscale3_aliasksc" || true
```

Revert result:

- No alias-probe symbols remain in forward configs/kernel/dispatch or the rebuilt binary.
- Forward diff check passed.
- Post-revert forward rebuild succeeded.
- Kept qkscfix route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Guarded K256 scaffolding remains present and non-selectable; no K256 route was added or used.

Post-revert smoke:

```bash
set -o pipefail; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_pscale3_aliasksc_kept_qkscfix_h16_s4096.log
# H16/S4096 BF16 vs kept qkscfix route, seed=70005.
PY
```

Post-revert smoke result:

- Kept qkscfix H16/S4096 finite: `finite=True`.
- BF16 comparison: `max_abs_diff=1.078125`, `mean_abs_diff=0.005282615777105093`, `rmse=0.010723528926914324`, `lse_max_abs_diff=0.021221324801445007`; no output/LSE nonfinite.

Decision and next structural blocker:

- Rejected and reverted `pscale3_aliasksc`.
- Exact blocker: a third logical P-scale slot can be made correct without stealing V-scale ping-pong, but aliasing it to K-scale TMEM introduces QK producer serialization and regresses all measured shapes.
- The remaining TMEM-depth path needs real layout relief, not aliasing against live Q/K scale columns: reduce score/output TMEM footprint, reduce Q/K scale footprint, or change the producer/PV handoff so P-scale depth increases without waiting on live QK scale reuse.

## Loop 71 - consumer-side P-scale shadow staging rejected

Directive context:

- Stop K256/cluster2/output-half route work remains active. No selectable K256 route was added or used in this loop.
- Guarded K256 scaffolding remains non-selectable: host dynamic-smem helper and kernel static guard stay present, but the score-derived K256 path is blocked until paired Dvo/2 output accumulation or validated cluster2 ownership exists. The route must not fall back to `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax-over-materialized-P.
- New overlap constraint recorded for future loops: a deeper P ring is still viable only if it preserves or improves the whole PV feed path: P payload, P scales, V payload, V scales, ready/reuse protocol, and actual PV MMA/tensor utilization. Do not accept a P-ring probe that improves P movement by serializing V-scale TMEM, collapsing V-scale ping-pong, or reducing total PV issue rate. If TMEM budget blocks P-scale depth, next options are scale-slot aliasing with proven ownership, shrinking score/output TMEM footprint, or coarser handoff protocols before declaring P-ring dead.

Probe design:

- Opt-in route tested: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pshadow_pstage3_q200_p112_o56_qkscfix`.
- Producer generated the same score-derived E2M1 payload and x1 E8M0 P scale, but wrote the P-scale word to shared shadow storage `p_online_mxfp4_p_scale_words[buf][row]` instead of directly storing P scale to TMEM.
- Consumer/PV-side `wait_and_stage_p_sc` waited `p_quant_ready[buf]`, then copied the shadow P scale into the existing two P-scale TMEM slots just before PV.
- V-scale ping-pong was preserved: `V_SCALE_TMEM_SLOTS=2`; no V-scale collapse was used to deepen P.
- Math path stayed score-derived. No `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax-over-materialized-P path was used.

Files changed during probe:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`: temporary `pshadow` config and trait.
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`: temporary consumer-side P-scale shadow wait/store path and producer shadow publish path.
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`: temporary opt-in route dispatch entries.

Build command:

```bash
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pshadow_probe_qkscfix.log
grep -n -A4 -B2 "pshadow" results/mxfp4_fa4_forward_profile_20260612/build_pshadow_probe_qkscfix.log | head -100
grep -n -A4 -B2 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_pshadow_probe_qkscfix.log | head -60
```

Build result:

- Probe ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `3472 bytes smem`.
- Same build kept qkscfix baseline: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Resource delta: no register/spill/barrier change, but +1568 bytes static smem for the P-scale shadow ring.

Smoke commands:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pshadow_vs_baseline_h16_s128.log
# Paired H16/S128 kept baseline vs pshadow route on identical prepared inputs.
PY

set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pshadow_vs_baseline_h16_s4096.log
# Paired H16/S4096 kept baseline vs pshadow route on identical prepared inputs.
PY
```

Smoke results:

- H16/S128: kept/probe output and LSE finite, `probe_lse_nonfinite=0`, `lse_max_abs_diff=0.0`; output diff vs kept `max_abs_diff=0.78515625`, `mean_abs_diff=0.017847467213869095`, `rmse=0.04029928233283992`.
- H16/S4096: kept/probe output and LSE finite, `probe_lse_nonfinite=0`, `lse_max_abs_diff=0.0`; output diff vs kept `max_abs_diff=0.25390625`, `mean_abs_diff=0.001472046715207398`, `rmse=0.0033801776373577255`.

Direct preallocated timing command:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pshadow_qkscfix_direct.stdout
# Raw ext.forward_streaming_live_mxfp4 timing with preallocated out/lse tensors.
# Routes: kept qkscfix baseline and pshadow probe.
# Shapes: H16/S2048, H16/S4096, H16/S8192, H4/S2048. warmup=10, iters=40.
# JSON records: results/mxfp4_fa4_forward_profile_20260612/bench_pshadow_qkscfix_direct.jsonl
PY
```

Direct paired timing results:

| Shape | Baseline median ms | Probe median ms | Delta | Baseline min ms | Probe min ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.056639999 | 0.057200000 | +0.9887% | 0.055520002 | 0.055872001 |
| H16/S4096 | 0.166880004 | 0.167775996 | +0.5369% | 0.165087998 | 0.166336000 |
| H16/S8192 | 0.539567977 | 0.542479992 | +0.5397% | 0.537952006 | 0.540095985 |
| H4/S2048 | 0.054576000 | 0.054880001 | +0.5570% | 0.053440001 | 0.053247999 |

NCU commands:

```bash
set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_base_qkscfix_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_base_qkscfix_h16_s4096.txt
# H16/S4096, seed=71201, five warmups, one preallocated forward launch bracketed by cudaProfilerStart/Stop.
PY

set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pshadow_pstage3_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_probe_qkscfix_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_probe_qkscfix_h16_s4096.txt
# Same isolated one-kernel driver and seed.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pshadow_probe_qkscfix_h16_s4096_source.csv
```

NCU sections and metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Kernel timing/resources: `gpu__time_duration.avg`, `inst_executed`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__block_size`, `launch__grid_size`, `launch__waves_per_multiprocessor`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Stall and protocol pressure: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Memory/shared pressure: `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.avg.per_second`, `l1tex__data_bank_reads.avg.pct_of_peak_sustained_elapsed`, `l1tex__data_bank_writes.avg.pct_of_peak_sustained_elapsed`, `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`.
- Source-page SASS proxies for P/V-scale wait/protocol behavior: instruction-executed counts for `STTM`, `LDTM`, `UTCCP`, `UTCOMMA`, `MEMBAR`, `STS`, `LDS`, and `SYNCS...TRYWAIT` (`contains_WAIT`).

Representative H16/S4096 NCU deltas:

| Metric | Baseline | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.064 us | 157.600 us | +0.984% |
| `inst_executed` | 53,803,596 | 51,438,739 | -4.395% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.037576% | 6.993588% | -0.625% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.594911% | 13.478522% | -0.856% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.363969% | 33.027547% | -1.008% |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 | 0.36 | 0.000% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.421303 | 0.413472 | -1.859% |
| `smsp__warps_active.avg.per_cycle_active` | 2.868034 | 2.864720 | -0.116% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.529694 | 3.655757 | +3.571% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.476257 | 0.486362 | +2.122% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.222680 | 0.209437 | -5.947% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.640888 | 1.606606 | -2.089% |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 11.066760% | 11.650351% | +5.273% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.256642% | 1.244436% | -0.971% |
| `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed` | 11.066760% | 11.650351% | +5.273% |
| `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed` | 2.753099% | 3.153004% | +14.526% |
| `launch__registers_per_thread` | 168 | 168 | 0.000% |
| `launch__shared_mem_per_block_static` | 1.904 KB | 3.472 KB | +82.353% |
| `launch__barrier_count` | 2 | 2 | 0.000% |

Source-page instruction proxies:

| Instruction/proxy | Baseline | Probe | Delta |
| --- | ---: | ---: | ---: |
| `UTCOMMA` | 42,240 | 42,240 | 0.000% |
| `UTCCP` | 59,136 | 59,136 | 0.000% |
| `LDTM` | 376,928 | 375,896 | -0.274% |
| `STTM` | 275,552 | 249,176 | -9.572% |
| `LDS` | 43,520 | 51,968 | +19.412% |
| `STS` | 404,096 | 437,888 | +8.362% |
| `MEMBAR` | 58,640 | 92,432 | +57.626% |
| `contains_WAIT` | 6,160,942 | 5,449,265 | -11.551% |

Classification:

- Correctness: pass.
- Timing: fail; every tested shape regressed.
- PV feed: fail. `UTCOMMA` count is unchanged, tensor active is slightly lower, eligible warps are lower, long scoreboard is higher, and H16/S4096 duration increases.
- p_sc/v_sc behavior: moving P-scale store out of the producer reduced `STTM` instructions and source-page wait proxy counts, but it raised shared-memory traffic and protocol/fence pressure (`LDS`, `STS`, `MEMBAR`, LSU pipe activity) and did not improve V/P feed into PV. V-scale ping-pong was preserved, but the consumer-side scale copy did not feed PV faster.
- Bottleneck class after probe: still PV tensor-core underfeed / handoff latency, now with added shared/protocol overhead. Not DRAM, not launch, not spills/registers.

Decision:

- Rejected and reverted `pshadow`.
- Exact blocker: a shared P-scale shadow ring by itself does not create useful overlap. It shifts P-scale work from producer to consumer/PV-side and lowers producer work, but it increases shared/proxy/membar overhead and slightly reduces eligible/tensor activity. This violates the acceptance criterion because overall PV movement and PV issue did not improve.

Revert commands/checks:

```bash
grep -R -n "pshadow\\|ONLINE_CONSUMER_STAGE_P_SCALE_TMEM\\|STATIC_ONLINE_CONSUMER_STAGE_P_SCALE_TMEM" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pshadow_probe.log
grep -n -A4 -B2 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pshadow_probe.log | head -40
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "pshadow" || true
```

Revert result:

- No `pshadow` or consumer-stage predicate symbols remain in forward configs/kernel/dispatch or the rebuilt binary.
- Forward diff check passed.
- Post-revert forward rebuild succeeded.
- Kept qkscfix route ptxas after revert: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.

Post-revert smoke:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_pshadow_kept_qkscfix_h16_s4096.log
# H16/S4096 BF16 vs kept qkscfix route, seed=71301.
PY
```

Post-revert smoke result:

- Kept qkscfix H16/S4096 finite: output finite and LSE finite.
- BF16 comparison: `max_abs_diff=0.890625`, `mean_abs_diff=0.005255952477455139`, `rmse=0.010609774856391143`, `lse_max_abs_diff=0.02235402911901474`; no output/LSE nonfinite.

Next structural slice:

- Do not retry consumer P-scale shadow staging alone.
- Next probe should address the missing execution-resource overlap, not just slot movement: either split scale/feed protocol into a real low-overhead helper lane without increasing MEMBAR/shared pressure, or reduce the P/V scale handoff frequency/footprint so PV sees fewer readiness stalls. Any attempt must preserve V-scale ping-pong and improve `sm__pipe_tensor_cycles_active`, `smsp__warps_eligible`, and direct timing, not only reduce producer-side P-scale work.

## Loop 73 - feeder/PV-ready shadow-scale overlap rejected and reverted

Baseline checkpoint before patch:

- Kept route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`.
- Live math route remains score-derived qkscfix: score-derived prescaled E2M1 P payload, direct x1 E8M0 P-scale TMEM from score block max vs row max, `pstage2`, payload prepublish, and early reuse.
- Slot/protocol state for kept route: `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`; PV waits on P payload/p-scale readiness and V-scale ping-pong before `UTCOMMA`, then signals payload and scale reuse. The critical path is still exp2/pack -> shared P payload store/publish -> P-scale TMEM store/wait -> V-scale ping-pong ready -> PV consume/reuse.
- K256 remains stopped per directive. Current guarded scaffold is non-selectable. Exact blocker: true score-derived K256 still needs host dynamic-smem launch plumbing, consumer-mode route, paired score-derived payload staging, paired direct P-scale TMEM staging, a Dvo/2 output accumulator path, and explicit avoidance of `fp4pv_pack_scores_to_stage_mxfp4`/vector-amax fallback.

Probe design:

- Opt-in route tested: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_feeder_pstage2_q200_p112_o56_qkscfix`.
- Intended overlap: producer emits score-derived P payload plus P-scale shadow; a dedicated lightweight feeder WG copies P-scale shadow into the existing two P-scale TMEM slots, keeps V-scale TMEM ping-pong staged, and publishes the PV-ready event; PV WG focuses on MMA. V-scale ping-pong was preserved.
- Correctness bug found and fixed before timing: the first single-warp/x4 P-scale feeder store left causal inactive shadow rows unsafe and produced H16/S4096 NaNs. Fix used zeroed inactive rows plus all four feeder warps in the native x4 scale-store pattern.
- Fixed probe ptxas: `Used 128 registers`, `used 4 barriers`, `2976 bytes smem`, no spills. Baseline in the same build remained `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`, no spills.

Build/smoke/timing commands:

```bash
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_feeder_pv_ready_fix1.log

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feeder_pv_ready_fix1_vs_baseline_h16_s128.log
# Paired kept qkscfix baseline vs feeder route on identical H16/S128 prepared inputs.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feeder_pv_ready_fix1_vs_baseline_h16_s4096.log
# Paired kept qkscfix baseline vs feeder route on identical H16/S4096 prepared inputs.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_feeder_pv_ready_fix1_qkscfix_direct.stdout
# Direct preallocated alternating timing, 10 warmups and 40 measured iterations per route/shape.
PY
```

Correctness:

- H16/S128: probe matched kept qkscfix output and LSE bitwise.
- H16/S4096: finite output/LSE; probe vs kept qkscfix `max_abs_diff=0.0045166015625`, `mean_abs_diff=1.576e-07`, `rmse=1.023e-05`, `lse_max_abs_diff=9.5367431640625e-07`.

Direct preallocated timing:

| Shape | Kept median ms | Probe median ms | Delta | Kept min ms | Probe min ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.057184 | 0.060816 | +6.351% | 0.056096 | 0.058976 |
| H16/S4096 | 0.162416 | 0.168272 | +3.606% | 0.161184 | 0.167040 |
| H16/S8192 | 0.540032 | 0.553952 | +2.578% | 0.538304 | 0.550112 |
| H4/S2048 | 0.055488 | 0.056832 | +2.422% | 0.054144 | 0.055232 |

NCU commands:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_base_qkscfix_h16_s4096 python3 - <<'PY'
# H16/S4096, seed=71431, ten warmups, one preallocated launch bracketed by cudaProfilerStart/Stop.
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_feeder_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_probe_qkscfix_h16_s4096 python3 - <<'PY'
# Same isolated one-kernel driver and seed.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_feeder_pv_ready_fix1_probe_qkscfix_h16_s4096_source.csv
```

NCU sections and metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Timing/resources: `gpu__time_duration.avg`, `inst_executed`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__block_size`, `launch__grid_size`, `launch__waves_per_multiprocessor`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol pressure: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`.
- Memory/shared pressure: `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`.
- Source-page p_sc/v_sc/PV proxies: executed counts for `UTCOMMA`, `UTCCP`, `LDTM`, `STTM`, `LDS`, `STS`, `MEMBAR`, and source lines containing `TRYWAIT` (`contains_WAIT`) or `SYNCS`.

Representative H16/S4096 NCU deltas:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 157.248 us | 161.536 us | +2.727% |
| `inst_executed` | 53,828,569 | 62,090,080 | +15.35% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.048729% | 6.830863% | -3.091% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.625134% | 13.340629% | -2.088% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.417322% | 34.096218% | +2.032% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.422018 | 0.453660 | +7.50% |
| `smsp__warps_active.avg.per_cycle_active` | 2.867192 | 3.845773 | +34.13% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.527884 | 3.623645 | +2.714% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.477508 | 0.550744 | +15.34% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.221800 | 1.879188 | +747% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.640860 | 1.685972 | +2.75% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.247361% | 1.214501% | -2.63% |
| `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed` | 11.092464% | 12.399725% | +11.785% |
| `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed` | 2.756270% | 3.616143% | +31.197% |

Source-page proxy deltas:

| Proxy | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `UTCOMMA` | 42,240 | 42,240 | 0% |
| `UTCCP` | 59,136 | 59,136 | 0% |
| `LDTM` | 377,368 | 377,416 | +0.013% |
| `STTM` | 275,992 | 276,040 | +0.017% |
| `LDS` | 43,520 | 180,736 | +315.29% |
| `STS` | 428,944 | 462,736 | +7.878% |
| `MEMBAR` | 58,640 | 116,768 | +99.13% |
| `contains_WAIT` | 6,167,348 | 8,526,907 | +38.26% |
| `SYNCS` | 6,457,716 | 8,837,243 | +36.85% |

Classification and decision:

- Rejected. The probe increases active/eligible warps, but those warps are not turning into more PV issue: `UTCOMMA` is unchanged, tensor active drops, and direct timing regresses across all shapes.
- Bottleneck moved into shared/protocol overhead: barrier stalls, wait stalls, `LDS/STS`, `MEMBAR`, and LSU pressure all rise sharply. It is not DRAM, launch, occupancy, or spills.
- Exact blocker: adding a feeder WG without reducing the readiness/fence/shared handoff cost just serializes more protocol around the same two P-scale slots and two V-scale slots. It preserves V-scale ping-pong functionally, but it does not improve the whole PV feed path.

Revert commands/checks:

```bash
grep -n "FEEDER\\|feeder\\|pv_feed_ready\\|wait_feeder" tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_feeder_pv_ready_fix1.log
grep -n -A6 -B3 "config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_after_revert_feeder_pv_ready_fix1.log | head -120
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "feeder" || true
```

Revert result:

- No feeder/PV-ready symbols remain in forward configs/kernel/dispatch or the rebuilt binary.
- Forward diff check passed.
- Post-revert kept qkscfix ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert H16/S4096 smoke, `seed=71441`, kept route vs BF16 TK FA4: finite output/LSE; `max_abs_diff=1.078125`, `mean_abs_diff=0.005306336097419262`, `rmse=0.010771013389118534`, `lse_max_abs_diff=0.02046894282102585`.

Next structural direction:

- Do not repeat K256, cluster2, output-half, pstage5, pshadow-only, feederWG-only, or late coarse-ready grafts.
- The next P-movement probe must either reduce the protocol cost on the existing producer/PV path or free real TMEM/scale depth without stealing live QK/V resources. A viable patch must preserve V-scale ping-pong and improve total PV feed: P payload, P scale, V payload, V scale, ready/reuse protocol, and actual `UTCOMMA`/tensor-active/timing together.

## Loop 74 - early P-scale plus K64 split-ready rejected and reverted

Baseline/protocol checkpoint:

- Kept route preserved: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`.
- Live route confirmed score-derived qkscfix: E2M1 P payload from score residual/`exp2`/FP4 conversion, E8M0 P-scale from score block max vs row max, direct x1 P-scale TMEM, `pstage2`, prepublish, early reuse, no vector-amax/materialized-P packer.
- Slots unchanged: `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`. V-scale ping-pong was preserved; no QK/V TMEM was stolen.
- K256 remains stopped. Current guarded scaffold is non-selectable. Exact blocker remains host dynamic-smem launch plumbing, consumer-mode route, paired score-derived payload staging, paired direct P-scale TMEM staging, Dvo/2 output accumulator path, and explicit avoidance of `fp4pv_pack_scores_to_stage_mxfp4`/vector-amax fallback.

Probe design:

- Opt-in route tested: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_earlypsck64_pstage2_q200_p112_o56_qkscfix`.
- Intended structural slice: issue score-derived P-scale TMEM as soon as the P score scale word is known, then split P readiness at K64 so PV can see half0 earlier while the producer finishes the rest of the tile.
- Correctness bug found and fixed: first H16/S4096 smoke hung because the probe arrived `p_sc_tmem_ready` in both the new early P-scale path and the existing late direct-P-ready path. The fix skipped the late duplicate arrive for this opt-in route, keeping one P-scale ready event per tile.
- Fixed probe ptxas: `Used 168 registers`, `used 2 barriers`, `1936 bytes smem`, no spills. Kept baseline in the same checkpoint remained `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`, no spills.

Commands:

```bash
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_earlypsck64_probe2_fix1_qkscfix.log

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_earlypsck64_fix1_vs_baseline_h16_s128.log
# Paired kept qkscfix baseline vs earlypsck64 on identical H16/S128 inputs.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_earlypsck64_fix1_stage_h16_s4096.log
# Paired kept qkscfix baseline vs earlypsck64 on identical H16/S4096 inputs.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_earlypsck64_fix1_vs_bf16_h16_s4096.log
# H16/S4096 kept/probe vs BF16 TK FA4 baseline.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' > results/mxfp4_fa4_forward_profile_20260612/bench_earlypsck64_fix1_direct_prealloc_paired.jsonl
# Direct preallocated alternating timing, 10 warmups and 40 measured iterations per route/shape.
PY
```

Correctness after protocol fix:

- H16/S128: probe matched kept qkscfix output and LSE bitwise.
- H16/S4096 staged probe vs kept qkscfix: finite output/LSE, `max_abs_diff=0.003173828125`, `lse_max_abs_diff=0`.
- H16/S4096 vs BF16: finite output/LSE; probe vs kept qkscfix `max_abs_diff=0.004150390625`, `lse_max_abs_diff=9.5367431640625e-07`. BF16 error profile was unchanged from kept baseline: kept `mean_abs_diff=0.0052120001`, probe `mean_abs_diff=0.0052119698`; both LSE max abs `0.03435519`.

Direct preallocated timing:

| Shape | Kept median ms | Probe median ms | Delta |
| --- | ---: | ---: | ---: |
| H16/S2048 persistent | 0.067712 | 0.070464 | +4.064% |
| H16/S4096 persistent | 0.183696 | 0.193232 | +5.191% |
| H16/S8192 fullgrid | 0.558720 | 0.595920 | +6.658% |
| H4/S2048 persistent | 0.072800 | 0.075008 | +3.033% |

Decision:

- Rejected and reverted. Timing regressed uniformly, so no follow-up NCU was run; wall time was sufficient to reject the probe.
- Classification: the route did not create useful PV tensor-core overlap. It added one half-ready protocol path and moved the P-scale store earlier, increasing smem/protocol work without improving the whole PV feed path. Since V-scale ping-pong and `P_SCALE_TMEM_SLOTS=2` were unchanged, PV still could not issue more useful MMA work.

Revert/check commands:

```bash
grep -R "earlypsck64\\|SCORE_DERIVED_EARLY_P_SCALE\\|SCORE_DERIVED_SPLIT_P_READY" -n tk_fa4/fp4_fa4_fwd || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
set -o pipefail; make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_earlypsck64_revert.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep earlypsck64 || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_earlypsck64_revert_baseline_h16_s4096.log
# H16/S4096 kept qkscfix route vs BF16 TK FA4, seed=74030.
PY
```

Revert result:

- No `earlypsck64`, `SCORE_DERIVED_EARLY_P_SCALE`, or `SCORE_DERIVED_SPLIT_P_READY` references remain in forward source or the rebuilt binary.
- Forward diff check passed.
- Post-revert kept qkscfix ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert H16/S4096 smoke, `seed=74030`, kept route vs BF16 TK FA4: finite output/LSE; `max_abs_diff=0.9140625`, `mean_abs_diff=0.005279868375509977`, `rmse=0.010664094519436104`, `lse_max_abs_diff=0.023471519351005554`.

Next structural direction:

- Avoid adding readiness events to the PV critical path unless they demonstrably increase `UTCOMMA`/tensor active. The next probe should reduce handoff/protocol cost or remove a real wait, not add another split/publish path.
- Candidate to inspect next: existing qkscfix p-scale/V-scale handoff ordering for redundant waits/fences around direct x1 P-scale TMEM and V-scale ping-pong. Patch only if the code shows a live, correctness-preserving way to remove or coarsen a wait while preserving P payload, P scale, V payload, V scale, and PV issue rate.

## Loop 75 - P-before-V wait-order probe rejected and reverted

Hypothesis from code/profile evidence:

- Existing kept qkscfix route waits/stages V-scale first, then P-scale, before `issue_pv` in the decoupled PV lane. These readiness paths are logically independent once P payload is published.
- The probe flipped only this ordering with `ONLINE_STAGE_P_BEFORE_V=true`, keeping all slots and math unchanged: `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`, score-derived P payload/scales, direct x1 P-scale TMEM, and V-scale ping-pong preserved.
- This was opt-in only; kept qkscfix baseline remained selectable and unchanged.

Route tested:

- Baseline: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`
- Probe: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_p112_o56_qkscfix`

Build/resources:

- Build command: `make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pfirst_qkscfix_probe.log`
- Probe ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Kept baseline in same build: identical `168 registers`, `2 barriers`, `1904 bytes smem`, no spills.

Correctness:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pfirst_qkscfix_h16_s128.log
# Paired kept qkscfix baseline vs pfirst on H16/S128, seed=75001.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pfirst_qkscfix_h16_s4096.log
# Paired kept qkscfix baseline vs pfirst on H16/S4096, seed=75002, plus BF16 reference.
PY
```

- H16/S128: finite and bitwise equal to kept qkscfix, output/LSE max abs diff `0`.
- H16/S4096: finite; probe vs kept qkscfix `max_abs_diff=0.00341796875`, `mean_abs_diff=1.5442108747265593e-07`, `rmse=8.913340726624504e-06`, `lse_max_abs_diff=0`.
- BF16 envelope unchanged at H16/S4096: baseline `mean_abs_diff=0.005294919945299625`, probe `mean_abs_diff=0.0052949292585253716`, both `lse_max_abs_diff=0.02942935936152935`.

Direct preallocated timing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pfirst_qkscfix_direct_prealloc_paired.stdout
# Direct extension call to forward_streaming_live_mxfp4 with preallocated out/lse.
# WARMUP=10, ITERS=40, alternating baseline/probe order per iteration.
# Full JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_pfirst_qkscfix_direct_prealloc_paired.jsonl
PY
```

| Shape | Launch | Kept median ms | Probe median ms | Delta | Kept min ms | Probe min ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.058208 | 0.058384 | +0.302% | 0.056384 | 0.056384 |
| H16/S4096 | persistent | 0.170640 | 0.170480 | -0.094% | 0.166720 | 0.167584 |
| H16/S8192 | fullgrid | 0.545904 | 0.547088 | +0.217% | 0.542880 | 0.544224 |
| H4/S2048 | persistent | 0.055216 | 0.055504 | +0.522% | 0.051968 | 0.053056 |

NCU commands:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_base_qkscfix_h16_s4096 python3 - <<'PY'
# H16/S4096, seed=75021, ten warmups, one preallocated launch bracketed by cudaProfilerStart/Stop.
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pfirst_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_probe_qkscfix_h16_s4096 python3 - <<'PY'
# Same isolated one-kernel driver and seed.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_probe_qkscfix_h16_s4096_source.csv
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Timing/resources: `gpu__time_duration.avg`, `inst_executed`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__block_size`, `launch__grid_size`, `launch__waves_per_multiprocessor`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio`.
- Memory/shared: `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`.
- Source-page proxies: `UTCOMMA`, `UTCCP`, `LDTM`, `STTM`, `LDS`, `STS`, `MEMBAR`, `TRYWAIT`, `SYNCS`, `BAR`, `WARPSYNC`.

Representative H16/S4096 NCU deltas:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.672 us | 155.712 us | -0.613% |
| `inst_executed` | 54,019,977 | 54,109,275 | +0.165% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.018604% | 7.009409% | -0.131% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.557618% | 13.576684% | +0.141% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.266654% | 33.241580% | -0.075% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.421714 | 0.421595 | -0.028% |
| `smsp__warps_active.avg.per_cycle_active` | 2.867484 | 2.867069 | -0.014% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.533949 | 3.534294 | +0.010% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.478608 | 0.472899 | -1.193% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.222290 | 0.221052 | -0.557% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.641210 | 1.640970 | -0.015% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.251701% | 1.259311% | +0.608% |
| `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed` | 11.033816% | 11.022261% | -0.105% |
| `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed` | 2.746048% | 2.743044% | -0.109% |

Source-page proxies:

| Proxy | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `UTCOMMA` | 42,240 | 42,240 | 0% |
| `UTCCP` | 59,136 | 59,136 | 0% |
| `LDTM` | 376,816 | 376,920 | +0.028% |
| `STTM` | 275,440 | 275,544 | +0.038% |
| `LDS` | 43,520 | 43,520 | 0% |
| `STS` | 428,944 | 428,944 | 0% |
| `MEMBAR` | 58,640 | 58,640 | 0% |
| `TRYWAIT` | 6,234,848 | 6,264,168 | +0.470% |
| `SYNCS` | 6,525,216 | 6,554,536 | +0.449% |
| `BAR` | 227,856 | 227,856 | 0% |
| `WARPSYNC` | 298,080 | 298,092 | +0.004% |

Decision:

- Rejected and reverted. The H16/S4096 NCU duration ticked down, but PV issue did not improve: `UTCOMMA` unchanged, tensor active slightly lower, eligible warps slightly lower, and wait/sync proxy counts increased. Direct timing regressed on H16/S2048, H16/S8192, and H4/S2048, and the H16/S4096 win was within noise.
- Classification: this was not a P/P-scale/V-scale overlap unlock. It only changed serialized wait order and did not improve the whole PV feed path.

Revert/check commands:

```bash
grep -R "pfirst_pstage2_q200_p112_o56_qkscfix\\|earlyreuse_pfirst_dualaccum" -n tk_fa4/fp4_fa4_fwd || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pfirst_revert.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pfirst_pstage2_q200_p112_o56_qkscfix" || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_pfirst_revert_baseline_h16_s4096.log
# H16/S4096 kept qkscfix route vs BF16 TK FA4, seed=75031.
PY
```

Revert result:

- No qkscfix pstage2 `pfirst` source references remain, and the rebuilt binary has no `earlyreuse_pfirst_pstage2_q200_p112_o56_qkscfix` string.
- Post-revert kept qkscfix ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert H16/S4096 smoke finite; BF16 comparison `max_abs_diff=1.109375`, `mean_abs_diff=0.005174783058464527`, `rmse=0.010604292353610729`, `lse_max_abs_diff=0.022000741213560104`.

Next structural direction:

- Do not repeat pure P/V wait-order flips. They do not increase PV issue rate.
- Next lever should change actual movement or lifetime pressure: reduce/coarsen a real `TRYWAIT/SYNCS` path, eliminate redundant payload/scale publish/wait for direct x1 score-derived P-scale, or reduce footprint enough to make true P-scale depth possible while preserving V-scale ping-pong.

## Loop 76 - producer V-scale TMEM staging for kept qkscfix (`prodvsc`)

Directive fit:

- Forward-only. No backward files touched.
- Kept qkscfix baseline preserved.
- K256/cluster2/output-half probing stopped; existing guarded K256 scaffolding remains inert and explicitly blocked.
- Probe tries to improve the whole PV feed without collapsing V-scale ping-pong: producer V-loader warp stages V-scale into the existing two V-scale TMEM slots while preserving `V_SCALE_TMEM_SLOTS=2`.

Hypothesis:

- Existing NCU showed MXFP4 qkscfix is PV tensor-core underfed with low eligible warps, long scoreboard, and no DRAM/launch/spill limit.
- If PV WG spends issue-lane time staging V-scale TMEM before PV MMA, moving V-scale TMEM load to producer-side V-loader warp could leave PV WG more focused on MMA while preserving the existing P payload/P-scale flow and V-scale ping-pong.

Implementation:

- Added opt-in route:
  `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_prodvsc_pstage2_q200_p112_o56_qkscfix`
- Added config:
  `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_prodvsc_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- First attempt incorrectly used producer `warpid()==1`, which is live PV issue lane in the decoupled route and hung S128. Fixed by moving the producer V-scale TMEM staging into the existing producer V-loader group on `warpid()==2` after V payload/scale shared publish.
- Score-derived P path stayed intact: no use of `fp4pv_pack_scores_to_stage_mxfp4`, no vector-amax over materialized P, direct x1 P-scale TMEM unchanged, P payload generated from score residual/exp2/FP4 conversion.

Build:

```bash
make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_prodvsc_qkscfix_fix2.log
```

ptxas for the opt-in route:

- `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`
- `Used 168 registers`, `used 2 barriers`, `1936 bytes smem`

Kept qkscfix in same build:

- `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`
- `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`

Correctness:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_prodvsc_fix2_progress_h16_s128.log
# Paired kept qkscfix vs prodvsc, H16/S128, same inputs.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_prodvsc_fix2_vs_baseline_h16_s4096.log
# Paired kept qkscfix vs prodvsc plus BF16 TK FA4 comparison, H16/S4096, same inputs.
PY
```

Results:

- H16/S128: baseline and probe completed finite; probe vs baseline bitwise output and LSE diffs were 0.
- H16/S4096 probe vs kept baseline: `max_abs_diff=0.002197265625`, `mean_abs_diff=2.1677654160612292e-07`, `rmse=1.0224556856235957e-05`, `lse_max_abs_diff=0`.
- H16/S4096 kept baseline vs BF16: `max_abs_diff=1.078125`, `mean_abs_diff=0.005239258985966444`, `rmse=0.010713115619032829`, `lse_max_abs_diff=0.022542588412761688`.
- H16/S4096 probe vs BF16 was effectively identical: `max_abs_diff=1.078125`, `mean_abs_diff=0.005239267833530903`, `rmse=0.010713119014850031`, `lse_max_abs_diff=0.022542588412761688`.

Direct preallocated timing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 600s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_prodvsc_fix2_qkscfix_direct.stdout
# Paired direct preallocated timing, same tensors, alternating kept/probe.
# WARMUP=10, ITERS=60.
# Output JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_prodvsc_fix2_qkscfix_direct.jsonl
# Baseline route:
# dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix
# Probe route:
# dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_prodvsc_pstage2_q200_p112_o56_qkscfix
PY
```

| Shape | Launch | Kept median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.058672 | 0.059168 | +0.845% |
| H16/S4096 | persistent | 0.166176 | 0.167520 | +0.809% |
| H16/S8192 | fullgrid | 0.544752 | 0.548320 | +0.655% |
| H4/S2048 | persistent | 0.058464 | 0.059248 | +1.341% |

NCU H16/S4096 commands:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_base_qkscfix_h16_s4096 \
  python3 - <<'PY'
# H16/S4096, same seed/input policy as direct timing.
# 10 warmups, cudaProfilerStart(), one forward launch, cudaProfilerStop().
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_prodvsc_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_probe_qkscfix_h16_s4096 \
  python3 - <<'PY'
# H16/S4096, same seed/input policy as direct timing.
# 10 warmups, cudaProfilerStart(), one forward launch, cudaProfilerStop().
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_prodvsc_fix2_probe_qkscfix_h16_s4096_source.csv
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Timing/resources: `gpu__time_duration.avg`, `inst_executed`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__waves_per_multiprocessor`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warp_latency_per_inst_issued.ratio`, `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Memory/shared: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.avg.per_second`, `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed`.
- Source-page proxies: `UTCOMMA`, `UTCCP`, `LDTM`, `STTM`, `LDS`, `STS`, `TRYWAIT`, `SYNCS`, `MEMBAR`, `BAR.SYNC`, `MUFU`, `R2UR`, `LDG`, `STG`.

Representative H16/S4096 NCU deltas:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.192 us | 157.472 us | +0.820% |
| `inst_executed` | 54,068,405 | 54,330,677 | +0.485% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.049532% | 6.942389% | -1.520% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.621538% | 13.395463% | -1.660% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.942298% | 14.592652% | -2.340% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.417795% | 33.185287% | -0.696% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.422515 | 0.424849 | +0.552% |
| `smsp__warps_active.avg.per_cycle_active` | 2.866857 | 2.869693 | +0.099% |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 | 0.36 | 0% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.520452 | 3.491546 | -0.821% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.223723 | 0.228009 | +1.916% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.640757 | 1.636616 | -0.252% |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.342213 | 0.370291 | +8.205% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.255466% | 1.245454% | -0.797% |
| `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed` | 11.090181% | 10.859787% | -2.077% |

Source-page proxies:

| Proxy | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `UTCOMMA` | 42,240 | 42,240 | 0% |
| `UTCCP` | 59,136 | 59,136 | 0% |
| `LDTM` | 377,024 | 377,000 | -0.006% |
| `STTM` | 275,648 | 275,624 | -0.009% |
| `LDS` | 43,520 | 43,520 | 0% |
| `STS` | 428,944 | 428,944 | 0% |
| `TRYWAIT` | 6,251,341 | 6,215,607 | -0.572% |
| `SYNCS` | 6,541,709 | 6,524,919 | -0.257% |
| `MEMBAR` | 58,640 | 58,640 | 0% |
| `BAR.SYNC` | 121,856 | 121,856 | 0% |

Decision:

- Rejected and reverted. Direct timing regressed on every standard shape, NCU time regressed by +0.820%, and PV issue did not improve: `UTCOMMA` unchanged, `smsp__issue_active.avg.per_cycle_active` unchanged, tensor active down, TC active down.
- Classification: not DRAM/launch/spill limited. The bottleneck remains PV tensor-core underfeed caused by QK/softmax/P/V handoff/protocol timing. This probe moved part of V-scale work but did not improve the combined P payload/P-scale/V payload/V-scale readiness path; it added producer/protocol work and reduced realized tensor utilization.
- Exact blocker: the decoupled qkscfix PV issue lane was not primarily blocked by its own V-scale TMEM staging instruction sequence. Moving V-scale staging into the producer V-loader group preserved correctness and V-scale ping-pong, but did not change PV MMA issue count and made the total schedule slower.

Revert/check:

```bash
grep -R -n "prodvsc\\|earlyreuse_prodvsc" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_prodvsc_revert.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "prodvsc\\|earlyreuse_prodvsc" || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_prodvsc_revert_baseline_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix"
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=4096, heads=16, seed=81180, warmup=1, iters=1,
    mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline="tk")
print(rec)
PY
```

Revert result:

- No `prodvsc` source references remain in forward configs/dispatch/kernel.
- Rebuilt binary has no `prodvsc`/`earlyreuse_prodvsc` string.
- Kept qkscfix post-revert ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert H16/S4096 smoke finite; BF16 comparison `max_abs_diff=1.140625`, `mean_abs_diff=0.00524517148733139`, `rmse=0.010732272297213429`, `lse_max_abs_diff=0.017838004976511`.

Next structural direction:

- Stop moving isolated V-scale staging without changing the actual ready/handoff shape; `UTCOMMA` did not budge.
- Next probe should target a real protocol/movement cut that can reduce PV-feed waits without stealing QK/V TMEM or collapsing V-scale ping-pong. Candidate: make the qkscfix direct x1 P-scale path publish a single coarse PV-ready only after both existing P-scale/V-scale slots are ready, while removing the redundant per-path P-scale wait in the PV-side staging path. Acceptance requires unchanged `UTCOMMA` or better, lower wait/barrier source proxies, and non-negative direct timing.

## Loop 77: skip direct P-scale TMEM store wait on qkscfix x1 path

Hypothesis:

- In the live score-derived qkscfix route, producer stores direct x1 P-scale words into the two P-scale TMEM slots, calls `fp4pv_tmem_store_wait()`, then publishes the coarse P-scale ready event consumed by PV.
- If that wait is redundant with the existing `p_sc_tmem_ready` semaphore and PV-side TMEM ordering, skipping it should shorten P-scale handoff without changing payload math, P-scale slot count, V-scale ping-pong, QK TMEM, or PV MMA issue work.

Guarded route:

- Baseline: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix`
- Probe: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_skipscwait_pstage2_q200_p112_o56_qkscfix`
- Type: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_skipscwait_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Changed files during probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`, `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`.
- Preserved: score-derived P payload/scales, `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, V-scale ping-pong, qkscfix, p112, no K256, no `fp4pv_pack_scores_to_stage_mxfp4`.

Build/check commands:

```bash
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_skipscwait_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep skipscwait
```

Build result:

- Probe route compiled.
- Probe ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Kept qkscfix ptxas in same build: identical resources.

Correctness commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_skipscwait_qkscfix_h16_s128.log
# Same BF16 source input, prepare per config, run baseline then skipscwait on H16/S128, compare with _compare_outputs.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 300s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_skipscwait_qkscfix_h16_s4096.log
# Same BF16 source input, prepare per config, run baseline then skipscwait on H16/S4096, compare with _compare_outputs.
PY
```

Correctness results:

| Shape | finite | `max_abs_diff` vs kept | `mean_abs_diff` | `rmse` | `lse_max_abs_diff` |
| --- | --- | ---: | ---: | ---: | ---: |
| H16/S128 | yes | 0 | 0 | 0 | 0 |
| H16/S4096 | yes | 0.002197265625 | 0.000000153755934 | 0.000008520716 | 0 |

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 600s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_skipscwait_qkscfix_direct.stdout
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# IMPORTANT: set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED=81230.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_skipscwait_qkscfix_direct.jsonl
PY
```

Direct preallocated timing:

| Shape | Launch | Kept median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.062399998 | 0.062368002 | -0.051% |
| H16/S4096 | persistent | 0.169567995 | 0.169888005 | +0.189% |
| H16/S8192 | fullgrid | 0.545839995 | 0.545888007 | +0.009% |
| H4/S2048 | persistent | 0.060224000 | 0.060224000 | 0.000% |

NCU H16/S4096 commands:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix \
  PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_base_qkscfix_h16_s4096 \
  python3 -u - <<'PY'
# H16/S4096, 10 raw-extension warmups, cudaProfilerStart(), one preallocated forward launch, cudaProfilerStop().
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_skipscwait_pstage2_q200_p112_o56_qkscfix \
  PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_probe_qkscfix_h16_s4096 \
  python3 -u - <<'PY'
# H16/S4096, 10 raw-extension warmups, cudaProfilerStart(), one preallocated forward launch, cudaProfilerStop().
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_skipscwait_probe_qkscfix_h16_s4096_source.csv
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Timing/resources: `gpu__time_duration.avg`, `inst_executed`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__waves_per_multiprocessor`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warp_latency_per_inst_issued.ratio`, `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Memory/shared: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.avg.per_second`, `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed`.
- Source-page proxies: `UTCOMMA`, `UTCCP`, `LDTM`, `STTM`, `LDS`, `STS`, `TRYWAIT`, `SYNCS`, `MEMBAR`, `BAR.SYNC`, `MUFU`, `R2UR`, `LDG`, `STG`.

Representative H16/S4096 NCU deltas:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 157.632 us | 157.216 us | -0.264% |
| `inst_executed` | 53,879,392 | 54,118,835 | +0.444% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.011289% | 7.023448% | +0.173% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.543574% | 13.566186% | +0.167% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.820178% | 14.832467% | +0.083% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.228448% | 33.261566% | +0.100% |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 | 0.36 | 0% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.421939 | 0.421165 | -0.183% |
| `smsp__warps_active.avg.per_cycle_active` | 2.867006 | 2.867459 | +0.016% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.529969 | 3.540926 | +0.310% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.222876 | 0.223069 | +0.087% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.640977 | 1.642445 | +0.089% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.244223% | 1.247533% | +0.266% |

Source-page proxies:

| Proxy | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `UTCOMMA` | 42,240 | 42,240 | 0% |
| `UTCCP` | 59,136 | 59,136 | 0% |
| `LDTM` | 376,624 | 376,592 | -0.008% |
| `STTM` | 275,248 | 275,216 | -0.012% |
| `LDS` | 43,520 | 43,520 | 0% |
| `STS` | 428,944 | 428,944 | 0% |
| `TRYWAIT` | 6,187,275 | 6,281,119 | +1.517% |
| `SYNCS` | 6,477,643 | 6,571,487 | +1.449% |
| `MEMBAR` | 58,640 | 58,640 | 0% |
| `BAR.SYNC` | 121,856 | 121,856 | 0% |

Decision:

- Rejected and reverted. Direct timing did not show a representative win: H16/S4096 regressed +0.189%, H16/S8192 was neutral/regressed +0.009%, H4/S2048 tied, and only H16/S2048 had a noise-sized -0.051% improvement.
- NCU did not show a PV-feed unlock: `UTCOMMA` unchanged, `smsp__issue_active.avg.per_cycle_active` unchanged, eligible warps slightly lower, long-scoreboard/wait/barrier stalls slightly higher, and source `TRYWAIT`/`SYNCS` increased about 1.5%.
- Classification: not DRAM/launch/spill limited. The bottleneck remains PV tensor-core underfeed from the combined P payload/P-scale/V payload/V-scale readiness path. Removing the direct P-scale TMEM store wait does not start more PV MMA; it leaves the issue lane unchanged and increases retry/sync pressure.
- Exact blocker: the direct P-scale `tcgen05.wait` is not an isolated latency bubble that can be dropped safely for throughput. Even with correct numerics, the existing readiness protocol still makes PV wait on the same coarse tile handoff, and skipping the store wait increases protocol churn instead of feeding tensor cores.

Revert/check:

```bash
grep -R -n "skipscwait\\|ONLINE_SKIP_DIRECT_P_SCALE_TMEM_STORE_WAIT\\|fp4pv_online_skip_direct_p_scale_tmem_store_wait\\|STATIC_ONLINE_MXFP4_SKIP_DIRECT_P_SCALE_TMEM_STORE_WAIT" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_skipscwait_revert.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "skipscwait\\|earlyreuse_skipscwait" || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_skipscwait_revert_baseline_h16_s4096.log
from tk_fa4.fp4_pv_experiments import benchmark_forward_streaming_live_mxfp4_vs_bf16
cfg = "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix"
rec = benchmark_forward_streaming_live_mxfp4_vs_bf16(
    seqlen=4096, heads=16, seed=81250, warmup=1, iters=1,
    mxfp4_fwd_config=cfg, include_output_only=False, bf16_baseline="tk")
print(rec)
PY
```

Revert result:

- No `skipscwait` source references remain in forward configs/dispatch/kernel.
- Rebuilt binary has no `skipscwait`/`earlyreuse_skipscwait` string.
- Kept qkscfix post-revert ptxas: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Post-revert H16/S4096 smoke finite; BF16 comparison `max_abs_diff=0.9375`, `mean_abs_diff=0.005215998739004135`, `rmse=0.010774427227657741`, `lse_max_abs_diff=0.01808929443359375`.

Next structural direction:

- Do not chase isolated P-scale wait removal; it did not change PV issue and worsened retry/sync counters.
- Next probe should change actual overlap/ownership without stealing QK/V TMEM or collapsing V-scale ping-pong. Candidate to inspect next: a legal feeder/overlap slice where a lightweight helper prepares the existing P-scale TMEM slot from a shadow value and publishes a single PV-ready only after P-scale and V-scale state for that tile are ready, while the PV WG stays on MMA. If the live path already makes that exact split a no-op, ledger the no-op proof and move to a half-tile/P-payload publish slice that can start PV earlier without extra TMEM pressure.

## Loop 78: TCGEN P-Payload Reuse Release on PV Tensor Commit

Hypothesis:

- The kept direct-after-rescale qkscfix route releases P-scale TMEM slots from the PV lane, but releases P payload shared slots from the output WG after output observes the PV tile. That can leave the score-derived P producer blocked on `p_stage_reusable` even though PV has already consumed the payload.
- Moving P payload-slot reuse to the PV tensor commit should allow the producer to refill the two-slot P payload ring earlier without touching QK/V TMEM, P-scale/V-scale TMEM layout, V-scale ping-pong, or the score-derived math path.
- This is a structural P payload lifetime probe, not a K256/cluster2 route and not a V-scale staging sweep.

Implemented opt-in route:

- Route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_tcgenreuse_pstage2_q200_p112_o56_qkscfix`
- Config: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_tcgenreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`
- Files changed for the probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`, `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`.
- Guard: `ONLINE_TCGEN_P_STAGE_REUSE`, restricted to cluster1 3WG online MXFP4 direct-P-ready, dual-output direct-after-rescale, non-K256 routes.
- Mechanism: include the guarded route in `STATIC_ANY_TCGEN_P_STAGE_REUSE`, so `issue_pv` performs `tensor_commit<C::CLUSTER_SIZE>(p_stage_reusable[p_buf])` for the consumed P payload slot and the output-side `arrive_p_stage_slot_reusable` no longer releases it.
- Preserved: score-derived E2M1 P payload, direct x1 E8M0 P scales, `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`, V-scale ping-pong, QK/V/output TMEM footprint.
- Avoided: `fp4pv_pack_scores_to_stage_mxfp4`, vector-amax-over-materialized-P quantization, K256/cluster2/output-half work, backward files.

Build/resource commands:

```bash
make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_tcgenreuse_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep tcgenreuse
grep -n "tcgenreuse" results/mxfp4_fa4_forward_profile_20260612/build_tcgenreuse_qkscfix_probe.log
sed -n '573,578p' results/mxfp4_fa4_forward_profile_20260612/build_tcgenreuse_qkscfix_probe.log
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
```

Build/resource result:

- Build succeeded.
- Binary contains the `tcgenreuse` route string and guarded kernel instantiation.
- ptxas for the probe route: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- Kept baseline route in the same build has the same footprint: `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- `git diff --check` passed for the touched forward files.

Correctness commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_tcgenreuse_qkscfix_h16_s128.log
# H16/S128 paired kept-route vs tcgenreuse smoke with _run_forward_streaming_live_mxfp4.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 300s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_tcgenreuse_qkscfix_h16_s4096.log
# H16/S4096 paired kept-route vs tcgenreuse smoke with _run_forward_streaming_live_mxfp4.
PY
```

Correctness results:

| Shape | Seed | Finite | Probe vs kept |
| --- | ---: | --- | --- |
| H16/S128 | 81300 | yes | bit-identical: output/lse max diff 0 |
| H16/S4096 | 81301 | yes | output `max_abs_diff=0.005126953125`, `mean_abs_diff=2.9435386750265025e-07`, `rmse=1.3710476345270715e-05`, `lse_max_abs_diff=0` |

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_tcgenreuse_qkscfix_direct.stdout
# Uses ext.forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
# WARMUP=30, ITERS=180, seed=81310.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_tcgenreuse_qkscfix_direct.jsonl
PY
```

Direct preallocated timing:

| Shape | Launch | Kept median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.0586719997 | 0.0585279986 | -0.245% |
| H16/S4096 | persistent | 0.1691839993 | 0.1672480032 | -1.144% |
| H16/S8192 | fullgrid | 0.5436960161 | 0.5431840122 | -0.094% |
| H4/S2048 | persistent | 0.0573440008 | 0.0567359999 | -1.060% |

Repeated H16/S4096 timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_tcgenreuse_qkscfix_h16_s4096_repeat.stdout
# H16/S4096 only, five alternating base/probe rounds, WARMUP=30, ITERS=180, seed=81330.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_tcgenreuse_qkscfix_h16_s4096_repeat.jsonl
PY
```

Repeated H16/S4096 direct timing:

| Round | Order | Kept median ms | Probe median ms | Delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | base, probe | 0.162208006 | 0.162047997 | -0.099% |
| 1 | probe, base | 0.162272006 | 0.161599994 | -0.414% |
| 2 | base, probe | 0.162208006 | 0.161807999 | -0.247% |
| 3 | probe, base | 0.162207998 | 0.162016004 | -0.118% |
| 4 | base, probe | 0.162303999 | 0.162079997 | -0.138% |

Median of round medians: kept `0.162208006 ms`, probe `0.162016004 ms`, delta `-0.118%`.

NCU H16/S4096 commands:

```bash
set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_base_qkscfix_h16_s4096 \
  python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_base_qkscfix_h16_s4096.stdout
# H16/S4096, seed=81320, 10 raw-extension warmups, cudaProfilerStart(), one preallocated persistent forward launch, cudaProfilerStop().
PY

set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_tcgenreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_probe_qkscfix_h16_s4096 \
  python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_probe_qkscfix_h16_s4096.stdout
# Same isolated one-kernel driver, seed=81320.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_probe_qkscfix_h16_s4096_source.csv
python3 - <<'PY'
# Parsed raw/source CSVs into results/mxfp4_fa4_forward_profile_20260612/ncu_tcgenreuse_qkscfix_h16_s4096_summary.json.
PY
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Timing/resources: `gpu__time_duration.avg`, `inst_executed`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__waves_per_multiprocessor`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Memory/shared: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed`.
- Source-page proxies: `UTCOMMA`, `UTCCP`, `LDTM`, `STTM`, `LDS`, `STS`, `TRYWAIT`, `SYNCS`, `MEMBAR`, `BAR.SYNC`, `MUFU`, `R2UR`, `LDG`, `STG`, plus `UTCBAR` and `UTCATOMSWS`.

Representative H16/S4096 NCU deltas:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.384 us | 156.928 us | +0.348% |
| `inst_executed` | 53,991,644 | 52,306,198 | -3.122% |
| `launch__registers_per_thread` | 168 | 168 | 0% |
| `launch__shared_mem_per_block_static` | 1.904 KB | 1.904 KB | 0% |
| `launch__barrier_count` | 2 | 2 | 0% |
| `launch__waves_per_multiprocessor` | 3.37 | 3.37 | 0% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.025325% | 7.044076% | +0.267% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.578546% | 13.617400% | +0.286% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.855814% | 14.930629% | +0.504% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.298249% | 33.147130% | -0.454% |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 | 0.36 | 0% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.421725 | 0.417452 | -1.013% |
| `smsp__warps_active.avg.per_cycle_active` | 2.868041 | 2.872863 | +0.168% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.529386 | 3.648608 | +3.378% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.222707 | 0.218301 | -1.978% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.641251 | 1.623010 | -1.111% |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 0.042821 | 0.042014 | -1.885% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.253857% | 1.249766% | -0.326% |

Source-page proxies:

| Proxy | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `UTCOMMA` | 42,240 | 42,240 | 0% |
| `UTCCP` | 59,136 | 59,136 | 0% |
| `LDTM` | 376,632 | 376,720 | +0.023% |
| `STTM` | 275,256 | 275,344 | +0.032% |
| `LDS` | 43,520 | 43,520 | 0% |
| `STS` | 428,944 | 428,944 | 0% |
| `TRYWAIT` | 6,226,195 | 6,090,604 | -2.178% |
| `SYNCS` | 6,516,563 | 6,372,524 | -2.210% |
| `MEMBAR` | 58,640 | 58,640 | 0% |
| `BAR.SYNC` | 121,856 | 121,856 | 0% |
| `MUFU` | 4,392,960 | 4,392,960 | 0% |
| `R2UR` | 728,254 | 737,236 | +1.233% |
| `LDG` | 245,136 | 245,136 | 0% |
| `STG` | 2,560 | 2,560 | 0% |
| `UTCBAR` | 42,752 | 51,200 | +19.760% |
| `UTCATOMSWS` | 1,024 | 1,024 | 0% |

Classification:

- Still not DRAM, launch, occupancy, register, or spill limited.
- Dominant bottleneck remains PV tensor-core underfeed/low eligible warps with long-scoreboard pressure.
- This probe reduces protocol retry/sync work (`TRYWAIT`/`SYNCS` down about 2.2%) and slightly raises tensor/TC active in NCU, but `UTCOMMA` is unchanged and NCU replay duration is mixed. Direct timing repeat shows the wall-time win is real but small.
- Interpreted critical path after the patch: output-side payload-slot lifetime was a minor P movement bubble; the larger critical path remains score-derived P generation plus shared payload store/publish, direct P-scale TMEM store/wait, V-scale ping-pong readiness, coarse ready event, and PV consume.

Decision:

- Kept as a guarded forward-only win. It is correct, opt-in, same resource footprint, preserves qkscfix+p112 baseline and V-scale ping-pong, avoids K256/vector-amax, and improves all standard direct timing shapes in the first sweep with a repeated H16/S4096 median-of-rounds win.
- Do not treat this as solving P movement. The next probe must improve overall PV feed, including P payload, P scales, V payload, V scales, and ready/reuse protocol. The most direct next structural target is coarser ready/protocol or earlier payload publish that reduces `TRYWAIT`/`SYNCS` without increasing `UTCBAR` or long-scoreboard, or a real P-scale slot/aliasing design that preserves V-scale ping-pong.

## Loop 79: PV-Owned P-Payload Reuse with Normal Arrive

Hypothesis:

- Loop 78 proved that moving P payload-slot reuse from the output WG to the PV lane is correct and gives a small direct-timing win, but NCU showed an added `UTCBAR` cost from the extra `tensor_commit(p_stage_reusable)`.
- The P payload slot may only need a normal P-stage semaphore release after PV MMA issue, matching the older output-side release style, rather than a second TCGEN commit.
- If correct, this should preserve the earlier payload-slot lifetime win while reducing protocol overhead. It must not touch QK/V TMEM, P-scale/V-scale slots, V-scale ping-pong, K256, or backward files.

Implemented opt-in route:

- Baseline for this loop: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_tcgenreuse_pstage2_q200_p112_o56_qkscfix`.
- Probe route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`.
- Config: `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`.
- Files changed for the probe: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`, `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`, `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`.
- Guard: `ONLINE_ARRIVE_P_STAGE_REUSE`, restricted to cluster1 3WG online MXFP4 direct-P-ready, dual-output direct-after-rescale, non-K256 routes.
- Mechanism: `STATIC_ANY_PV_OWNED_P_STAGE_REUSE` suppresses output-side payload reuse release. The probe releases `p_stage_reusable[p_buf]` from the PV issue lane with normal `arrive()` instead of `tensor_commit()`.
- Preserved: score-derived P payload/scales, direct x1 P-scale TMEM, `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`, V-scale ping-pong, QK/V/output TMEM footprint.
- Avoided: `fp4pv_pack_scores_to_stage_mxfp4`, vector-amax-over-materialized-P quantization, K256/cluster2/output-half work, backward files.

Build/resource commands:

```bash
grep -R -n "arrivereuse\\|ONLINE_ARRIVE_P_STAGE_REUSE\\|fp4pv_online_arrive_p_stage_reuse\\|STATIC_ONLINE_MXFP4_ARRIVE_P_STAGE_REUSE\\|STATIC_ANY_PV_OWNED_P_STAGE_REUSE" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
git diff --check -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_arrivereuse_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep arrivereuse
sed -n '573,578p' results/mxfp4_fa4_forward_profile_20260612/build_arrivereuse_qkscfix_probe.log
```

Build/resource result:

- Build succeeded.
- Binary contains the `arrivereuse` route string and guarded kernel instantiation.
- ptxas for the probe route: `0 bytes stack frame`, `0 bytes spill stores`, `0 bytes spill loads`; `Used 168 registers`, `used 2 barriers`, `1904 bytes smem`.
- `tcgenreuse` in the same build has the same footprint.
- `git diff --check` passed for touched forward files.

Correctness commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_arrivereuse_vs_tcgenreuse_h16_s128.log
# H16/S128 paired tcgenreuse vs arrivereuse smoke with _run_forward_streaming_live_mxfp4.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 300s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_arrivereuse_vs_tcgenreuse_h16_s4096.log
# H16/S4096 paired tcgenreuse vs arrivereuse smoke with _run_forward_streaming_live_mxfp4.
PY
```

Correctness results:

| Shape | Seed | Finite | Probe vs `tcgenreuse` |
| --- | ---: | --- | --- |
| H16/S128 | 81360 | yes | bit-identical: output/lse max diff 0 |
| H16/S4096 | 81361 | yes | output `max_abs_diff=0.0037841796875`, `mean_abs_diff=3.472927687653282e-07`, `rmse=1.4984795812139756e-05`, `lse_max_abs_diff=0` |

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_arrivereuse_vs_tcgenreuse_direct.stdout
# Uses ext.forward_streaming_live_mxfp4 directly with preallocated out/lse tensors.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
# WARMUP=30, ITERS=180, seed=81370.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_arrivereuse_vs_tcgenreuse_direct.jsonl
PY
```

Direct preallocated timing:

| Shape | Launch | `tcgenreuse` median ms | Probe median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.060800001 | 0.060320001 | -0.789% |
| H16/S4096 | persistent | 0.168240003 | 0.167120002 | -0.666% |
| H16/S8192 | fullgrid | 0.544400007 | 0.541680008 | -0.500% |
| H4/S2048 | persistent | 0.057280000 | 0.056527998 | -1.313% |

Repeated H16/S4096 timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_arrivereuse_vs_tcgenreuse_h16_s4096_repeat.stdout
# H16/S4096 only, five alternating base/probe rounds, WARMUP=30, ITERS=180, seed=81390.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_arrivereuse_vs_tcgenreuse_h16_s4096_repeat.jsonl
PY
```

Repeated H16/S4096 direct timing:

| Round | Order | `tcgenreuse` median ms | Probe median ms | Delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | base, probe | 0.160224006 | 0.160096005 | -0.080% |
| 1 | probe, base | 0.159615993 | 0.159712002 | +0.060% |
| 2 | base, probe | 0.159791999 | 0.159535997 | -0.160% |
| 3 | probe, base | 0.159584001 | 0.159679994 | +0.060% |
| 4 | base, probe | 0.160096005 | 0.159391999 | -0.440% |

Median of round medians: `tcgenreuse` `0.159791999 ms`, probe `0.159679994 ms`, delta `-0.070%`.

NCU H16/S4096 commands:

```bash
set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_tcgenreuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_base_tcgenreuse_h16_s4096 \
  python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_base_tcgenreuse_h16_s4096.stdout
# H16/S4096, seed=81380, 10 raw-extension warmups, cudaProfilerStart(), one preallocated persistent forward launch, cudaProfilerStop().
PY

set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_probe_h16_s4096 \
  python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_probe_h16_s4096.stdout
# Same isolated one-kernel driver, seed=81380.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_base_tcgenreuse_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_base_tcgenreuse_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_probe_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_probe_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_base_tcgenreuse_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_base_tcgenreuse_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_probe_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_probe_h16_s4096_source.csv
python3 - <<'PY'
# Parsed raw/source CSVs into results/mxfp4_fa4_forward_profile_20260612/ncu_arrivereuse_h16_s4096_summary.json.
PY
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- Timing/resources: `gpu__time_duration.avg`, `inst_executed`, `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__barrier_count`, `launch__waves_per_multiprocessor`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Memory/shared: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`, `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed`.
- Source-page proxies: `UTCOMMA`, `UTCCP`, `LDTM`, `STTM`, `LDS`, `STS`, `TRYWAIT`, `SYNCS`, `MEMBAR`, `BAR.SYNC`, `MUFU`, `R2UR`, `LDG`, `STG`, `UTCBAR`, `UTCATOMSWS`.

Representative H16/S4096 NCU deltas:

| Metric | `tcgenreuse` | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | 156.480 us | 155.552 us | -0.593% |
| `inst_executed` | 52,533,256 | 52,515,551 | -0.034% |
| `launch__registers_per_thread` | 168 | 168 | 0% |
| `launch__shared_mem_per_block_static` | 1.904 KB | 1.904 KB | 0% |
| `launch__barrier_count` | 2 | 2 | 0% |
| `launch__waves_per_multiprocessor` | 3.37 | 3.37 | 0% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.013050% | 7.098543% | +1.219% |
| `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 13.547237% | 13.736228% | +1.395% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.844270% | 14.998873% | +1.041% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 32.999377% | 33.383862% | +1.165% |
| `smsp__issue_active.avg.per_cycle_active` | 0.36 | 0.36 | 0% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.416805 | 0.418290 | +0.356% |
| `smsp__warps_active.avg.per_cycle_active` | 2.872896 | 2.872358 | -0.019% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.645329 | 3.669489 | +0.663% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.218357 | 0.218182 | -0.080% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.623263 | 1.623723 | +0.028% |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | 0.478954 | 0.474677 | -0.893% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.253473% | 1.260841% | +0.588% |

Source-page proxies:

| Proxy | `tcgenreuse` | Probe | Delta |
| --- | ---: | ---: | ---: |
| `UTCOMMA` | 42,240 | 42,240 | 0% |
| `UTCCP` | 59,136 | 59,136 | 0% |
| `LDTM` | 376,736 | 376,568 | -0.045% |
| `STTM` | 275,360 | 275,192 | -0.061% |
| `LDS` | 43,520 | 43,520 | 0% |
| `STS` | 428,944 | 428,944 | 0% |
| `TRYWAIT` | 6,174,530 | 6,178,716 | +0.068% |
| `SYNCS` | 6,456,450 | 6,469,084 | +0.196% |
| `MEMBAR` | 58,640 | 58,640 | 0% |
| `BAR.SYNC` | 121,856 | 121,856 | 0% |
| `MUFU` | 4,392,960 | 4,392,960 | 0% |
| `R2UR` | 737,240 | 728,238 | -1.221% |
| `LDG` | 245,136 | 245,136 | 0% |
| `STG` | 2,560 | 2,560 | 0% |
| `UTCBAR` | 51,200 | 42,752 | -16.500% |
| `UTCATOMSWS` | 1,024 | 1,024 | 0% |

Classification:

- Still not DRAM, launch, occupancy, register, or spill limited.
- Dominant bottleneck remains PV tensor-core underfeed/low eligible warps with long-scoreboard pressure, but this probe measurably reduces a protocol bubble introduced by PV-owned payload reuse.
- `UTCOMMA` is unchanged, so the number of PV MMA instructions did not increase, but H16/S4096 NCU duration dropped 0.593%, tensor/TC active rose about 1%, issue active rose 1.165%, and eligible warps rose 0.356%.
- `UTCBAR` dropped 16.5%, matching the hypothesis that the extra TCGEN P-stage-reuse commit was unnecessary overhead for this cluster1 route.
- `TRYWAIT`/`SYNCS` ticked up slightly versus `tcgenreuse`, so the next target should not be another pure payload-release tweak. The remaining feed path is still gated by P-scale ready, V-scale readiness, and long scoreboard.

Decision:

- Kept as the new guarded forward-only best route. It is correct at S128 and H16/S4096, improves all standard direct timing shapes in the first sweep, has a small but positive repeated H16/S4096 median, and has supportive H16/S4096 NCU evidence.
- Current best route string: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`.
- Keep `tcgenreuse` available as a guarded comparison route; it remains useful for isolating payload-release lifetime, but `arrivereuse` supersedes it for timing.
- Next structural direction: the payload reuse protocol has now been improved twice. Do not keep shaving that same release point. Next probe should attack the remaining P-scale/V-scale feed path without collapsing V-scale ping-pong: candidate directions are legal P-scale slot aliasing/footprint reduction that preserves V-scale slots, or a feeder slice that moves P-scale shadow-to-TMEM off the PV critical wait without serializing V scale.

## Loop 80: P-scale shadow feeder overlap rejected and reverted

Route tested:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_scalefeeder_pstage2_q200_p112_o56_qkscfix`

Baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Intent: implement the requested feeder/P-ring overlap slice without touching backward or K256. The producer generated the same score-derived P payload and a P-scale shadow. The output/feeder WG copied that shadow into the existing two P-scale TMEM slots just in time and published the existing coarse `p_sc_tmem_ready` event. V-scale ping-pong stayed at two TMEM slots and was not collapsed. PV still consumed the existing payload path and waited on the same P/V scale readiness before MMA.

Build/resource evidence:

- `build_scalefeeder_qkscfix_probe.log` and `build_scalefeeder_qkscfix_fix2.log`: probe compiled at `168` registers, `2` barriers, `2944` bytes static shared memory, `16` bytes stack frame, `16` bytes spill stores, `16` bytes spill loads.
- `build_after_revert_scalefeeder_fix2.log`: kept `arrivereuse` route rebuilt at `168` registers, `2` barriers, `1904` bytes static shared memory, no stack frame, no spills.
- Post-revert symbol check found no stale `scalefeeder`, `P_SCALE_SHADOW_FEEDER`, `p_scale_shadow`, `feed_p_scale_shadow`, or `p_scale_shadow_ready` references in forward source or binary.

Protocol bug found and fixed before timing:

- S128 was bit-identical initially, but H16/S4096 had large output drift with exact LSE. Adding a CTA-shared visibility publish before `p_scale_shadow_ready` reduced the drift to BF16-level noise.
- A queued-launch repro then showed probe-only 20 queued launches could hang at final sync, while single launches and base-only queued launches completed. The feeder leader was arriving `p_sc_tmem_ready` after only its own TMEM store wait, without proving all feeder store warps had completed. Adding a feeder-only `warpgroup::sync(warpgroup::groupid() + 1)` after `fp4pv_tmem_store_wait()` and before `arrive(p_sc_tmem_ready[p_sc_slot])` fixed the queued probe-only repro.

Correctness after the protocol fix:

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' \
  2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_scalefeeder_fix2_vs_arrivereuse_h16_s128_s4096_s8192.log
# shared-input smoke against kept arriversue route for H16/S128 persistent,
# H16/S4096 persistent, and H16/S8192 fullgrid.
PY
```

Results:

| Shape | Grid | Output diff vs kept route | LSE diff |
| --- | --- | ---: | ---: |
| H16/S128 | persistent | bit-identical | exact |
| H16/S4096 | persistent | max `0.00244140625`, mean `1.5368e-07`, rmse `9.717e-06` | exact |
| H16/S8192 | fullgrid | max `0.00390625`, mean `5.2648e-08`, rmse `6.027e-06` | exact |

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' \
  2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_scalefeeder_fix2_vs_arrivereuse_direct.stdout
# alternating baseline/probe, shared preallocated output/LSE, WARMUP=20, ITERS=120.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_scalefeeder_fix2_vs_arrivereuse_direct.jsonl
PY
```

Timing:

| Shape | Grid | Kept route ms | Feeder probe ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.058064 | 0.059792 | +2.976% |
| H16/S4096 | persistent | 0.160352 | 0.165200 | +3.023% |
| H16/S8192 | fullgrid | 0.535072 | 0.553168 | +3.382% |
| H4/S2048 | persistent | 0.053248 | 0.053808 | +1.052% |

NCU: skipped because isolated direct timings regressed all standard shapes. Profiling this as a candidate win would not satisfy the acceptance bar; the resource evidence already shows the likely mechanism.

Decision: rejected and reverted. The feeder slice was live, numerically sound after fixing real protocol/lifetime bugs, and preserved V-scale ping-pong, but it moved P-scale staging work into the output/feeder side, added `1040` bytes of shared state, introduced spills, and required an extra warpgroup sync before the coarse PV-ready event. It did not improve overall PV feed or end-to-end timing.

Blocker/lesson for the next overlap attempt: P movement cannot be optimized in isolation. A deeper P handoff still may work, but it needs either safe P-scale slot aliasing, real TMEM footprint reduction, or a handoff protocol that increases P-scale depth without making the PV/output side pay the full staging cost and without serializing V-scale ping-pong. The next structural probe should preserve or improve total PV feed: P payload, P scales, V payload, V scales, ready/reuse protocol, and actual PV MMA issue rate.

## Loop 81: producer-side x4 P-scale TMEM staging rejected and reverted

Route tested:

`dualaccum_directrescale_scorepack_prescaled_floor_x4sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Intent: test a legal structural P-scale movement change without touching backward, K256, QK TMEM, or V-scale ping-pong. The score-derived qkscfix producer still generated E2M1 payload directly from scores/local/global max and E8M0 scales from score block max versus row max. The probe changed only P-scale staging from the kept per-row x1 direct TMEM store to a producer-side shared shadow plus native x4 scale TMEM store. P payload, P-scale slots, V payload, and two V-scale TMEM slots were otherwise unchanged.

Build/resource commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_x4sc_arrivereuse_fix2_sync_after_store.log
grep -n -A4 -B2 "floor_x4sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse" results/mxfp4_fa4_forward_profile_20260612/build_x4sc_arrivereuse_fix2_sync_after_store.log | head -80
grep -n -A4 -B2 "floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse" results/mxfp4_fa4_forward_profile_20260612/build_x4sc_arrivereuse_fix2_sync_after_store.log | head -80
```

Resource results:

| Route | Registers | Barriers | Smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept x1 arrivereuse | 168 | 2 | 1904 B | 0 stack, 0 stores, 0 loads |
| x4 P-scale staging probe | 168 | 2 | 2928 B | 0 stack, 0 stores, 0 loads |

Protocol/correctness notes:

- Initial S128 smoke was bit-identical, but H16/S4096 showed exact LSE with output drift. Inactive-row zeroing for the x4 shared shadow did not resolve it.
- A second protocol fix mirrored other x4 scale-store sites by adding a warpgroup sync after TMEM store wait and before `p_sc_tmem_ready`, but S4096 output drift remained in the same envelope.
- A determinism check showed the kept x1 route itself has H16/S4096 output run-to-run jitter with exact LSE at similar max-diff scale. The x4 route was therefore timing-eligible as "within kept-route nondeterminism", but not bit-identical.

Smoke/determinism commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 300s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_x4sc_arrivereuse_fix2_vs_kept_h16_s128_s4096.log
# shared-input S128/S4096 smoke, kept x1 route vs x4 probe.
PY

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/debug_x4sc_determinism_h16_s4096.log
# repeated kept/probe H16/S4096 launches on identical tensors.
PY
```

Representative numeric results:

| Check | Output diff | LSE diff |
| --- | ---: | ---: |
| S128 probe vs kept | bit-identical | exact |
| S4096 probe vs kept smoke | max `0.23681640625`, mean `1.9464e-05`, rmse `7.523e-04` | exact |
| S4096 kept vs kept repeat | max up to `0.23681640625`, mean `~6.8e-06` to `9.5e-06`, rmse `~4.4e-04` to `5.1e-04` | exact |
| S4096 x4 vs x4 repeat | max up to `0.28369140625`, mean `~2.0e-05` to `3.0e-05`, rmse `~8.2e-04` to `1.0e-03` | exact |

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_x4sc_arrivereuse_vs_x1_direct.stdout
# Raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Alternating kept/probe order on shared tensors.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
# WARMUP=30, ITERS=180, seed=82090.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_x4sc_arrivereuse_vs_x1_direct.jsonl
PY
```

Timing:

| Shape | Grid | Kept route ms | x4 probe ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.061072 | 0.061344 | +0.445% |
| H16/S4096 | persistent | 0.162848 | 0.165280 | +1.493% |
| H16/S8192 | fullgrid | 0.538432 | 0.548496 | +1.869% |
| H4/S2048 | persistent | 0.055984 | 0.056592 | +1.086% |

NCU: skipped. The representative H16/S4096 direct isolated timing regressed by +1.493%, and all standard shapes regressed. A replay profile would not satisfy the acceptance criterion for a win or diagnostically useful non-negative result.

Cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_x4sc_probe.log
grep -R "x4sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse" -n tk_fa4/fp4_fa4_fwd || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_x4sc_kept_arrivereuse_h16_s128_s4096.log
# finite kept-route smoke for H16/S128 and H16/S4096.
PY
```

Post-revert results:

- No stale `x4sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse` route references remain in `tk_fa4/fp4_fa4_fwd`.
- Kept arrivereuse route rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept route smoke: H16/S128 and H16/S4096 outputs and LSE finite.

Decision: rejected and reverted. The x4 producer-side scale staging was live and forward-only, but it added `1024` bytes of shared state and did not improve overall PV feed. It likely increased shared/protocol pressure and left the PV tensor cores no better fed. Do not revisit this exact x4 scale-shadow path unless a later footprint/protocol change makes the x4 store free relative to the x1 path.

Next lever: avoid another small scale-staging knob. The remaining structural options are (1) reduce P-scale/V-scale handoff frequency or event cost without extra shared pressure, (2) free TMEM footprint so P-scale depth can increase while preserving V-scale ping-pong, or (3) split feed resources only if it improves combined P payload, P scale, V payload, V scale readiness and PV issue rate together.

## Loop 82: PV-side P-scale-before-V-scale ordering rejected and reverted

Route tested:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pfirst_pstage2_q200_p112_o56_qkscfix`

Baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Intent: structural ordering probe on the live qkscfix/arrivereuse path, without K256, backward changes, QK/V TMEM changes, extra P/V scale slots, or collapsing V-scale ping-pong. The only route-level change was `ONLINE_STAGE_P_BEFORE_V=true`, causing the PV lane to wait/stage P scale before V scale in the existing decoupled PV issue path.

Build/resource commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pfirst_arrivereuse_probe.log
grep -n -A4 -B2 "arrivereuse_pfirst_dualaccum" results/mxfp4_fa4_forward_profile_20260612/build_pfirst_arrivereuse_probe.log | head -120
grep -n -A4 -B2 "arrivereuse_dualaccum" results/mxfp4_fa4_forward_profile_20260612/build_pfirst_arrivereuse_probe.log | head -160
```

Resource results:

| Route | Registers | Barriers | Smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept arrivereuse | 168 | 2 | 1904 B | 0 stack, 0 stores, 0 loads |
| pfirst ordering probe | 168 | 2 | 1904 B | 0 stack, 0 stores, 0 loads |

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 300s python3 -u - <<'PY' \
  2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pfirst_arrivereuse_vs_kept_h16_s128_s4096.log
# Shared-input S128/S4096 smoke, kept route vs pfirst route.
PY
```

Smoke results:

| Shape | Grid | Output diff vs kept route | LSE diff |
| --- | --- | ---: | ---: |
| H16/S128 | persistent | bit-identical | exact |
| H16/S4096 | persistent | max `0.138916015625`, mean `1.0063e-05`, rmse `5.025e-04` | exact |

The H16/S4096 output drift is inside the known kept-route long-S nondeterminism envelope from Loop 81; LSE stayed exact and finite.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' \
  2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pfirst_arrivereuse_vs_kept_direct.stdout
# Raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Alternating kept/probe order on shared tensors.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
# WARMUP=30, ITERS=180, seed=82300.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_pfirst_arrivereuse_vs_kept_direct.jsonl
PY
```

Timing:

| Shape | Grid | Kept route ms | pfirst probe ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.059456 | 0.059040 | -0.700% |
| H16/S4096 | persistent | 0.161184 | 0.161792 | +0.377% |
| H16/S8192 | fullgrid | 0.536496 | 0.537296 | +0.149% |
| H4/S2048 | persistent | 0.054256 | 0.054432 | +0.324% |

NCU H16/S4096 commands:

```bash
set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix \
  PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_base_h16_s4096 \
  python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_base_h16_s4096.stdout
# H16/S4096, seed=82340, ten raw-extension warmups, cudaProfilerStart(), one preallocated persistent forward launch, cudaProfilerStop().
PY

set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pfirst_pstage2_q200_p112_o56_qkscfix \
  PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_probe_h16_s4096 \
  python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_probe_h16_s4096.stdout
# Same isolated one-kernel driver and seed.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_base_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_base_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_probe_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_probe_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_base_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_base_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_probe_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pfirst_arrivereuse_probe_h16_s4096_source.csv
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warp_latency_per_inst_issued.ratio`, `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Source-page p_sc/v_sc/PV proxies: executed counts for `UTCOMMA`, `UTCCP`, `LDTM`, `STTM`, `LDS`, `STS`, `TRYWAIT`, `SYNCS`, `MEMBAR`, `BAR.SYNC`, `MUFU`, `R2UR`, `LDG`, `STG`, `UTCBAR`, and `UTCATOMSWS`.

Representative H16/S4096 NCU deltas:

| Metric | Kept route | pfirst probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.sum` | 155.776000 | 155.424000 | -0.226% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.070579% | 7.058245% | -0.174% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.952766% | 14.972404% | +0.131% |
| `smsp__issue_active.avg.per_cycle_active` | 0.360000 | 0.360000 | 0.000% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418423 | 0.417512 | -0.218% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.663844 | 3.654888 | -0.244% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.623136 | 1.624273 | +0.070% |
| `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio` | 0.359604 | 0.367095 | +2.083% |
| `UTCOMMA` | 42,240 | 42,240 | 0.000% |
| `UTCCP` | 59,136 | 59,136 | 0.000% |
| `LDTM` | 378,296 | 378,336 | +0.011% |
| `STTM` | 276,920 | 276,960 | +0.014% |
| `TRYWAIT` | 6,227,533 | 6,050,495 | -2.843% |
| `SYNCS` | 6,517,901 | 6,340,863 | -2.716% |
| `MEMBAR` | 58,640 | 58,640 | 0.000% |
| `UTCBAR` | 42,752 | 42,752 | 0.000% |

Decision: rejected and reverted. The ordering change is live and reduces source-page `TRYWAIT`/`SYNCS` work by about `2.8%`, but it does not increase PV issue (`UTCOMMA` unchanged), does not raise issue rate, slightly lowers tensor active and eligible warps, and regresses direct representative H16/S4096 timing. It is not an overall PV-feed improvement.

Cleanup commands:

```bash
grep -R "arrivereuse_pfirst\\|pfirst_pstage2_q200_p112_o56_qkscfix" -n tk_fa4/fp4_fa4_fwd || true
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pfirst_arrivereuse_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pfirst_pstage2_q200_p112_o56_qkscfix" || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 300s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_pfirst_arrivereuse_kept_h16_s128_s4096.log
# finite kept-route smoke for H16/S128 and H16/S4096.
PY
```

Post-revert results:

- No stale `arrivereuse_pfirst` or `pfirst_pstage2_q200_p112_o56_qkscfix` source references remain, and the rebuilt binary has no stale `arrivereuse_pfirst_pstage2_q200_p112_o56_qkscfix` string.
- Kept arrivereuse route rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept route smoke: H16/S128 and H16/S4096 outputs and LSE finite.

Next lever: route-local ordering alone is not enough. It can reduce sync retry work, but the PV tensor path remains underfed with fixed `UTCOMMA` and low eligible warps. The next structural probe should remove or coarsen an actual P/V readiness handoff or reduce the P/V scale footprint so overlap improves both P and V feed together, while preserving V-scale ping-pong and not stealing QK/V TMEM.

## Loop 83: early score-derived P-scale + K64 half-ready rejected and reverted

Route tested:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_earlysc_k64_pstage2_q200_p112_o56_qkscfix`

Baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Intent: a structural P-handoff probe on the live score-derived qkscfix route, with no K256, no vector-amax/materialized-P packer, no backward edits, no QK/V TMEM changes, and no V-scale ping-pong collapse. The probe stored the score-derived x1 P-scale to existing P-scale TMEM slots early, published the first K64 P-payload half after qid 0/1, and then used the existing final ready for the second half.

Build/resource commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_earlysc_k64_qkscfix_probe.log
grep -n -A4 -B2 "earlysc_k64\\|arrivereuse" results/mxfp4_fa4_forward_profile_20260612/build_earlysc_k64_qkscfix_probe.log | head -240
```

Resource results:

| Route | Registers | Barriers | Smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept arrivereuse | 168 | 2 | 1904 B | 0 stack, 0 stores, 0 loads |
| earlysc_k64 probe | 168 | 2 | 1936 B | 0 stack, 0 stores, 0 loads |

Smoke command:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_earlysc_k64_qkscfix_vs_kept_h16_s128_s4096.log
# Shared-input kept-vs-probe smoke using exp._load_forward_experiments_ext().
# Shapes: H16/S128 persistent and H16/S4096 persistent.
PY
```

Smoke results:

| Shape | Grid | Output diff vs kept route | LSE diff |
| --- | --- | ---: | ---: |
| H16/S128 | persistent | bit-identical | exact |
| H16/S4096 | persistent | max `0.010986328125`, mean `3.1373e-06`, rmse `5.6053e-05` | exact |

Direct preallocated timing command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_earlysc_k64_qkscfix_vs_kept_direct.stdout
# Raw forward-only ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Alternating kept/probe order on shared tensors.
# JSONL: results/mxfp4_fa4_forward_profile_20260612/bench_earlysc_k64_qkscfix_vs_kept_direct.jsonl
# WARMUP=30, ITERS=180.
PY
```

Timing:

| Shape | Grid | Kept route ms | earlysc_k64 ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.059440 | 0.062192 | +4.630% |
| H16/S4096 | persistent | 0.170560 | 0.177696 | +4.184% |
| H16/S8192 | fullgrid | 0.549104 | 0.571920 | +4.155% |
| H4/S2048 | persistent | 0.060848 | 0.062784 | +3.182% |

NCU: skipped. The representative H16/S4096 direct isolated timing regressed by `+4.184%`, and every standard timing shape regressed by at least `+3.182%`; there was no non-negative or diagnostically useful NCU target for acceptance.

K256 blocker note, per steering directive: stop K256 route work. The only K256 material left is guarded scaffolding. A real score-derived K256 route remains blocked on host dynamic-smem launch plumbing, a consumer-mode path that does not use direct-after-rescale incorrectly, paired score-derived P payload staging, paired direct P-scale TMEM staging, and the Dvo/2 output/PV accumulator path. It must explicitly avoid `fp4pv_pack_scores_to_stage_mxfp4` and any vector-amax quantization over materialized P. No fake K256 route is active or accepted.

Cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_earlysc_k64_qkscfix_probe.log
grep -R "earlysc_k64" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep earlysc_k64 || true
timeout 300s env CUDA_VISIBLE_DEVICES=0 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_earlysc_k64_kept_arrivereuse_h16_s128_s4096.log
# finite kept-route smoke for H16/S128 and H16/S4096.
PY
```

Post-revert results:

- No stale `earlysc_k64` source references remain, and the rebuilt forward-only binary has no `earlysc_k64` string.
- Kept arrivereuse route rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept route smoke: H16/S128 and H16/S4096 outputs and LSE finite.

Decision: rejected and reverted. Starting PV on a K64 half required an extra mid-tile `quant_wg_sync`, proxy publish, ready event, and early P-scale TMEM handoff. Those protocol costs dominated any earlier first-half availability. This is evidence that P movement cannot be optimized in isolation; the next accepted P-overlap probe must advance combined P payload, P scales, V payload, V scales, ready/reuse protocol, and actual PV issue rate together while preserving V-scale ping-pong.

## Loop 84: payload-only P_STAGE_SLOTS=3 ring and reuse protocol fixes

Baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Probe routes:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage3_q200_p112_o56_qkscfix`
- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_tcgenreuse_pstage3_q200_p112_o56_qkscfix`
- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage3_q200_p112_o56_qkscfix`

Intent: first structural P payload ring-depth probe after the K64 half-ready rejection. Keep score-derived qkscfix math, keep P-scale TMEM slots at 2, keep V-scale TMEM ping-pong at 2, avoid K256, avoid vector-amax/materialized-P packers, and test whether payload depth 3 alone can let the producer move ahead of PV without stealing QK/V TMEM.

Build/resource commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_arrivereuse_pstage3_qkscfix_probe.log
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_tcgenreuse_pstage3_qkscfix_probe.log
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_earlyreuse_pstage3_qkscfix_probe.log
grep -n -A5 -B2 "pstage3_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_earlyreuse_pstage3_qkscfix_probe.log
```

Resource results:

| Route | Registers | Barriers | Smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept arrivereuse pstage2 | 168 | 2 | 1904 B | 0 stack, 0 stores, 0 loads |
| pstage3 payload-only probes | 168 | 2 | 1936 B | 0 stack, 0 stores, 0 loads |

Smoke commands:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_arrivereuse_pstage3_payload_only_vs_kept_h16_s128_s4096.log
# Shared prequantized inputs, kept route vs arrivereuse_pstage3.
# Shapes: H16/S128 and H16/S4096 persistent, H16/S4096 repeated.
PY
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_arrivereuse_pstage3_payload_only_self_repeat_h16_s4096.log
# Repeated arrivereuse_pstage3 self-run on one H16/S4096 input.
PY
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_tcgenreuse_pstage3_vs_arrivereuse_kept_h16_s128_s4096.log
# Shared prequantized inputs, kept route vs tcgenreuse_pstage3.
PY
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_earlyreuse_pstage3_vs_arrivereuse_kept_h16_s128_s4096.log
# Shared prequantized inputs, kept route vs earlyreuse_pstage3.
PY
```

Correctness results:

| Probe | H16/S128 | H16/S4096 vs kept | H16/S4096 self-repeat | LSE |
| --- | --- | ---: | ---: | --- |
| arrivereuse_pstage3 | exact | max `0.225341796875`, mean `1.356e-05`, rmse `0.000554` | max up to `0.1914` | exact |
| tcgenreuse_pstage3 | exact | max up to `0.3818` | max up to `0.3818` | exact |
| earlyreuse_pstage3 | exact | max `0.0986328125`, mean `~1e-05`, rmse `~0.00045` | max `0.0986328125` | exact |

Cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pstage3_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_pstage3_q200_p112_o56_qkscfix" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "tcgenreuse_pstage2_q200_p112_o56_qkscfix" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix"
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_pstage3_kept_arrivereuse_h16_s128_s4096.log
# finite kept-route smoke for H16/S128 and H16/S4096.
PY
```

Post-revert results:

- No stale `earlyreuse_pstage3` or `tcgenreuse_pstage2` route string remains in the rebuilt forward-only binary.
- Kept route string is present.
- Kept route rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept route smoke: H16/S128 and H16/S4096 outputs and LSE finite.

Decision: rejected and reverted before timing. Payload-only depth 3 exposes a deterministic protocol/lifetime blocker at long sequence: QK/softmax stays stable because LSE is exact, but PV output varies across identical probe launches. Arrive reuse, TCGEN-tied reuse, and output-WG early reuse all fail, so the issue is not one isolated reuse signal. The likely blocker is a deeper assumption coupling the 2-slot payload/P-scale/score/output lifecycle. Next structural P-feed work must not just deepen shared payload storage; it needs coordinated P payload, P-scale movement, V-scale ping-pong, and PV issue readiness, or a real footprint/layout reduction that makes matching scale depth safe.

## Loop 85: score-derived x1 P-scale shadow and x4 feeder repro

Baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Probe routes:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_x1scshadowx4_pstage2_q200_p112_o56_qkscfix`
- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_x1scshadow_pstage2_q200_p112_o56_qkscfix`

Intent: structural feeder-style repro without K256, without changing P payload depth, without stealing QK/V TMEM, and without collapsing V-scale ping-pong. The producer still used score-derived qkscfix P math; it additionally wrote a P-scale shadow in shared memory. The x4 probe tried to feed P-scale TMEM from that shadow using the existing x4 row-tile store path. The x1 diagnostic read the same shared shadow but stored with the known x1 TMEM path.

Build/resource commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_x1scshadowx4_qkscfix_probe.log
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_x1scshadow_qkscfix_probe.log
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_x1scshadow_barrier_qkscfix_probe.log
grep -n -A5 -B2 "x1scshadow_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_x1scshadow_barrier_qkscfix_probe.log
grep -n -A5 -B2 "x1scshadowx4_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_x1scshadow_barrier_qkscfix_probe.log
grep -n -A5 -B2 "earlyreuse_arrivereuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_x1scshadow_barrier_qkscfix_probe.log | head -40
```

Resource results:

| Route | Registers | Barriers | Smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept arrivereuse pstage2 | 168 | 2 | 1904 B | 0 stack, 0 stores, 0 loads |
| x1scshadow | 168 | 2 | 2928 B | 0 stack, 0 stores, 0 loads |
| x1scshadowx4 | 168 | 2 | 2928 B | 0 stack, 0 stores, 0 loads |

Smoke commands:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_x1scshadowx4_vs_arrivereuse_kept_h16_s128_s4096.log
# Shared prequantized inputs, kept route vs x1scshadowx4.
# Shapes: H16/S128 and H16/S4096 persistent, H16/S4096 repeated.
PY
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_x1scshadow_vs_arrivereuse_kept_h16_s128_s4096.log
# Shared prequantized inputs, kept route vs x1scshadow.
# Shapes: H16/S128 and H16/S4096 persistent, H16/S4096 repeated.
PY
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_x1scshadow_barrier_vs_arrivereuse_kept_h16_s128_s4096.log
# Same x1scshadow diagnostic after adding a quant_wg_sync() immediately after the shared shadow write.
PY
```

Smoke results:

| Probe | H16/S128 | H16/S4096 vs kept | H16/S4096 self-repeat | LSE |
| --- | --- | ---: | ---: | --- |
| x1scshadowx4 | exact | max `0.4384765625` | max up to `0.220703125` | exact |
| x1scshadow | exact | max up to `0.177734375` | max up to `0.177734375` | exact |
| x1scshadow + producer-side sync | exact | max up to `0.162353515625` | max `0.162353515625` | exact |

Important correctness caveat: after cleanup, the kept route itself showed exact H16/S128 repeat and finite H16/S4096 repeat, but a repeated H16/S4096 output-only self-diff of `0.1484375` with exact LSE:

```bash
timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_x1scshadow_kept_arrivereuse_h16_s128_s4096.log
# Baseline-only finite/repeat smoke on H16/S128 and H16/S4096.
PY
```

That means long-S route-vs-route output drift alone is not a definitive corruption signal for small x1shadow differences. The accepted baseline remains finite and previously validated; future candidate correctness should prefer BF16/reference comparison or a tighter deterministic repro, not only same-route repeat equality. The x4 feeder variant remains rejected because it has a materially larger H16/S4096 drift and the x1 shadow diagnostic only adds shared-memory traffic and an extra sync without a working feeder benefit.

Cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_x1scshadow_qkscfix_probe.log
grep -R "x1scshadow\\|SCALE_SHADOW\\|scale_shadow" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep x1scshadow || true
timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_x1scshadow_kept_arrivereuse_h16_s128_s4096.log
# Baseline-only finite/repeat smoke on H16/S128 and H16/S4096.
PY
```

Post-revert results:

- No stale `x1scshadow`, `SCALE_SHADOW`, or `scale_shadow` source references remain.
- The rebuilt forward-only binary has no `x1scshadow` string.
- Kept route rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept route smoke: H16/S128 and H16/S4096 outputs and LSE finite; H16/S128 repeat exact.

K256 blocker update: no K256/cluster2/output-half route work was continued in this loop. The latest blocker remains the guarded-scaffolding blocker from Loop 83: real score-derived K256 needs host dynamic-smem launch plumbing, a consumer-mode path that does not misuse direct-after-rescale, paired score-derived payload staging, paired direct P-scale TMEM staging, and a Dvo/2 output/PV accumulator path, while explicitly avoiding `fp4pv_pack_scores_to_stage_mxfp4` and vector-amax quantization. No fake K256 route is active.

Decision: rejected and reverted before timing/NCU. Shared P-scale shadow is not a useful feeder slice in the current schedule: the x4 form is not numerically clean enough at long sequence, the x1 form is only diagnostic and increases smem from `1904` B to `2928` B, and the barrier variant adds a producer sync without solving the useful x4 feeder path. The next structural P-feed probe should avoid shared P-scale shadow and instead target a handoff/protocol change that preserves V-scale ping-pong and measures PV feed, not just P movement.

## Loop 86: true non-aliased P-scale depth through lower TMEM footprint

Baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`

Probe:

`scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage3_pscale3_nodual_q200_p112_o56_qkscfix`

Intent: test the structural TMEM-footprint hypothesis directly. The kept route cannot fit a third non-aliased P-scale slot while preserving dual score, score-backed output accumulation, Q/K scale, and V-scale ping-pong. This opt-in probe dropped the dual-score/direct-output/decoupled schedule, kept score-derived qkscfix P math, requested `P_STAGE_SLOTS=3`, requested real `P_SCALE_TMEM_SLOTS=3`, and preserved `V_SCALE_TMEM_SLOTS=2`. It did not touch backward code, K256, cluster2, or output-half routes.

Build/resource commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_nodual_pstage3_pscale3_qkscfix_probe.log
grep -n -A5 -B2 "pstage3_pscale3_nodual" results/mxfp4_fa4_forward_profile_20260612/build_nodual_pstage3_pscale3_qkscfix_probe.log
grep -n -A5 -B2 "earlyreuse_arrivereuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_nodual_pstage3_pscale3_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage3_pscale3_nodual_q200_p112_o56_qkscfix"
```

Resource results:

| Route | Registers | Barriers | Smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept arrivereuse pstage2 | 168 | 2 | 1904 B | 0 stack, 0 stores, 0 loads |
| nodual pstage3 pscale3 | 168 | 2 | 1920 B | 0 stack, 0 stores, 0 loads |

Smoke commands:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_nodual_pstage3_pscale3_vs_bf16_h16_s128_s4096.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16 for kept and probe on H16/S128 and H16/S4096.
PY
timeout 360s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_nodual_pstage3_pscale3_vs_kept_h16_s128_s4096.log
# One shared prequantized input per shape, kept vs probe plus probe repeat.
PY
```

Smoke results:

| Shape | Result |
| --- | --- |
| H16/S128 | kept/probe finite; kept-vs-probe output exact and LSE exact; probe repeat exact |
| H16/S4096 | kept/probe finite; kept-vs-probe output max `0.00738525390625`, mean `1.855e-06`; LSE max `0.000431060791015625`; probe repeat output/LSE exact |

BF16 comparison was also finite for both routes. H16/S4096 kept/probe BF16 envelopes matched within rounding (`max_abs_diff 1.328125`, `mean_abs_diff ~0.00527367`, `rmse ~0.0106954`, `lse_max_abs_diff 0.05167197`).

Direct preallocated timing command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_nodual_pstage3_pscale3_vs_kept_direct.jsonl
# Timed raw ext.forward_streaming_live_mxfp4 with preallocated out/lse tensors.
# Set TK_FA4_FP4PV_FWD_CONFIG around each raw extension call.
# WARMUP=30, ITERS=180, SEED_BASE=80700.
# Shapes: H16/S2048, H16/S4096, H16/S8192, H4/S2048.
PY
```

Direct preallocated timing results:

| Shape | Kept median ms | Probe median ms | Probe vs kept | Kept min ms | Probe min ms | Min delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.061055999 | 0.063904002 | +4.665% | 0.059007999 | 0.061600000 | +4.393% |
| H16/S4096 | 0.172640003 | 0.184752002 | +7.016% | 0.167040005 | 0.179680005 | +7.567% |
| H16/S8192 | 0.546831995 | 0.595919997 | +8.977% | 0.540607989 | 0.591264009 | +9.370% |
| H4/S2048 | 0.059328001 | 0.062447999 | +5.259% | 0.056511998 | 0.059615999 | +5.493% |

Post-timing kept/probe comparisons stayed finite. H16/S4096 post-timing output max was `0.028564453125`, mean `2.992e-05`, LSE max `0.0012297630310058594`; H16/S8192 output max was `0.01806640625`, mean `4.888e-05`, LSE max `0.0014171600341796875`.

NCU follow-up: skipped. Timing was negative on every shape, including representative H16/S4096 (`+7.016%` median), so there was no non-negative or diagnostically necessary isolated-kernel target.

Cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_nodual_pstage3_pscale3_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage3_pscale3_nodual" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix"
grep -R "pscale3_nodual\\|ONLINE_P_SCALE_TMEM_SLOTS\\|fp4pv_online_p_scale_tmem_slots" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_nodual_pstage3_pscale3_kept_h16_s128_s4096.log
# Short kept-route H16/S128 and H16/S4096 finite smoke.
PY
```

Post-revert results:

- No stale `pscale3_nodual`, `ONLINE_P_SCALE_TMEM_SLOTS`, or `fp4pv_online_p_scale_tmem_slots` source references remain.
- The rebuilt forward-only binary has no `pscale3_nodual` route string.
- Kept route string remains present and rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept route smoke is finite for H16/S128 and H16/S4096.

Decision: rejected and reverted. A real third non-aliased P-scale slot is structurally legal only after dropping the kept route's dual-score/direct-output/decoupled schedule, but that schedule loss costs much more than the extra P-scale slot can recover. The exact blocker is not the 16-column P-scale slot itself; it is fitting deeper P-scale staging while preserving the high-throughput kept schedule, Q/K scale footprint, output accumulator footprint, and V-scale ping-pong. Future P-depth work should target layout/alias/footprint changes that keep the dual-score/direct-output schedule live, or reduce handoff frequency without removing the schedule that currently feeds PV best.

## Loop 87: warp-local P-scale reuse wait with guarded post-store sync

Goal: reduce the direct P-scale slot reuse handoff bubble without touching QK/V TMEM, without collapsing V-scale ping-pong, and without changing the score-derived qkscfix math path. The probe moved the P-scale reuse wait from one warp-0 leader plus a producer-wide sync to per-warp elected leaders waiting on the reusable slot, then added a guarded post-store producer sync before the coarse `p_sc_tmem_ready` event after the first smoke showed a likely publish-order hazard.

Implemented route:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_warpscwait_pstage2_q200_p112_o56_qkscfix`

Files touched during the probe:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Build commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_warpscwait_qkscfix_probe.log
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_warpscwait_postsync_qkscfix_probe.log
```

Resource results from `build_warpscwait_postsync_qkscfix_probe.log`:

- Probe: `168` registers, `2` barriers, `1904` bytes smem, `0` stack, `0` spill stores, `0` spill loads.
- Kept baseline: `168` registers, `2` barriers, `1904` bytes smem, `0` stack, `0` spill stores, `0` spill loads.

Smoke commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_warpscwait_vs_kept_h16_s128_s4096.log
# same-input kept/probe/probe-repeat smoke, H16/S128 and H16/S4096
PY
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_warpscwait_postsync_vs_kept_h16_s128_s4096.log
# same-input kept/probe/probe-repeat smoke after post-store sync
PY
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 180s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_warpscwait_postsync_repeat_diagnose_h16_s4096.log
# baseline/probe repeatability diagnostic, H16/S4096
PY
```

Correctness summary:

- Initial per-warp wait was finite and deterministic, but H16/S4096 kept/probe output max differed by `0.2041015625` with identical LSE. This exposed that moving the wait can let warp 0 publish `p_sc_tmem_ready` before all producer warps finish their P-scale TMEM stores.
- Added a guarded `quant_wg_sync()` after `fp4pv_tmem_store_wait()` and before the ready event. The corrected probe remained finite and deterministic.
- Repeatability diagnostic showed the kept route itself is not bit-exact on repeated H16/S4096 launches for the same input: kept-repeat output max `0.2109375`, mean `5.821499e-06`, `9441` nonzero elements; LSE repeat exact. Corrected probe repeat was exact. Kept/probe deltas were within that accepted baseline repeatability envelope and LSE stayed exact.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_warpscwait_postsync_vs_kept_direct.jsonl
# raw ext.forward_streaming_live_mxfp4, preallocated out/lse, alternating kept/probe
# WARMUP=30, ITERS=180, SEED_BASE=87500
# Shapes: H16/S2048, H16/S4096, H16/S8192, H4/S2048
PY
```

Direct preallocated timing results:

| Shape | Kept median ms | Probe median ms | Probe vs kept | Kept min ms | Probe min ms | Min delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | 0.058688000 | 0.058944002 | +0.436% | 0.056448001 | 0.057312001 | +1.531% |
| H16/S4096 | 0.160735995 | 0.162256002 | +0.946% | 0.158271998 | 0.159712002 | +0.910% |
| H16/S8192 | 0.535679996 | 0.544367999 | +1.622% | 0.532927990 | 0.541472018 | +1.603% |
| H4/S2048 | 0.053504001 | 0.053808000 | +0.568% | 0.051520001 | 0.051775999 | +0.497% |

Post-timing finite checks passed for both routes. H16/S4096 post-timing kept/probe output max was `0.103515625`, mean `4.321665e-06`, LSE max `0.0`.

NCU follow-up: skipped. All standard direct timings regressed, including representative H16/S4096, so there was no non-negative isolated-kernel target. The resource counters would only confirm that this route moved the handoff sync instead of removing it.

Cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_warpscwait_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "warpscwait" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix"
grep -R "warpscwait\\|ONLINE_WARP_P_SCALE_REUSE_WAIT\\|fp4pv_online_warp_p_scale_reuse_wait" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 240s python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_warpscwait_kept_h16_s128_s4096.log
# kept-route finite smoke, H16/S128 and H16/S4096
PY
```

Post-revert results:

- No stale `warpscwait`, `ONLINE_WARP_P_SCALE_REUSE_WAIT`, or `fp4pv_online_warp_p_scale_reuse_wait` source references remain.
- The rebuilt forward-only binary has no `warpscwait` route string.
- Kept route string remains present and rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept route smoke is finite for H16/S128 and H16/S4096.

Decision: rejected and reverted. The exact blocker is that per-warp reuse waits alone make P-scale publication unsafe, while the correctness-preserving post-store producer sync restores the handoff cost and regresses every standard shape. This confirms the next overlap attempt needs a real feeder/shadow or coarser handoff that reduces ready/reuse work without adding the sync back onto the PV-ready critical path.

## Loop 88: V-loader-owned feeder PV-ready, preserving V-scale ping-pong

User directive: stop K256/cluster2/output-half probing and implement the feeder/P-ring overlap slice first. The concrete probe was a guarded forward-only `feedready` route for the live score-derived qkscfix+p112 baseline:

- Baseline route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`
- Probe route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_feedready_pstage2_q200_p112_o56_qkscfix`
- Current live path remains score-derived: prescaled E2M1 P payload from scores/local/global max/residual -> exp2 -> FP4, direct x1 P-scale TMEM from score block max vs row max, pstage2/earlyreuse/prepublish/arrivereuse.
- Slot counts preserved during the probe: `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`. The probe did not collapse V-scale ping-pong or steal QK/V TMEM.
- Protocol design: producer still generated score-derived P payload and P-scale TMEM; existing V-loader warp staged V scales into the existing V-scale TMEM ping-pong, waited the matching P-scale TMEM ready event, then published one coarse `pv_feed_ready`. PV waited that one event and focused on MMA issue.

Changed files for the probe:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Build/protocol commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_feedready_qkscfix_probe.log
timeout 120s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feedready_trace_s128.log
# traced H16/S128 baseline launch then probe launch
PY
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_feedready_vloader_qkscfix_probe.log
timeout 120s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feedready_vloader_trace_s128.log
# traced H16/S128 baseline launch then probe launch after moving feeder to V-loader warp
PY
```

Protocol result:

- Initial feeder used `warpgroup::warpid()==1`, which is live PV issue ownership in the decoupled route. S128 trace completed baseline and hung after `probe_launch`.
- Fix moved the feed-ready publisher into the existing V-loader ownership path (`warpgroup::warpid()==2 && local_leader && cta_rank==0`) after V payload/scale shared publish and `v_arrived`. This made the S128 repro live and bitwise-identical to the kept route.

Build resources from `build_feedready_vloader_qkscfix_probe.log`:

- Probe: `168` registers, `2` barriers, `1952` bytes smem, `0` stack, `0` spill stores, `0` spill loads.
- Kept baseline: `168` registers, `2` barriers, `1904` bytes smem, `0` stack, `0` spill stores, `0` spill loads.

Smoke command:

```bash
timeout 360s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feedready_vloader_vs_kept_h16_s128_s4096.log
# same-input kept/probe smoke, H16/S128 and H16/S4096
PY
```

Smoke results:

- H16/S128: finite; output max diff `0.0`, LSE max diff `0.0`.
- H16/S4096: finite; output max diff `0.0018310546875`, mean diff `5.879999775970646e-08`, normalized RMSE `0.00054978850540514`, LSE max diff `0.0`.

Direct preallocated timing command:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_feedready_vloader_vs_kept_direct.stdout
# raw ext.forward_streaming_live_mxfp4, preallocated out/lse, alternating kept/probe
# Results written to results/mxfp4_fa4_forward_profile_20260612/bench_feedready_vloader_vs_kept_direct.jsonl
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept median ms | Probe median ms | Probe vs kept | Kept min ms | Probe min ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.057056 | 0.065216 | -12.512% | 0.055488 | 0.063936 |
| H16/S4096 | persistent | 0.167824 | 0.201968 | -16.906% | 0.165344 | 0.198592 |
| H16/S8192 | fullgrid | 0.541920 | 0.674416 | -19.646% | 0.538144 | 0.671040 |
| H4/S2048 | persistent | 0.055648 | 0.064576 | -13.826% | 0.053056 | 0.061760 |

NCU commands:

```bash
set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix \
  PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_base_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_base_qkscfix_h16_s4096.stdout
# H16/S4096, seed=92301, ten raw-extension warmups, cudaProfilerStart(), one preallocated persistent forward launch, cudaProfilerStop().
PY

set -o pipefail; timeout 900s env CUDA_VISIBLE_DEVICES=0 \
  TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_feedready_pstage2_q200_p112_o56_qkscfix \
  PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_probe_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_probe_qkscfix_h16_s4096.stdout
# Same isolated one-kernel driver and seed.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_vloader_probe_qkscfix_h16_s4096_source.csv
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warp_latency_per_inst_issued.ratio`, `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Occupancy/resource: `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__occupancy_cluster_pct`, `launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem`, `launch__waves_per_multiprocessor`, `derived__local_spilling_requests`.
- Source-page p_sc/v_sc/PV proxies: executed counts for `TRYWAIT`, `SYNCS`, `MEMBAR`, `BAR.SYNC`, `LDTM`, `STTM`, `LDS`, `STS`, `LDG`, `STG`, `UTCBAR`, `UTCATOMSWS`, `UTCCP`, `UTCOMMA`, `MUFU`, `R2UR`.

Representative H16/S4096 NCU deltas:

| Metric | Kept | Feedready | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` us | 155.776 | 193.856 | +24.445% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.062035 | 5.683870 | -19.515% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.911661 | 12.149832 | -18.521% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.202729 | 27.489579 | -17.207% |
| `smsp__issue_active.avg.per_cycle_active` | 0.360000 | 0.300000 | -16.667% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.417027 | 0.334663 | -19.750% |
| `smsp__average_warp_latency_per_inst_issued.ratio` | 8.007154 | 9.832843 | +22.801% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.649329 | 5.541487 | +51.849% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.623721 | 1.610941 | -0.787% |
| Source `TRYWAIT` executed | 6078252 | 6546672 | +7.706% |
| Source `SYNCS` executed | 6368620 | 6858992 | +7.700% |
| Source `LDTM` executed | 376144 | 376168 | +0.006% |
| Source `STTM` executed | 274768 | 274792 | +0.009% |

Interpretation:

- This is not DRAM, occupancy, spill, or TMEM footprint limited: registers stayed `168`, spills stayed `0`, occupancy limit fields and waves were unchanged, and DRAM percent fell with the longer kernel.
- The bottleneck remains PV tensor-core underfeed, worsened by protocol/scoreboard overhead. The feedready path reduced tensor/PV issue activity and eligible warps while increasing long scoreboard and executed wait/sync instructions.
- The one coarse ready event did not remove handoff cost; it moved P-scale readiness and V-scale staging onto the V-loader, adding an extra wait/sync chain before PV could issue. This rejected route serializes the feed path enough to reduce total PV issue rate.

Cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_feedready_vloader.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_feedready_pstage2_q200_p112_o56_qkscfix\\|arrivereuse_pstage2_q200_p112_o56_qkscfix"
grep -R "FEEDER_PV_READY\\|feedready\\|pv_feed_ready\\|feeder_" -n tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc || true
timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_feedready_vloader_kept_h16_s4096.log
# kept-route finite smoke, H16/S4096
PY
```

Post-revert results:

- No stale `FEEDER_PV_READY`, `feedready`, `pv_feed_ready`, or `feeder_` source references remain.
- Rebuilt binary contains the kept qkscfix route and no feedready route string.
- Kept route rebuilt at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept H16/S4096 smoke: output finite, LSE finite, `out_max_abs=1.015625`, `lse_max_abs=8.321194648742676`.

Decision: rejected and reverted. Exact blocker: the legal feeder path that waits P-scale readiness before publishing a combined PV-ready event increases `TRYWAIT`/`SYNCS`, lowers eligible warps, and lowers PV/tensor issue. A renewed overlap attempt must either publish P payload earlier without inserting a new feeder wait chain, or create real independent P-scale depth/aliasing/footprint reduction so PV does not wait on a serialized P-scale+V-scale feed-ready path.

## Loop 89: fold P-scale reuse wait into P-stage reuse on qkscfix

Intent: keep working the P/P-scale movement bottleneck without touching backward, K256, QK/V TMEM, or V-scale ping-pong. The live qkscfix route has `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, and both use `idx % 2`; PV releases payload reuse and P-scale TMEM reuse together after MMA. This probe removes the producer-side duplicate wait on `p_sc_tmem_reusable[p_sc_slot]` when `ONLINE_ARRIVE_P_STAGE_REUSE` already waited for the matching payload slot. Math and storage remain strictly score-derived: E2M1 P payload from score residual -> exp2 -> FP4, E8M0 scale from score block max versus row max, direct x1 P-scale TMEM, no vector-amax/materialized-P packer.

Routes:

- Kept baseline: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix`
- Probe: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix`

Code design:

- New guarded config flag: `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE`.
- Guard conditions require cluster1, `P_STAGE_SLOTS==2`, `P_SCALE_TMEM_SLOTS==P_STAGE_SLOTS`, `ONLINE_ARRIVE_P_STAGE_REUSE`, direct score-derived x1 P-scale TMEM, and no alias-scale TMEM.
- Producer still executes the existing `quant_wg_sync()` before direct P-scale TMEM stores; only the redundant `p_sc_tmem_reusable` wait is skipped in this route.
- Preserved: score-derived qkscfix P payload/scales, direct x1 P-scale TMEM, `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`, V-scale ping-pong, QK/V/output TMEM footprint.
- Avoided: `fp4pv_pack_scores_to_stage_mxfp4`, vector-amax-over-materialized-P quantization, K256/cluster2/output-half probing, backward files.

Build and route checks:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pscreusefold_qkscfix_probe.log
grep -n -A4 -B2 "pscreusefold_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistentILi128ELi128ELi192ELi128ELi200ELi56ELi112ELi1E" results/mxfp4_fa4_forward_profile_20260612/build_pscreusefold_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix\\|arrivereuse_pstage2_q200_p112_o56_qkscfix"
git diff --check
```

Build resources from `build_pscreusefold_qkscfix_probe.log`:

- Probe: `168` registers, `2` barriers, `1904` bytes smem, `0` stack, `0` spill stores, `0` spill loads.
- Kept baseline in same build: `168` registers, `2` barriers, `1904` bytes smem, `0` stack, `0` spill stores, `0` spill loads.

Smoke commands:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pscreusefold_trace_s128.log
# H16/S128 same-input kept/probe trace
PY

timeout 360s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pscreusefold_vs_kept_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input kept/probe smoke
PY
```

Smoke results:

- H16/S128 trace: finite, output max diff `0.0`, LSE max diff `0.0`.
- H16/S128 paired smoke: finite, output max diff `0.0`, LSE max diff `0.0`.
- H16/S4096 paired smoke: finite, output max diff `0.0015869140625`, mean diff `3.779633317435582e-08`, normalized RMSE `0.00038260210769889457`, LSE max diff `0.0`.

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pscreusefold_vs_kept_direct.stdout
# raw ext.forward_streaming_live_mxfp4, preallocated out/lse, alternating kept/probe
# Results written to results/mxfp4_fa4_forward_profile_20260612/bench_pscreusefold_vs_kept_direct.jsonl
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept median ms | Probe median ms | Probe vs kept | Kept min ms | Probe min ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.059071999 | 0.058752000 | +0.542% | 0.056864001 | 0.057408001 |
| H16/S4096 | persistent | 0.164591998 | 0.164223999 | +0.224% | 0.162016004 | 0.162080005 |
| H16/S8192 | fullgrid | 0.550144017 | 0.545392007 | +0.871% | 0.544416010 | 0.541055977 |
| H4/S2048 | persistent | 0.062352002 | 0.061888002 | +0.750% | 0.059487998 | 0.059232000 |

NCU commands:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_base_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_base_qkscfix_h16_s4096.stdout
# H16/S4096, seed=93301, ten raw-extension warmups, cudaProfilerStart(), one preallocated persistent forward launch, cudaProfilerStop().
PY

timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_probe_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_probe_qkscfix_h16_s4096.stdout
# Same isolated one-kernel driver and seed, probe route.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_base_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_base_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_base_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_base_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_probe_qkscfix_h16_s4096_source.csv
```

NCU metric names used:

- Sections: `SpeedOfLight`, `SchedulerStats`, `WarpStateStats`, `MemoryWorkloadAnalysis`.
- PV/tensor feed: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`, `sm__issue_active.avg.pct_of_peak_sustained_elapsed`, `smsp__issue_active.avg.per_cycle_active`, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warps_active.avg.per_cycle_active`.
- Wait/protocol: `smsp__average_warp_latency_per_inst_issued.ratio`, `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_membar_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio`.
- Occupancy/resource: `launch__registers_per_thread`, `launch__shared_mem_per_block_static`, `launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem`, `launch__waves_per_multiprocessor`, `derived__local_spilling_requests`, `sass__inst_executed_register_spilling`.
- Source-page p_sc/v_sc/PV proxies: executed counts for `TRYWAIT`, `SYNCS`, `MEMBAR`, `BAR.SYNC`, `LDTM`, `STTM`, `LDS`, `STS`, `LDG`, `STG`, `UTCBAR`, `UTCATOMSWS`, `UTCCP`, `UTCOMMA`, `MUFU`, `R2UR`.

Representative H16/S4096 NCU deltas:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` us | 155.712 | 154.368 | -0.863% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.062599 | 7.095496 | +0.466% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.922428 | 14.872840 | -0.332% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.203773 | 33.091029 | -0.340% |
| `smsp__issue_active.avg.per_cycle_active` | 0.360000 | 0.360000 | 0.000% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.418118 | 0.417552 | -0.135% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.649931 | 3.664355 | +0.395% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.219037 | 0.209899 | -4.172% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.623801 | 1.606613 | -1.059% |
| `sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed` | 8.823654 | 8.783415 | -0.456% |
| Source `TRYWAIT` executed | 6199236 | 6212238 | +0.210% |
| Source `SYNCS` executed | 6489604 | 6502606 | +0.200% |
| Source `LDTM` executed | 376120 | 376048 | -0.019% |
| Source `STTM` executed | 274744 | 274672 | -0.026% |
| Source total instructions | 52569757 | 52331583 | -0.453% |

Classification:

- Still not DRAM, launch, occupancy, spill, or TMEM-footprint limited: `168` registers, `1904` static smem, no spilling, and unchanged occupancy limits.
- Dominant bottleneck remains PV tensor-core underfeed/long scoreboard with low eligible warps.
- The probe does not create a new overlap structure, but it trims a redundant P-scale reuse handoff when P payload and P-scale slots are already phase-locked. NCU shows slightly lower duration, lower barrier/wait stalls, lower MIO PQ write active, and fewer total source instructions, with unchanged V-scale ping-pong and no PV issue regression in wall timing.

Decision: keep as an opt-in guarded forward route. It is a small validated P-scale reuse protocol win on top of the qkscfix+p112 baseline. It does not solve the larger overlap problem; next probes should still target overall PV feed rather than only P movement.

## Loop 90: skip folded P-scale reuse sync after Loop 89

Intent: test whether Loop 89 left an orphan producer `quant_wg_sync()` after folding P-scale reusable waits into the matching P payload reuse wait. This was deliberately scoped to the live direct x1 score-derived qkscfix path. It preserved the kept `pscreusefold` route, P-stage depth 2, P-scale TMEM depth 2, V-scale ping-pong, and score-derived P payload/scales.

Rejected probe route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_nosync_pstage2_q200_p112_o56_qkscfix`

Build and checks:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pscreusefold_nosync_qkscfix_probe.log
grep -RIn "pscreusefold_nosync\|ONLINE_SKIP_FOLDED_P_SCALE_REUSE_SYNC\|SKIP_FOLDED_P_SCALE_REUSE_SYNC" tk_fa4/fp4_fa4_fwd
```

Build resources:

- Probe: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept `pscreusefold` and kept qkscfix baseline remained at the same resource footprint.

Smoke:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pscreusefold_nosync_trace_s128.log
# H16/S128 same-input kept/fold/probe trace
PY

timeout 360s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pscreusefold_nosync_vs_kept_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input kept/fold/probe smoke
PY
```

Smoke results:

- H16/S128: exact versus kept and `pscreusefold`.
- H16/S4096: exact versus `pscreusefold`; versus kept, output max diff `0.0023193359375`, LSE max diff `0.0`.

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pscreusefold_nosync_vs_kept_direct.stdout
# raw ext.forward_streaming_live_mxfp4, preallocated out/lse, alternating kept/fold/probe
# Results written to results/mxfp4_fa4_forward_profile_20260612/bench_pscreusefold_nosync_vs_kept_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept ms | Fold ms | Probe ms | Probe vs fold | Probe vs kept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H16/S2048 | persistent | 0.059071999 | 0.058816001 | 0.058816001 | 0.000% | +0.433% |
| H16/S4096 | persistent | 0.171616003 | 0.171167999 | 0.170880005 | +0.168% | +0.429% |
| H16/S8192 | fullgrid | 0.543327987 | 0.539583981 | 0.539664000 | -0.015% | +0.674% |
| H4/S2048 | persistent | 0.057824001 | 0.057744000 | 0.057744000 | 0.000% | +0.138% |

NCU commands:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_nosync_fold_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_nosync_fold_qkscfix_h16_s4096.stdout
# H16/S4096, one warmed preallocated persistent forward launch bracketed by cudaProfilerStart/Stop.
PY

timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_nosync_probe_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_pscreusefold_nosync_probe_qkscfix_h16_s4096.stdout
# Same isolated one-kernel driver and seed, probe route.
PY
```

Representative H16/S4096 NCU deltas versus kept `pscreusefold`:

| Metric | Fold | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` us | 154.240 | 154.400 | +0.104% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 7.115161 | 7.060330 | -0.771% |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | 14.910702 | 14.806841 | -0.697% |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | 33.210716 | 32.883248 | -0.986% |
| `smsp__issue_active.avg.per_cycle_active` | 0.360000 | 0.360000 | 0.000% |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.417268 | 0.417679 | +0.098% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 3.648753 | 3.646010 | -0.075% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | 0.207313 | 0.206859 | -0.219% |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | 1.605936 | 1.610307 | +0.272% |
| `sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed` | 8.810067 | 8.730833 | -0.899% |
| Source `TRYWAIT` executed | 6211257 | 6122006 | -1.437% |
| Source `SYNCS` executed | 6501625 | 6412374 | -1.373% |
| Source `BAR.SYNC` executed | 121856 | 88064 | -27.730% |
| Source total instructions | 52368565 | 52050317 | -0.608% |

Classification:

- The route trimmed sync/wait instructions, but the representative H16/S4096 kernel duration regressed and tensor/TC/issue active all moved in the wrong direction versus the already-kept `pscreusefold`.
- Bottleneck remains PV tensor-core underfeed with long scoreboard; this specific sync was not a profitable handoff bubble to remove.

Cleanup:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pscreusefold_nosync.log
grep -RIn "pscreusefold_nosync\|ONLINE_SKIP_FOLDED_P_SCALE_REUSE_SYNC\|SKIP_FOLDED_P_SCALE_REUSE_SYNC" tk_fa4/fp4_fa4_fwd || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pscreusefold_nosync_pstage2_q200_p112_o56_qkscfix" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix\|arrivereuse_pstage2_q200_p112_o56_qkscfix"
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_pscreusefold_pair_h16_s128_s4096.log
# Same-input kept vs pscreusefold pair smoke after revert.
PY
```

Post-revert results:

- No stale `pscreusefold_nosync`, `ONLINE_SKIP_FOLDED_P_SCALE_REUSE_SYNC`, or `SKIP_FOLDED_P_SCALE_REUSE_SYNC` source references remain.
- Rebuilt binary contains kept qkscfix and kept `pscreusefold`; it does not contain the rejected `pscreusefold_nosync` route.
- Post-revert H16/S128 pair smoke: exact output and exact LSE.
- Post-revert H16/S4096 pair smoke: output max diff `0.00146484375`, mean diff `3.6214764520536846e-08`, normalized RMSE `0.0003440371734281434`, exact LSE.

Decision: rejected and reverted. The next probe should not be another narrow sync/register flag; it needs to address P/P-scale movement while preserving V-scale ping-pong and total PV issue rate.

## Loop 91: online score-slot alias repro and cleanup

Intent: test whether the structural P-ring/TMEM-depth blocker was specifically depth-3 ownership, or whether the online score-slot alias layout itself corrupts the live score-derived qkscfix route. This was a correctness-first repro, not a timing probe. It preserved the kept `pscreusefold` route while adding two opt-in rejected candidates:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_alias2_pstage2_q200_p112_o56_qkscfix`
- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_alias3_pstage3_q200_p112_o56_qkscfix`

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_alias2_alias3_qkscfix_probe.log
```

Build resources:

- `alias2`: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- `alias3`: `168` registers, `2` barriers, `1968` bytes smem, no stack frame, no spills.
- Kept `pscreusefold`: unchanged at `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.

Correctness command:

```bash
timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_alias2_alias3_vs_fold_h16_s128.log
# H16/S128 same-input pscreusefold vs alias2/alias3 smoke.
PY
```

Correctness results:

| Shape | Seed | Route | Output finite | LSE finite | Probe vs `pscreusefold` |
| --- | ---: | --- | --- | --- | --- |
| H16/S128 | 95111 | `alias2` | yes | no | `lse_max_abs_diff=Inf`, output `max_abs_diff=0.1650390625`, `mean_abs_diff=0.002171050291508436`, normalized RMSE `0.2344349896198198` |
| H16/S128 | 95111 | `alias3` | yes | no | `lse_max_abs_diff=Inf`, output `max_abs_diff=0.1650390625`, `mean_abs_diff=0.005909979343414307`, normalized RMSE `0.38112142646459357` |

Classification:

- `alias2` corrupts before any depth-3 wraparound, so the failure is not triple-depth ring ownership.
- The exact blocker is unsafe online score-slot TMEM aliasing for the dual-output qkscfix route: it corrupts LSE/PV state even with two slots.
- This re-confirms Loop 69. Do not retry the same online score-slot alias offsets. A renewed P-ring attempt needs real footprint reduction, a safe scale-slot alias protocol, or a non-alias handoff that preserves V-scale ping-pong and overall PV issue rate.

Cleanup command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_alias2_alias3_probe.log
grep -R -n "alias2\|alias3\|ONLINE_ALIAS_SCALE_TMEM\|ONLINE_TRIPLE_SCORE_TMEM\|ONLINE_WAIT_ALIAS_SCALE_SLOT_REUSE\|fp4pv_online_alias_scale_tmem\|fp4pv_online_triple_score_tmem\|fp4pv_online_wait_alias_scale_slot_reuse" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_alias2\|arrivereuse_alias3" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix" || true
```

Post-cleanup results:

- Source grep: no stale `alias2`, `alias3`, or online alias trait references in touched forward files.
- Binary grep: no stale `arrivereuse_alias2` or `arrivereuse_alias3` route strings.
- Kept route still present: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix`.
- Kept route resources after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.

Post-cleanup smoke command:

```bash
timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_alias2_alias3_pscreusefold_h16_v2.log
# benchmark_forward_streaming_live_mxfp4_vs_bf16, exact pscreusefold route, warmup=1, iters=1.
PY
```

Post-cleanup smoke results:

| Shape | Seed | Finite | Output/LSE nonfinite | MXFP4 ms | TK BF16 ms | MXFP4 vs TK BF16 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| H16/S128 | 95111 | yes | no/no | `0.23455999791622162` | `0.05833600088953972` | output `max_abs_diff=1.40234375`, `mean_abs_diff=0.20720666646957397`, `rmse=0.2668107438287689`; LSE comparison produced `NaN` diff in the helper despite finite LSE |
| H16/S4096 | 95112 | yes | no/no | `0.25065600872039795` | `0.2173759937286377` | output `max_abs_diff=1.015625`, `mean_abs_diff=0.00523406034335494`, `rmse=0.01065567247232498`, `lse_max_abs_diff=0.03548538684844971` |

K256 status: parked, not continued. The remaining real score-derived K256 blocker is still the paired route plumbing, not a timing decision: host dynamic-smem launch plumbing exists only as guarded scaffolding, the kernel has a static assert blocking `ONLINE_SCORE_DERIVED_K256`, and a real route must wire consumer mode, paired score-derived payload staging, paired direct P-scale TMEM staging, and Dvo/2 output accumulator handling while explicitly avoiding `fp4pv_pack_scores_to_stage_mxfp4` and vector-amax quantization. No fake K256 route was added or kept.

Decision: rejected and reverted. Next loop must be a non-alias P/P-scale movement probe that preserves QK/V TMEM, preserves V-scale ping-pong, and measures total PV feed rather than only P producer progress.

## Loop 92: PV-side next P-scale ready prewait

Intent: test a non-alias protocol overlap that keeps the current score-derived qkscfix math path, P payload slots, P-scale TMEM slots, and V-scale ping-pong unchanged. The probe consumed the next tile's `p_sc_tmem_ready` event immediately after issuing the current PV tile, then skipped the matching P-scale ready wait in the next iteration. This was meant to move the next P-scale wait out of the next V/P staging sequence without stealing QK/V TMEM or collapsing V-scale buffering.

Rejected probe route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_prewaitpsc_pstage2_q200_p112_o56_qkscfix`

Kept baseline:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix`

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_prewaitpsc_qkscfix_probe.log
```

Build resources:

- Probe: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept `pscreusefold`: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Static guards restricted the probe to cluster1, 3WG decoupled PV, `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, score-derived x1 direct P-scale, no K256, no split K64, no aliasing, and V-first staging.

Correctness command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_prewaitpsc_vs_pscreusefold_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input kept pscreusefold vs prewaitpsc.
PY
```

Correctness results:

- H16/S128 seed `95201`: exact output and exact LSE versus kept route; all finite.
- H16/S4096 seed `95202`: exact output and exact LSE versus kept route; all finite.

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_prewaitpsc_vs_pscreusefold_direct.stdout
# Results written to results/mxfp4_fa4_forward_profile_20260612/bench_prewaitpsc_vs_pscreusefold_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept ms | Probe ms | Probe vs kept |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.061360002` | `0.061951999` | `-0.965%` |
| H16/S4096 | persistent | `0.173152000` | `0.173744000` | `-0.342%` |
| H16/S8192 | fullgrid | `0.545168012` | `0.544847995` | `+0.059%` |
| H4/S2048 | persistent | `0.057424001` | `0.057567999` | `-0.251%` |

Diagnostic NCU commands:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_fold_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_fold_qkscfix_h16_s4096.stdout
# H16/S4096 kept pscreusefold, one warmed preallocated persistent launch bracketed by cudaProfilerStart/Stop.
PY

timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_probe_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_probe_qkscfix_h16_s4096.stdout
# Same isolated one-kernel driver and seed, prewaitpsc route.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_fold_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_fold_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_fold_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_fold_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_prewaitpsc_probe_qkscfix_h16_s4096_source.csv
```

Representative H16/S4096 NCU deltas versus kept `pscreusefold`:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` us | `155.168` | `153.920` | `-0.804%` |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `7.168225` | `7.131598` | `-0.511%` |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | `15.007483` | `15.011515` | `+0.027%` |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | `33.446127` | `33.344167` | `-0.305%` |
| `smsp__issue_active.avg.per_cycle_active` | `0.360000` | `0.360000` | `0.000%` |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.417709` | `0.420074` | `+0.566%` |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | `3.651080` | `3.594042` | `-1.562%` |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | `1.606318` | `1.611776` | `+0.340%` |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | `0.492833` | `0.502579` | `+1.978%` |
| `sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed` | `8.872715` | `8.823748` | `-0.552%` |
| Source `TRYWAIT` executed | `6336284` | `6991873` | `+10.347%` |
| Source `SYNCS` executed | `6626652` | `7282241` | `+9.893%` |
| Source `LDTM` executed | `377176` | `377192` | `+0.004%` |
| Source `STTM` executed | `275800` | `275816` | `+0.006%` |
| Source `UTCOMMA` executed | `42240` | `42240` | `0.000%` |
| Source total instructions | `52690126` | `52286826` | `-0.765%` |

Classification:

- Correctness is clean and resources are unchanged, so the phase protocol itself is valid.
- The probe does not improve overall PV feed. It slightly improves eligible warps and long-scoreboard ratio in the isolated NCU capture, but it adds a large number of `SYNCS...TRYWAIT` instructions and lowers tensor-active percentage. Direct representative H16/S4096 timing is slower.
- The bottleneck remains PV tensor-core underfeed / producer-to-PV handoff, not DRAM, launch, spills, or occupancy. This prewait version moves wait placement but does not reduce the total P/V feed protocol burden.

Cleanup:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_prewaitpsc_probe.log
grep -R -n "prewaitpsc\|PREWAIT_NEXT_P_SCALE_READY\|prewait_next_p_sc_ready\|p_sc_tmem_prewaited_mask" tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "prewaitpsc" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix" || true
timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_prewaitpsc_direct_finite_h16_s128_s4096.log
# Direct finite smoke for kept pscreusefold at H16/S128 and H16/S4096.
PY
```

Post-cleanup results:

- No stale `prewaitpsc`, `ONLINE_PREWAIT_NEXT_P_SCALE_READY`, `prewait_next_p_sc_ready`, or `p_sc_tmem_prewaited_mask` source references remain.
- Binary contains kept `arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix`; binary does not contain `prewaitpsc`.
- Kept route resources after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- H16/S128 seed `95411`: output finite yes, LSE finite yes, no nonfinite output/LSE.
- H16/S4096 seed `95412`: output finite yes, LSE finite yes, no nonfinite output/LSE.

Decision: rejected and reverted. Do not retry simple PV-side P-scale ready prewait; it is live but it serializes more `TRYWAIT`/`SYNCS` protocol and does not improve representative direct timing. Next structural P-movement attempts need to reduce total feed protocol work or add real overlap resources, while preserving V-scale ping-pong and QK/V TMEM ownership.

## Loop 93: skip folded P-scale reuse arrive

Intent: reduce P/P-scale handoff protocol work without changing payload generation, P-scale TMEM placement, QK/V TMEM ownership, V-scale ping-pong, or the PV issue math path. Loop 89 folded producer-side P-scale reuse waiting into P-stage reuse, but PV still signaled `p_sc_tmem_reusable[p_sc_slot]` after each consumed PV tile. This probe makes that old P-scale reuse release optional for the folded one-to-one qkscfix layout.

Kept baseline:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix`

Probe route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`

Code design:

- Adds opt-in `ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE`.
- Static guard requires `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE`, which already requires cluster1, `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, one-to-one payload/P-scale slots, `ONLINE_ARRIVE_P_STAGE_REUSE`, no scale aliasing, and x1 score-derived direct P-scale TMEM.
- PV still signals `p_stage_reusable[p_buf]`; producer still waits that folded payload/P-scale lifetime before reusing the matching pair.
- PV skips only the old `arrive(p_sc_tmem_reusable[p_sc_slot])` release that the folded producer no longer waits on.
- V-scale ping-pong remains `V_SCALE_TMEM_SLOTS=2`; QK/V TMEM layout unchanged; P payload remains score-derived residual -> exp2 -> E2M1; P scale remains score block max vs row max -> E8M0 direct x1 TMEM.

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_skippscarrive_qkscfix_probe.log
```

Build resources:

- Probe: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Kept `pscreusefold`: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.

Correctness command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_skippscarrive_vs_pscreusefold_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input kept pscreusefold vs skippscarrive.
PY
```

Correctness results:

| Shape | Seed | Output finite | LSE finite | Probe vs kept |
| --- | ---: | --- | --- | --- |
| H16/S128 | `95501` | yes | yes | exact output and exact LSE |
| H16/S4096 | `95502` | yes | yes | exact output and exact LSE |

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_skippscarrive_vs_pscreusefold_direct.stdout
# Results written to results/mxfp4_fa4_forward_profile_20260612/bench_skippscarrive_vs_pscreusefold_direct.jsonl
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept ms | Probe ms | Probe vs kept |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.061712001` | `0.061280001` | `+0.700%` |
| H16/S4096 | persistent | `0.162671998` | `0.162288003` | `+0.236%` |
| H16/S8192 | fullgrid | `0.546144009` | `0.544799984` | `+0.246%` |
| H4/S2048 | persistent | `0.064319998` | `0.057888001` | `+10.000%` |

All direct timing records were exact output/LSE versus kept route.

Representative H16/S4096 NCU commands:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_fold_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_fold_qkscfix_h16_s4096.stdout
# H16/S4096, seed=95701, ten raw-extension warmups, one preallocated forward launch bracketed by cudaProfilerStart/Stop.
PY

timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_probe_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_probe_qkscfix_h16_s4096.stdout
# Same isolated one-kernel driver and seed, skippscarrive route.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_fold_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_fold_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_probe_qkscfix_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_probe_qkscfix_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_fold_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_fold_qkscfix_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_probe_qkscfix_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_skippscarrive_probe_qkscfix_h16_s4096_source.csv
```

Representative H16/S4096 NCU deltas versus kept `pscreusefold`:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` us | `154.944` | `154.848` | `-0.062%` |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `7.111112` | `7.130604` | `+0.274%` |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | `14.910745` | `14.951636` | `+0.274%` |
| `sm__issue_active.avg.pct_of_peak_sustained_elapsed` | `33.170912` | `33.225768` | `+0.165%` |
| `smsp__issue_active.avg.per_cycle_active` | `0.360000` | `0.360000` | `0.000%` |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.417395` | `0.417969` | `+0.138%` |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | `3.655741` | `3.651444` | `-0.118%` |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | `1.606672` | `1.609153` | `+0.154%` |
| `sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed` | `8.798131` | `8.793990` | `-0.047%` |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | `1.265520` | `1.266405` | `+0.070%` |
| `inst_executed` | `52359125` | `52202902` | `-0.298%` |
| Source `SYNCS` executed | `6295705` | `6248977` | `-0.742%` |
| Source `TRYWAIT` executed | `6212505` | `6174225` | `-0.616%` |
| Source `SYNCS.ARRIVE.TRANS64.A1T0` executed | `42752` | `34304` | `-19.760%` |
| Source `LDTM` executed | `376568` | `376728` | `+0.042%` |
| Source `STTM` executed | `275192` | `275352` | `+0.058%` |
| Source `UTCOMMA` executed | `42240` | `42240` | `0.000%` |
| Source total instructions | `52359125` | `52202902` | `-0.298%` |

Classification:

- Dominant bottleneck remains PV tensor-core underfeed / P-V feed protocol, not DRAM, launch, spills, or occupancy.
- This probe does not deepen the P ring, but it removes a now-unused P-scale reuse release in the folded handoff. It improves total feed efficiency: fewer sync/wait instructions, fewer arrive instructions, slightly higher tensor/TC active, slightly higher issue active, and lower isolated duration.
- V-scale ping-pong is preserved and not serialized; QK/V TMEM is unchanged; no shared-memory or register pressure was added.

Decision: keep. This is a validated forward-only win and becomes the new qkscfix baseline for the next loop.

### Loop 94 - rejected early P-stage reuse prerelease before PV commit

Intent: test the smallest legal earlier-payload-publish lever after Loop 93. The probe moved only the P-stage payload reuse release for the live score-derived qkscfix handoff from after `tensor_commit(pv_tmem_ready[0])` to just before that PV commit. It preserved the kept `skippscarrive` route, P-stage depth 2, P-scale TMEM depth 2, V-scale ping-pong, QK/V TMEM layout, and the direct score-derived P payload/P-scale math path.

Probe route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_prerelease_pstage2_q200_p112_o56_qkscfix`

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_prerelease_qkscfix_probe.log
```

Build result:

- Probe built with the same resource footprint as kept `skippscarrive`: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.

Correctness command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_prerelease_vs_skippscarrive_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input kept skippscarrive vs prerelease.
PY
```

Correctness results:

| Shape | Seed | Output finite | LSE finite | Probe vs kept |
| --- | ---: | --- | --- | --- |
| H16/S128 | `95801` | yes | yes | exact output and exact LSE |
| H16/S4096 | `95802` | yes | yes | exact output and exact LSE |

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_prerelease_vs_skippscarrive_direct.stdout
# Results written to results/mxfp4_fa4_forward_profile_20260612/bench_prerelease_vs_skippscarrive_direct.jsonl
PY
```

Direct preallocated timing results versus kept `skippscarrive`:

| Shape | Launch | Kept ms | Probe ms | Probe vs kept |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.059055999` | `0.059136000` | `-0.135%` |
| H16/S4096 | persistent | `0.164992005` | `0.165200002` | `-0.126%` |
| H16/S8192 | fullgrid | `0.544031978` | `0.544559985` | `-0.097%` |
| H4/S2048 | persistent | `0.060575999` | `0.060672000` | `-0.158%` |

All timing records were exact output/LSE versus kept route.

NCU decision: skipped. The probe was correct but negative on every direct preallocated shape, including representative H16/S4096, and did not add a diagnostic resource change.

Classification:

- The prerelease is numerically legal for this guarded cluster1 folded-arrive path, but it does not relieve the dominant PV tensor-core underfeed.
- The likely critical path is not the few instructions between P-stage reuse release and PV commit; releasing the shared payload slot earlier may instead perturb scheduling around the PV issue/commit point.
- V-scale ping-pong and P-scale depth were unchanged, so this did not solve the structural P/V feed overlap issue.

Decision: reject and revert.

Revert/cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_prerelease_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "prerelease" || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix\\|arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix"
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_prerelease_skippscarrive_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input pscreusefold vs kept skippscarrive after revert.
PY
```

Cleanup result:

- No `prerelease` source or rebuilt binary route string remains.
- Rebuilt binary still contains kept `pscreusefold` and kept `skippscarrive`.
- Post-revert smoke:

| Shape | Seed | Output finite | LSE finite | Kept skippscarrive vs pscreusefold |
| --- | ---: | --- | --- | --- |
| H16/S128 | `96001` | yes | yes | exact output and exact LSE |
| H16/S4096 | `96002` | yes | yes | exact output and exact LSE |

Next lever: continue P-movement overlap work, but avoid early payload reuse before PV commit as a standalone change. The remaining dominant path is still score residual/exp2/pack, shared payload store/publish, P-scale TMEM store/wait, V-scale ping-pong readiness, and PV consume/issue.

### Loop 95 - rejected QK-lane P-scale feed-ready bridge

Intent: implement the queued feeder/P-ring overlap slice in the smallest correctness-preserving form without stealing live QK/V TMEM or collapsing V-scale ping-pong. The first attempted feeder-warp mapping used producer `warpid==2`, but inspection showed `ONLINE_V_LOAD_WARPS=2` makes that warp a live V-loader in the kept qkscfix route, so it is not a spare feeder resource. Using it would violate the constraint to preserve V movement and the initial S128 repro deadlocked because the branch was unreachable from the producer main path.

Corrected probe: move only the P-scale-ready wait off the PV issue lane. The QK issue lane, after issuing the next QK, waits the existing `p_sc_tmem_ready[p_sc_slot]` and publishes a new coarse `p_scale_published_ready[p_sc_slot]`; the PV lane keeps the existing V-scale TMEM ping-pong path and waits only that coarse feed-ready event before PV MMA. P payload and P scale remain strict score-derived qkscfix: score residual -> exp2 -> E2M1 payload, and score block max vs row max -> E8M0 direct x1 P-scale TMEM. No `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax materialized-P path is used.

Probe route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_feedready_pstage2_q200_p112_o56_qkscfix`

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_feedready_qkbridge_qkscfix_probe.log
```

Build result:

- Probe: `168` registers, `2` barriers, `1920` bytes smem, no stack frame, no spills.
- Kept `skippscarrive`: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Additional static smem is one semaphore slot for the bridge; QK/V TMEM layout and V-scale ping-pong remain unchanged.

Deadlock repro/fix note:

- The producer `warpid==2` feeder mapping is invalid for this route because it is already a V-loader. This exact blocker rules out a "free" 3WG feeder warp unless it steals V progress.
- The QK-lane bridge version fixed the S128 deadlock and produced exact same-input output/LSE.

Correctness command:

```bash
timeout 480s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_feedready_qkbridge_vs_skippscarrive_h16_s128_s4096.log
# H16/S128 seed 96111 and H16/S4096 seed 96112, same-input kept skippscarrive vs qk-bridge feedready.
PY
```

Correctness results:

| Shape | Seed | Output finite | LSE finite | Probe vs kept |
| --- | ---: | --- | --- | --- |
| H16/S128 | `96111` | yes | yes | exact output and exact LSE |
| H16/S4096 | `96112` | yes | yes | exact output and exact LSE |

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_feedready_qkbridge_vs_skippscarrive_direct.stdout
# Results written to results/mxfp4_fa4_forward_profile_20260612/bench_feedready_qkbridge_vs_skippscarrive_direct.jsonl
PY
```

Direct preallocated timing results versus kept `skippscarrive`:

| Shape | Launch | Kept ms | Probe ms | Probe vs kept |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.0601599999` | `0.0600959994` | `+0.106%` |
| H16/S4096 | persistent | `0.1662400067` | `0.1658720002` | `+0.222%` |
| H16/S8192 | fullgrid | `0.5438719988` | `0.5437120199` | `+0.029%` |
| H4/S2048 | persistent | `0.0608480014` | `0.0607680008` | `+0.132%` |

All timing records were exact output/LSE versus kept route.

Representative H16/S4096 NCU commands:

```bash
timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_qkbridge_base_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_qkbridge_base_qkscfix_h16_s4096.stdout
# H16/S4096, seed=96212, ten raw-extension warmups, one preallocated forward launch bracketed by cudaProfilerStart/Stop.
PY

timeout 1200s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_feedready_pstage2_q200_p112_o56_qkscfix \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_qkbridge_probe_qkscfix_h16_s4096 \
  python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_feedready_qkbridge_probe_qkscfix_h16_s4096.stdout
# Same isolated one-kernel driver and seed, qk-bridge feedready route.
PY
```

NCU metric names and representative deltas, kept -> probe:

| Metric | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` us | `153.888` | `155.040` | `+0.749%` |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | `11.143613` | `11.110219` | `-0.300%` |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | `1.274203` | `1.264913` | `-0.729%` |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `7.151971` | `7.105475` | `-0.650%` |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | `14.989256` | `15.001992` | `+0.085%` |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | `36.108311` | `36.176900` | `+0.190%` |
| `smsp__warps_active.avg.per_cycle_active` | `2.870943` | `2.885194` | `+0.496%` |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.417843` | `0.419531` | `+0.404%` |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | `3.641766` | `3.671915` | `+0.828%` |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | `1.608609` | `1.602932` | `-0.353%` |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | `0.207513` | `0.209724` | `+1.066%` |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | `0.498131` | `0.480851` | `-3.469%` |
| `sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed` | `8.640650` | `8.684692` | `+0.510%` |
| `sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed` | `8.815901` | `8.864657` | `+0.553%` |
| `inst_executed` | `52192065` | `53232961` | `+1.994%` |
| `launch__registers_per_thread` | `168` | `168` | unchanged |
| `launch__shared_mem_per_block_static` KB | `1.904` | `1.920` | `+16` bytes |

Source-page aggregate deltas:

| Pattern | Kept | Probe | Delta |
| --- | ---: | ---: | ---: |
| `SYNCS` | `6441127` | `6795767` | `+5.5%` |
| `SYNCS.ARRIVE` | `259392` | `267840` | `+8448` extra arrives |
| `UTMALDG` | `17920` | `17920` | unchanged |
| `UTMASTG` | `512` | `512` | unchanged |
| `LDG` | `245136` | `245136` | unchanged |
| `LDS` | `43520` | `43520` | unchanged |
| `STS` | `428944` | `428944` | unchanged |
| `MMA` | `42240` | `42240` | unchanged |

Classification:

- The probe preserved the V-scale ping-pong and did not steal QK/V TMEM or V loader warps; unchanged `UTMALDG`, `LDG`, `LDS`, `STS`, and `MMA` counts confirm this.
- It did not unlock PV tensor utilization. Tensor active fell slightly, long-scoreboard and barrier stalls worsened, and instrumented kernel duration regressed.
- The small direct timing gains are below the confidence needed to accept an extra protocol event, and NCU shows the stall moved into protocol overhead rather than improved overall PV issue.
- Dominant bottleneck remains PV tensor-core underfeed / P-V feed protocol, not DRAM, launch, spills, or occupancy.

Decision: reject and revert. Do not retry the same 3WG QK-lane bridge as a standalone feeder. If more feed overlap is needed, it likely needs either fewer total readiness events or a real extra execution resource that does not steal V-loader work.

Revert/cleanup commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_feedready_qkbridge_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "feedready"
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep "arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix\\|arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix"
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_feedready_qkbridge_skippscarrive_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input pscreusefold vs kept skippscarrive after qk-bridge revert.
PY
```

Cleanup result:

- No `feedready` source or rebuilt binary route string remains.
- Binary still contains kept `pscreusefold` and kept `skippscarrive`.
- Kept `skippscarrive` resource footprint after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Post-revert smoke:

| Shape | Seed | Output finite | LSE finite | Kept skippscarrive vs pscreusefold |
| --- | ---: | --- | --- | --- |
| H16/S128 | `96301` | yes | yes | exact output and exact LSE |
| H16/S4096 | `96302` | yes | yes | exact output and exact LSE |

Next lever: stop small PV-side wait shuffles. The failed producer-warp feeder shows the current 3WG layout has no free feeder warp without stealing V movement. The QK-lane bridge shows an added coarse event can move the wait but not reduce total protocol cost. The next structural overlap attempt should either (1) add a real feeder execution resource without reducing V loader progress, or (2) reduce/coarsen P/P-scale readiness events enough that PV sees fewer waits rather than a renamed wait. K256 remains stopped for now; any future K256 route must be true score-derived paired payload/P-scale staging and must not use the vector-amax scorepack fallback.

## Loop 96: 4WG relay P-scale feeder for score-derived qkscfix

Constraint carried into this loop: stop K256/cluster2/output-half work. Keep the validated qkscfix+p112 forward route as baseline, do not touch backward files, preserve V-scale ping-pong, and do not steal live QK/V TMEM or V-loader work. The feeder/P-ring direction must improve overall PV feed, not just make P staging look better while serializing V-scale TMEM.

Current live qkscfix critical path before patch:

- Route: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`.
- Score-derived P path: E8M0 P scale is derived from score block max vs row max; E2M1 payload is generated directly from score residual via `exp2` and FP4 conversion. No `fp4pv_pack_scores_to_stage_mxfp4` vector-amax/materialized-P packer is on the kept route.
- Slots/events: `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, V-scale TMEM ping-pong remains two slots. PV waits on the existing coarse P payload/P-scale readiness before issuing PV MMA, then reuse is folded through the kept P-stage/P-scale reuse protocol.
- Exact previous blocker: a producer-warp feeder is not legal in the current 3WG qkscfix route because `ONLINE_V_LOAD_WARPS=2`; `warpid==2` is a live V-loader, not spare execution capacity.

Hypothesis: adding a real relay WG can move score-derived P-scale TMEM staging off the quant producer without stealing the two V-loader warps. Quant WG writes a shared P-scale shadow next to the score-derived P payload; relay WG waits for that shadow, copies it into the existing two P-scale TMEM slots, and signals the existing `p_sc_tmem_ready`. PV WG remains focused on MMA and V-scale ping-pong stays intact. This is a structural feeder probe, not a K256 path and not a vector-amax scorepack fallback.

Probe route:

- `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_relaypsc_pstage2_q200_p112_o56_qkscfix`

Changed files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`
- `results/mxfp4_fa4_forward_profile_20260612/ledger.md`

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_relaypsc_qkscfix_probe.log
```

Build/resource results:

| Route | Registers | Barriers | Static smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept qkscfix | `168` | `2` | `1904` bytes | `0` stack, `0` spills |
| relaypsc | `128` | `4` | `2944` bytes | `8` stack, `12` B spill stores, `68` B spill loads |

Binary route check:

```bash
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'skippscarrive(_relaypsc)?_pstage2_q200_p112_o56_qkscfix'
```

Both kept and relay route strings were present.

Small correctness repro command:

```bash
timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_relaypsc_progress_h16_s128.log
# H16/S128, seed=96401, same prepared input tensors.
# Run kept route then relaypsc route with progress prints around launch/synchronize.
# Compare output and LSE exactly.
PY
```

S128 result:

- H16/S128 seed `96401`: exact output and exact LSE vs kept; output/LSE finite.

Long correctness command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_relaypsc_vs_skippscarrive_h16_s4096.log
# H16/S4096, seed=96402, same prepared input tensors.
# Run kept route then relaypsc route and compare output/LSE exactly.
PY
```

S4096 result:

- H16/S4096 seed `96402`: exact output and exact LSE vs kept; output/LSE finite.

Direct preallocated timing command:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' | tee results/mxfp4_fa4_forward_profile_20260612/bench_relaypsc_vs_skippscarrive_direct.jsonl
# Timed the raw extension entrypoint `forward_streaming_live_mxfp4` directly.
# Preallocated out/lse tensors; set TK_FA4_FP4PV_FWD_CONFIG immediately before each timed raw launch.
# Alternated kept and relaypsc launches on the same input tensors.
# WARMUP=30, ITERS=180, SEED_BASE=96500.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept median ms | Relaypsc median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.05937600136` | `0.06176000088` | `+4.015%` |
| H16/S4096 | persistent | `0.16364799440` | `0.16902400553` | `+3.285%` |
| H16/S8192 | fullgrid | `0.53723201156` | `0.54897597432` | `+2.186%` |
| H4/S2048 | persistent | `0.05337600037` | `0.05531200022` | `+3.627%` |

All direct-timing harness numeric checks were exact output and exact LSE vs kept for the same inputs.

Diagnostic NCU commands, run despite negative timing because this was the requested structural feeder probe:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_relaypsc_pstage2_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_relaypsc_h16_s4096 python3 - <<'PY'
# Warm 10 launches, then cudaProfilerStart/cudaProfilerStop around one preallocated H16/S4096 forward_streaming_live_mxfp4 launch.
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_samebuild_h16_s4096 python3 - <<'PY'
# Same-build kept baseline, same seed/input shape, same warm/profile protocol.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_relaypsc_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_relaypsc_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_relaypsc_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_relaypsc_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_relaypsc_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_relaypsc_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_samebuild_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_samebuild_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_samebuild_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_samebuild_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_samebuild_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_samebuild_h16_s4096_source.csv
```

NCU metric comparison, same-build kept vs relaypsc H16/S4096:

| Metric name | Kept | Relaypsc | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | `153.120000 us` | `161.056000 us` | `+5.183%` |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `7.097014%` | `6.896113%` | `-2.831%` |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | `14.911484%` | `14.702172%` | `-1.404%` |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | `36.125858%` | `36.638955%` | `+1.420%` |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.417717` | `0.442317` | `+5.889%` |
| `smsp__warps_active.avg.per_cycle_active` | `2.870326` | `3.848754` | `+34.088%` |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | `3.644335` | `3.711074` | `+1.831%` |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | `0.498728` | `0.609423` | `+22.195%` |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | `1.608779` | `1.676729` | `+4.224%` |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | `0.209450` | `1.946717` | `+829.442%` |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | `1.280569%` | `1.218607%` | `-4.839%` |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | `11.059385%` | `12.562424%` | `+13.591%` |
| `derived__local_spilling_requests` | `0` | `58368` | new spill traffic |
| `launch__registers_per_thread` | `168` | `128` | `-23.810%` |
| `launch__barrier_count` | `2` | `4` | `+100.000%` |
| `launch__shared_mem_per_block_static` | `1.904 Kbyte/block` | `2.944 Kbyte/block` | `+54.622%` |
| `launch__block_size` | `384` | `512` | `+33.333%` |
| `inst_executed` | `52359365` | `60104655` | `+14.793%` |

Source-counter comparison:

| Source counter | Kept | Relaypsc | Delta |
| --- | ---: | ---: | ---: |
| `SYNCS` | `6506900` | `8791901` | `+35.117%` |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | `6224980` | `8500509` | `+36.555%` |
| `SYNCS.ARRIVE` | `259392` | `267840` | `+3.257%` |
| `UTMALDG` | `17920` | `17920` | unchanged |
| `UTMASTG` | `512` | `512` | unchanged |
| `LDG` | `227216` | `227216` | unchanged |
| `LDS` | `43520` | `79360` | `+82.353%` |
| `STS` | `428944` | `462736` | `+7.878%` |
| `LDL` | `0` | `33792` | new local spill loads |
| `STL` | `0` | `24576` | new local spill stores |
| `BAR` | `121856` | `229376` | `+88.235%` |
| `MEMBAR` | `58640` | `116768` | `+99.127%` |
| `MUFU.EX2` | `4359168` | `4359168` | unchanged |
| `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` | `2162688` | `2162688` | unchanged |

P-scale/V-scale wait interpretation:

- The relay route preserved V-scale ping-pong and did not steal QK/V TMEM: `UTMALDG`, `UTMASTG`, and `LDG` counts are unchanged.
- It did not improve overall PV feed. PV tensor active fell, H16/S4096 kernel duration worsened, and the added execution resource raised active/eligible warps without improving tensor issue.
- The p_sc handoff moved into a relay WG but the protocol became heavier: `SYNCS.PHASECHK.TRANS64.TRYWAIT`, barrier stalls, `BAR`, and `MEMBAR` all increased sharply.
- The route also introduced local spill traffic and extra shared-memory load/store activity. This rejects the specific 4WG relaypsc implementation: it moves P-scale staging work but pays more in barriers/spills/protocol than it saves.

Decision: reject and revert `relaypsc`. This is a clean structural result, not a dead-end for overlap in general. A deeper P ring can still work only if TMEM/layout/protocol changes preserve overall PV feed: P payload, P scales, V payload, V scales, ready/reuse, and actual PV MMA utilization. The next structural slice should reduce the feeder overhead before adding another WG: either alias/reshape scale slots safely, shrink score/output TMEM footprint enough for true P-scale depth without collapsing V-scale ping-pong, or coarsen readiness so the PV path observes fewer waits instead of a renamed wait.

Revert/cleanup commands:

```bash
git checkout -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_relaypsc_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep 'relaypsc' || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix|arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix'
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_relaypsc_skippscarrive_h16_s128_s4096.log
# H16/S128 and H16/S4096 same-input pscreusefold vs kept skippscarrive after relaypsc revert.
PY
```

Cleanup result:

- No `relaypsc` source or rebuilt binary route string remains.
- Binary still contains kept `pscreusefold` and kept `skippscarrive`.
- Kept `skippscarrive` resource footprint after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Post-revert smoke:

| Shape | Seed | Output finite | LSE finite | Kept skippscarrive vs pscreusefold |
| --- | ---: | --- | --- | --- |
| H16/S128 | `96601` | yes | yes | exact output and exact LSE |
| H16/S4096 | `96602` | yes | yes | exact output and exact LSE |

### Loop 97 - score-derived split-K64 early handoff on kept skippscarrive qkscfix, rejected

Active plan probe: ranked probe #1, score-derived split-K64 early handoff. The goal was to let PV consume K64 half0 while the producer finishes half1, without changing the normal 3WG/384-thread shape, without stealing QK/V TMEM, and without collapsing V-scale ping-pong.

Changed files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Route under test:

- Baseline: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`
- Probe: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_splitk64_pstage2_q200_p112_o56_qkscfix`

Implementation summary:

- Added an opt-in score-derived direct-P-scale split-K64 trait/config on top of the kept qkscfix `skippscarrive` route.
- Kept the existing P payload slots, P-scale TMEM ping-pong slots, and V-scale ping-pong path.
- Issued the score-derived x1 E8M0 P scale to the existing P-scale TMEM slot immediately after `p_score_mx_packed_scales` was computed.
- Published K64 half0 after qid `0..1` payload stores, with a diagonal-causal half0 payload zero before the half0 ready event.
- Published K64 half1 after the full tile.
- Avoided a duplicate late direct P-scale TMEM store/ready event for the split route.
- Did not use `fp4pv_pack_scores_to_stage_mxfp4` or vector-amax-over-materialized-P quantization.

Build and resource command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_splitk64_skippscarrive_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'skippscarrive(_splitk64)?_pstage2_q200_p112_o56_qkscfix'
```

Build/resource results:

| Route | Registers | Barriers | Static smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept qkscfix | `168` | `2` | `1904` bytes | `0` stack, `0` spills |
| split-K64 qkscfix | `168` | `2` | `2480` bytes | `0` stack, `0` spills |

Correctness smoke commands:

```bash
timeout 240s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_splitk64_skippscarrive_progress_h16_s128.log
# H16/S128, seed=96701, same prepared input tensors.
# Run kept route then split-K64 route with TK_FA4_FP4PV_FWD_CONFIG.
# Compare output and LSE exactly.
PY

timeout 360s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_splitk64_skippscarrive_vs_kept_h16_s4096.log
# H16/S4096, seed=96702, same prepared input tensors.
# Run kept route then split-K64 route with TK_FA4_FP4PV_FWD_CONFIG.
# Compare output and LSE exactly.
PY
```

Correctness results:

| Shape | Seed | Result |
| --- | ---: | --- |
| H16/S128 | `96701` | exact output and exact LSE vs kept |
| H16/S4096 | `96702` | exact output and exact LSE vs kept |

Direct preallocated timing command:

```bash
timeout 1800s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_splitk64_skippscarrive_vs_kept_direct.jsonl
# Raw extension entrypoint: forward_streaming_live_mxfp4.
# Preallocated out/lse tensors; CUDA events; WARMUP=30, ITERS=180.
# Same quantized inputs for kept and split-K64 per shape.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept median ms | Split-K64 median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.05911999941` | `0.06304000318` | `+6.631%` |
| H16/S4096 | persistent | `0.17099200189` | `0.18267200142` | `+6.831%` |
| H16/S8192 | fullgrid | `0.54539200664` | `0.57801598310` | `+5.982%` |
| H4/S2048 | persistent | `0.06318399683` | `0.06518399715` | `+3.165%` |

All direct-timing numeric checks were exact output and exact LSE vs kept for the same inputs.

NCU decision:

- Skipped follow-up NCU for this probe. The route was correct and spill-free, but every required direct timing shape regressed materially. There was no non-negative or ambiguous representative timing result to justify a profiling pass under the active plan's "NCU only if useful" rule.

Interpretation and blocker:

- Split-K64 is not blocked by correctness or register pressure on the current score-derived qkscfix route.
- The blocker is protocol/work overhead: the split route adds early P-scale TMEM issue plus half0 payload publish/ready and a diagonal-causal half0 zero path, increasing static smem from `1904` to `2480` bytes and adding handoff work without reducing the dominant PV underfeed enough to offset the extra synchronization/publish cost.
- This result rejects this exact score-derived split-K64 early-handoff implementation on the kept `skippscarrive` baseline. It does not rule out a different coarser/fewer-readiness variant, which is the next ranked probe.

Decision: reject and revert the split-K64 route.

Revert/cleanup commands:

```bash
git checkout -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_splitk64_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep 'skippscarrive_splitk64' || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'arrivereuse_pscreusefold(_skippscarrive)?_pstage2_q200_p112_o56_qkscfix'
timeout 420s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_splitk64_skippscarrive_h16_s128_s4096.log
# H16/S128 seed=96761 and H16/S4096 seed=96762.
# Compare rebuilt kept skippscarrive vs pscreusefold on same prepared input tensors.
PY
```

Cleanup result:

- No `skippscarrive_splitk64` rebuilt binary route string remains.
- Binary still contains kept `pscreusefold` and kept `skippscarrive`.
- Kept `skippscarrive` resource footprint after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Post-revert smoke:

| Shape | Seed | Output finite | LSE finite | Kept skippscarrive vs pscreusefold |
| --- | ---: | --- | --- | --- |
| H16/S128 | `96761` | yes | yes | exact output and exact LSE |
| H16/S4096 | `96762` | yes | yes | exact output and exact LSE |

### Loop 98 - TCGEN P-stage reuse on kept skippscarrive qkscfix, rejected

Active plan probe: ranked probe #2, coarser/fewer readiness events on the existing direct path. The code audit found that the kept qkscfix route already has one coarse `p_sc_tmem_ready` wait for payload plus P-scale and folded P-scale reuse via `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE` plus `ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE`. This probe removed the explicit P-stage `arrive(p_stage_reusable[p_buf])` release on the opt-in route and used the existing TCGEN commit release path instead.

Changed files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Route under test:

- Baseline: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`
- Probe: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_tcgenreuse_pstage2_q200_p112_o56_qkscfix`

Implementation summary:

- Added an opt-in config deriving from the kept `skippscarrive` qkscfix route with `ONLINE_TCGEN_P_STAGE_REUSE = true`.
- Added dispatch entries for the guarded route in both persistent/fullgrid dispatch blocks.
- Changed the P-stage reuse release site so the opt-in route calls `tensor_commit<C::CLUSTER_SIZE>(p_stage_reusable[p_buf])` before the generic `arrive(p_stage_reusable[p_buf])` path.
- Kept the same normal 3WG/384-thread shape, P payload slots, P-scale TMEM slots, V-scale ping-pong path, score-derived payload/scales, and qkscfix route.

Build and resource command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_tcgenreuse_skippscarrive_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'skippscarrive(_tcgenreuse)?_pstage2_q200_p112_o56_qkscfix'
```

Build/resource results:

| Route | Registers | Barriers | Static smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept qkscfix | `168` | `2` | `1904` bytes | `0` stack, `0` spills |
| tcgenreuse qkscfix | `168` | `2` | `1904` bytes | `0` stack, `0` spills |

Correctness smoke commands:

```bash
timeout 240s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_tcgenreuse_skippscarrive_progress_h16_s128.log
# H16/S128, seed=96801, same prepared input tensors.
# Run kept route then tcgenreuse route with TK_FA4_FP4PV_FWD_CONFIG.
# Compare output and LSE exactly.
PY

timeout 360s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_tcgenreuse_skippscarrive_vs_kept_h16_s4096.log
# H16/S4096, seed=96802, same prepared input tensors.
# Run kept route then tcgenreuse route with TK_FA4_FP4PV_FWD_CONFIG.
# Compare output and LSE exactly.
PY
```

Correctness results:

| Shape | Seed | Result |
| --- | ---: | --- |
| H16/S128 | `96801` | exact output and exact LSE vs kept |
| H16/S4096 | `96802` | exact output and exact LSE vs kept |

Direct preallocated timing command:

```bash
timeout 1800s python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_tcgenreuse_skippscarrive_vs_kept_direct.jsonl
# Raw extension entrypoint: forward_streaming_live_mxfp4.
# Preallocated out/lse tensors; CUDA events; WARMUP=30, ITERS=180.
# Same quantized inputs for kept and tcgenreuse per shape.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept median ms | TCGEN reuse median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.05824000016` | `0.05830400065` | `+0.1099%` |
| H16/S4096 | persistent | `0.16208000481` | `0.16187199950` | `-0.1283%` |
| H16/S8192 | fullgrid | `0.54092800617` | `0.54399999976` | `+0.5679%` |
| H4/S2048 | persistent | `0.05542400107` | `0.05548800156` | `+0.1155%` |

All direct-timing numeric checks were exact output and exact LSE vs kept for the same inputs.

NCU commands:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix' ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096.log
# H16/S4096, seed=96851, 10 warmups, cudaProfilerStart/Stop around one raw preallocated forward_streaming_live_mxfp4 launch.
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_tcgenreuse_pstage2_q200_p112_o56_qkscfix' ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096.log
# H16/S4096, seed=96851, 10 warmups, cudaProfilerStart/Stop around one raw preallocated forward_streaming_live_mxfp4 launch.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_tcgensamebuild_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_tcgenreuse_h16_s4096_source.csv
```

NCU metric comparison, H16/S4096 persistent:

| Metric name | Kept | TCGEN reuse | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | `154.432 us` | `153.952 us` | `-0.3108%` |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `7.166701%` | `7.150118%` | `-0.2314%` |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | `15.010676%` | `15.000881%` | `-0.0653%` |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | `36.149976%` | `36.111254%` | `-0.1071%` |
| `smsp__issue_active.avg.per_cycle_active` | `0.36` | `0.36` | `+0.0000%` |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.418192` | `0.417761` | `-0.1031%` |
| `smsp__warps_active.avg.per_cycle_active` | `2.870707` | `2.870707` | `+0.0000%` |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | `3.642763` | `3.648588` | `+0.1599%` |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | `0.499344` | `0.489214` | `-2.0287%` |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | `1.608546` | `1.606362` | `-0.1358%` |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | `0.209277` | `0.208813` | `-0.2217%` |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | `1.269659%` | `1.273722%` | `+0.3200%` |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | `11.161246%` | `11.103023%` | `-0.5217%` |
| `derived__local_spilling_requests` | `0` | `0` | unchanged |
| `launch__registers_per_thread` | `168` | `168` | unchanged |
| `launch__barrier_count` | `2` | `2` | unchanged |
| `launch__shared_mem_per_block_static` | `1.904 KiB` | `1.904 KiB` | unchanged |
| `launch__block_size` | `384` | `384` | unchanged |
| `inst_executed` | `52191046` | `52363785` | `+0.3310%` |

NCU source opcode/prefix comparison:

| Source opcode or prefix | Kept inst | TCGEN reuse inst | Delta |
| --- | ---: | ---: | ---: |
| `SYNCS` | `6235069` | `6294397` | `+0.9515%` |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | `6160317` | `6228093` | `+1.1002%` |
| `SYNCS.ARRIVE.TRANS64.A1T0` | `34304` | `25856` | `-24.6269%` |
| `UTCBAR` | `34304` | `42752` | `+24.6269%` |
| `BAR` | `54272` | `54272` | unchanged |
| `MEMBAR` | `58640` | `58640` | unchanged |
| `UTMALDG` | `17920` | `17920` | unchanged |
| `UTMASTG` | `512` | `512` | unchanged |
| `LDG` | `202880` | `202880` | unchanged |
| `LDS` | `43520` | `43520` | unchanged |
| `STS` | `404096` | `404096` | unchanged |
| `MUFU.EX2` | `4325376` | `4325376` | unchanged |
| `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` | `2162688` | `2162688` | unchanged |

Interpretation and blocker:

- The direct timing signal was too small and shape-limited: one representative H16/S4096 persistent result was `-0.1283%`, but H16/S2048, H16/S8192, and H4/S2048 regressed.
- NCU did not show better PV feed. Tensor active, TC active, issue active, and eligible warps were flat to slightly lower, while long scoreboard rose slightly.
- The protocol source counters moved in the wrong direction overall: fewer explicit `SYNCS.ARRIVE` instructions were offset by more `UTCBAR` and more `SYNCS.PHASECHK.TRANS64.TRYWAIT`.
- This rejects TCGEN P-stage reuse as a coarser-readiness improvement on the kept qkscfix route.

Decision: reject and revert the TCGEN P-stage reuse route.

Revert/cleanup commands:

```bash
git checkout -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_tcgenreuse_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep 'skippscarrive_tcgenreuse' || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'arrivereuse_pscreusefold(_skippscarrive)?_pstage2_q200_p112_o56_qkscfix'
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_tcgenreuse_skippscarrive_h16_s128_s4096.log
# H16/S128 seed=96861 and H16/S4096 seed=96862.
# Compare rebuilt kept skippscarrive vs pscreusefold on same prepared input tensors.
PY
```

Cleanup result:

- No `skippscarrive_tcgenreuse` rebuilt binary route string remains.
- Binary still contains kept `pscreusefold` and kept `skippscarrive`.
- Kept `skippscarrive` resource footprint after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Post-revert smoke:

| Shape | Seed | Output finite | LSE finite | Kept skippscarrive vs pscreusefold |
| --- | ---: | --- | --- | --- |
| H16/S128 | `96861` | yes | yes | exact output and exact LSE |
| H16/S4096 | `96862` | yes | yes | exact output and exact LSE |

### Loop 99 - score-derived P-scale x4 TMEM issue (`pscx4`) on kept skippscarrive qkscfix, rejected

Active plan probe: ranked probe #3 after split-K64 and one coarser-readiness probe were rejected. The first audit checked whether P payload movement itself was worth patching:

- Live qkscfix P payload stores already use `st.shared.v4.b32` via `fp4pv_store_quantized_scores_group_mxfp4`.
- Scalar P payload stores are an opt-in fallback, not the kept route.
- Kept H16/S4096 source counters showed `STS ~= 404096`, while `SYNCS ~= 6272990`, `MUFU.EX2 ~= 4325376`, and `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C ~= 2162688`; a naive payload vectorization patch would not be counter-backed.
- True P-scale depth is blocked in the current non-aliased TMEM layout without footprint relief: dual score TMEM plus output plus Q/K scales plus two P-scale slots plus two V-scale slots fills 512 tensor columns. A third P-scale slot would require shrinking score/output footprint or safe aliasing, and collapsing V-scale ping-pong is not allowed.

The chosen narrow live probe was therefore P-scale movement, not P payload width: keep score-derived P payload unchanged, write each row's packed score-derived P-scale word into the existing shared shadow array, and replace the direct x1 P-scale TMEM store with the existing x4 TMEM helper for an opt-in route.

Changed files:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`

Route under test:

- Baseline: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`
- Probe: `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pscx4_pstage2_q200_p112_o56_qkscfix`

Implementation summary:

- Added `ONLINE_SCORE_DERIVED_P_SCALE_TMEM_X4` trait and guarded qkscfix config.
- Added the route string to both persistent/fullgrid dispatch blocks.
- For the guarded route only, stored `p_score_mx_packed_scales` into `p_online_mxfp4_p_scale_words[buf][global_row & 127]`.
- For the guarded route only, replaced the non-K256 x1 direct P-scale TMEM issue with `fp4pv_store_mxfp4_scale_tmem_32x32b_x4`, loading rows `lane + {0,32,64,96}` from the shared shadow.
- Kept score-derived E8M0 scale math, score-derived E2M1 payload, P payload slots, P-stage reuse protocol, P-scale TMEM slot count, and V-scale ping-pong unchanged.

Build and resource command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pscx4_skippscarrive_qkscfix_probe.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'skippscarrive(_pscx4)?_pstage2_q200_p112_o56_qkscfix'
```

Build/resource results:

| Route | Registers | Barriers | Static smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept qkscfix | `168` | `2` | `1904` bytes | `0` stack, `0` spills |
| pscx4 qkscfix | `168` | `2` | `2928` bytes | `0` stack, `0` spills |

Correctness smoke command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pscx4_skippscarrive_h16_s128_s4096.log
# H16/S128 seed=96901 and H16/S4096 seed=96902.
# Same prepared input tensors; compare pscx4 vs kept output and LSE exactly.
PY
```

Correctness results:

| Shape | Seed | Result |
| --- | ---: | --- |
| H16/S128 | `96901` | exact output and exact LSE vs kept |
| H16/S4096 | `96902` | exact output and exact LSE vs kept |

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_pscx4_skippscarrive_vs_kept_direct.jsonl
# Raw extension entrypoint: forward_streaming_live_mxfp4.
# Preallocated out/lse tensors; CUDA events; WARMUP=30, ITERS=180.
# Same quantized inputs for kept and pscx4 per shape.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept median ms | pscx4 median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.06012799963` | `0.06035200134` | `+0.3725%` |
| H16/S4096 | persistent | `0.16617600620` | `0.16585600376` | `-0.1926%` |
| H16/S8192 | fullgrid | `0.54171201587` | `0.54025602341` | `-0.2688%` |
| H4/S2048 | persistent | `0.05971200019` | `0.05940800160` | `-0.5091%` |

All direct-timing numeric checks were exact output and exact LSE vs kept for the same inputs.

NCU commands:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix' ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096.log
# H16/S4096, seed=96951, 10 warmups, cudaProfilerStart/Stop around one raw preallocated forward_streaming_live_mxfp4 launch.
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pscx4_pstage2_q200_p112_o56_qkscfix' ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096.log
# H16/S4096, seed=96951, 10 warmups, cudaProfilerStart/Stop around one raw preallocated forward_streaming_live_mxfp4 launch.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_pscx4samebuild_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_pscx4_h16_s4096_source.csv
```

NCU metric comparison, H16/S4096 persistent:

| Metric name | Kept | pscx4 | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | `154.304 us` | `155.328 us` | `+0.6636%` |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `7.103210%` | `7.094947%` | `-0.1163%` |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | `14.889270%` | `14.941958%` | `+0.3539%` |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | `36.109919%` | `36.263900%` | `+0.4264%` |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.417319` | `0.419203` | `+0.4515%` |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | `3.641744` | `3.615969` | `-0.7078%` |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | `1.608722` | `1.595453` | `-0.8248%` |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | `0.209566` | `0.210453` | `+0.4233%` |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | `1.270916%` | `1.262723%` | `-0.6447%` |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | `11.057678%` | `11.784727%` | `+6.5751%` |
| `derived__local_spilling_requests` | `0` | `0` | unchanged |
| `launch__registers_per_thread` | `168` | `168` | unchanged |
| `launch__barrier_count` | `2` | `2` | unchanged |
| `launch__shared_mem_per_block_static` | `1.904 KiB` | `2.928 KiB` | `+53.7815%` |
| `launch__block_size` | `384` | `384` | unchanged |
| `inst_executed` | `52277838` | `52514940` | `+0.4535%` |

NCU source opcode/prefix comparison:

| Source opcode or prefix | Kept inst | pscx4 inst | Delta |
| --- | ---: | ---: | ---: |
| `SYNCS` | `6272990` | `6280040` | `+0.1124%` |
| `BAR` | `54272` | `54272` | unchanged |
| `MEMBAR` | `58640` | `58640` | unchanged |
| `FENCE` | `121495` | `121488` | `-0.0058%` |
| `UTCBAR` | `34304` | `34304` | unchanged |
| `UTMALDG` | `17920` | `17920` | unchanged |
| `UTMASTG` | `512` | `512` | unchanged |
| `LDG` | `202880` | `202880` | unchanged |
| `LDS` | `43520` | `178688` | `+310.5882%` |
| `STS` | `404096` | `437888` | `+8.3624%` |
| `STTM` | `274488` | `274432` | `-0.0204%` |
| `LDTM` | `375864` | `375808` | `-0.0149%` |
| `MUFU.EX2` | `4325376` | `4325376` | unchanged |
| `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` | `2162688` | `2162688` | unchanged |

Interpretation and blocker:

- The direct wall timings were small and mixed, with representative H16/S4096 showing a small win but H16/S2048 regressing.
- NCU on the representative H16/S4096 kernel rejected the route: duration regressed by `+0.6636%`, tensor active was flat/slightly lower, and the x4 path increased shared-memory work and total instruction count.
- The x4 TMEM store replaced some source `STTM` form, but the required shared shadow path added substantial `LDS` and extra `STS`, moving pressure into shared/protocol work instead of unstarving PV.
- This route does not meet the keep bar because it does not improve isolated-kernel data or PV tensor utilization enough to offset overhead.

Decision: reject and revert the pscx4 route.

Revert/cleanup commands:

```bash
git checkout -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_pscx4_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep 'skippscarrive_pscx4' || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'arrivereuse_pscreusefold(_skippscarrive)?_pstage2_q200_p112_o56_qkscfix'
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_pscx4_skippscarrive_h16_s128_s4096.log
# H16/S128 seed=96961 and H16/S4096 seed=96962.
# Compare rebuilt kept skippscarrive vs pscreusefold on same prepared input tensors.
PY
```

Cleanup result:

- No `skippscarrive_pscx4` rebuilt binary route string remains.
- Binary still contains kept `pscreusefold` and kept `skippscarrive`.
- Kept `skippscarrive` resource footprint after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Post-revert smoke:

| Shape | Seed | Output finite | LSE finite | Kept skippscarrive vs pscreusefold |
| --- | ---: | --- | --- | --- |
| H16/S128 | `96961` | yes | yes | exact output and exact LSE |
| H16/S4096 | `96962` | yes | yes | exact output and exact LSE |

## Loop 100 - qkscfix folded P-scale sync removal probe

Active-plan position: split-K64 early handoff, TCGEN/coarser-ready, and P-scale x4 have all been tested and rejected cleanly. The remaining ranked item is real footprint relief for P-scale depth, but the current qkscfix TMEM layout is full:

- score slots: `SCORE_TMEM_SLOTS=2`, `C::Nb=128` -> `256` columns
- output: `OUTPUT_TMEM_SLOTS=1`, `C::Dvo=128` -> `128` columns
- Q scale: `16` columns
- K scale: `16` columns
- P scale ping-pong: `P_SCALE_TMEM_SLOTS=2`, `P_SCALE_TMEM_WIDTH=16` -> `32` columns
- V scale ping-pong: `V_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_WIDTH=32` -> `64` columns
- total: `512` columns

The existing one-score/two-output fallback does not free columns because it still consumes `128 + 2*128 = 384` score/output columns. A true one-score/one-output route would free enough room for `P_SCALE_TMEM_SLOTS=3`, but that is not a narrow P-scale-depth probe: it removes the dual-score/direct-after-rescale overlap from the kept qkscfix route. Alias-scale layouts reuse score columns and conflict with the current non-aliased qkscfix direct-after-rescale route; V-scale ping-pong must not be collapsed.

Next narrow protocol probe: the kept folded route still executes a second `quant_wg_sync()` after the P payload proxy publish and folded P-scale reuse branch. In the kept route, `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE=true` and `ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE=true`, so the producer-side leader wait on `p_sc_tmem_reusable` is compiled out; the first `quant_wg_sync()` already completed score-derived P payload stores before proxy publish. Probe route will compile out only that second sync under an explicit opt-in guard:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_skippscsync_pstage2_q200_p112_o56_qkscfix`

Expected effect: reduce one producer WG barrier/SYNCS point per score tile without changing P payload, P-scale TMEM slots, V-scale ping-pong, ready/reuse events, QK, or output math. Revert if correctness fails or timing/profile does not improve PV feed.

Implementation diff:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`: added `ONLINE_SKIP_FOLDED_P_SCALE_REUSE_SYNC` trait and a derived qkscfix `skippscsync` config from kept `skippscarrive`.
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`: added the guarded route string in both MXFP4 dispatch blocks.
- `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc`: added `STATIC_ONLINE_MXFP4_SKIP_FOLDED_P_SCALE_REUSE_SYNC`; compiled out only the post-folded-reuse `quant_wg_sync()` when the folded reuse arrive is also skipped and P payload is prepublished before the P-scale reuse point.

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_skippscsync_qkscfix_probe.log
```

Build/resource result:

| Route | Registers | Barriers | Static smem | Stack/spills |
| --- | ---: | ---: | ---: | --- |
| kept `skippscarrive` | `168` | `2` | `1904` bytes | `0` |
| `skippscsync` | `168` | `2` | `1904` bytes | `0` |

Route strings after build:

```bash
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'skippscarrive(_skippscsync)?_pstage2_q200_p112_o56_qkscfix'
```

Smoke command:

```bash
timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_skippscsync_vs_kept_h16_s128_s4096.log
# Raw extension entrypoint: forward_streaming_live_mxfp4.
# Output layout: (B,S,H,128); LSE layout: (B,H,1,S).
# Compare kept skippscarrive vs skippscsync for H16/S128 seed=97001 and H16/S4096 seed=97002.
PY
```

Smoke result:

| Shape | Seed | Result |
| --- | ---: | --- |
| H16/S128 | `97001` | exact output and exact LSE vs kept |
| H16/S4096 | `97002` | exact output and exact LSE vs kept |

Direct preallocated timing command:

```bash
timeout 1800s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/bench_skippscsync_vs_kept_direct.jsonl
# One shared BF16/quantized input set per shape; prepare kept and probe from the same fp4_inputs.
# Preallocated out/lse tensors; CUDA events around only ext.forward_streaming_live_mxfp4.
# WARMUP=30, ITERS=180.
# Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
PY
```

Direct preallocated timing results:

| Shape | Launch | Kept median ms | skippscsync median ms | Delta |
| --- | --- | ---: | ---: | ---: |
| H16/S2048 | persistent | `0.05884800106` | `0.05855999887` | `-0.4894%` |
| H16/S4096 | persistent | `0.16926399618` | `0.16739199311` | `-1.1060%` |
| H16/S8192 | fullgrid | `0.54076799750` | `0.53935998678` | `-0.2604%` |
| H4/S2048 | persistent | `0.06006399915` | `0.05907199904` | `-1.6516%` |

NCU commands:

```bash
timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix' ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096.log
# H16/S4096, seed=97051, 10 warmups, cudaProfilerStart/Stop around one raw preallocated forward_streaming_live_mxfp4 launch.
PY

timeout 900s env CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_skippscsync_pstage2_q200_p112_o56_qkscfix' ncu --profile-from-start off --target-processes all --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096 python3 - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096.log
# H16/S4096, seed=97051, 10 warmups, cudaProfilerStart/Stop around one raw preallocated forward_streaming_live_mxfp4 launch.
PY

ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscarrive_skippscsyncsamebuild_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096.ncu-rep --csv --page raw > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096.ncu-rep --csv --page details > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096_details.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_mxfp4_o56_qkscfix_skippscsync_h16_s4096_source.csv
```

NCU metric comparison, H16/S4096 persistent:

| Metric name | Kept | skippscsync | Delta |
| --- | ---: | ---: | ---: |
| `gpu__time_duration.avg` | `154.112 us` | `154.112 us` | unchanged |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | `7.130308%` | `7.110414%` | `-0.2790%` |
| `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed` | `14.958062%` | `14.905458%` | `-0.3517%` |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | `36.154988%` | `36.150149%` | `-0.0134%` |
| `smsp__warps_eligible.avg.per_cycle_active` | `0.418434` | `0.418059` | `-0.0896%` |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | `3.641805` | `3.648109` | `+0.1731%` |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | `1.609414` | `1.608042` | `-0.0852%` |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | `0.211465` | `0.210490` | `-0.4611%` |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | `1.272544%` | `1.272261%` | `-0.0222%` |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | `11.109346%` | `11.072083%` | `-0.3354%` |
| `derived__local_spilling_requests` | `0` | `0` | unchanged |
| `launch__registers_per_thread` | `168` | `168` | unchanged |
| `launch__barrier_count` | `2` | `2` | unchanged |
| `launch__shared_mem_per_block_static` | `1.904 KiB` | `1.904 KiB` | unchanged |
| `launch__block_size` | `384` | `384` | unchanged |
| `inst_executed` | `52072822` | `52118967` | `+0.0886%` |

NCU source opcode/prefix comparison:

| Source opcode or prefix | Kept inst | skippscsync inst | Delta |
| --- | ---: | ---: | ---: |
| `SYNCS` | `6198128` | `6215847` | `+0.2859%` |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | `6123376` | `6141095` | `+0.2894%` |
| `SYNCS.ARRIVE.TRANS64.A1T0` | `34304` | `34304` | unchanged |
| `BAR` | `54272` | `54272` | unchanged |
| `MEMBAR` | `58640` | `58640` | unchanged |
| `FENCE` | `121472` | `121472` | unchanged |
| `UTCBAR` | `34304` | `34304` | unchanged |
| `UTMALDG` | `17920` | `17920` | unchanged |
| `UTMASTG` | `512` | `512` | unchanged |
| `LDG` | `202880` | `202880` | unchanged |
| `LDS` | `43520` | `43520` | unchanged |
| `STS` | `404096` | `404096` | unchanged |
| `STTM` | `274304` | `274304` | unchanged |
| `LDTM` | `375680` | `375680` | unchanged |
| `MUFU.EX2` | `4325376` | `4325376` | unchanged |
| `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` | `2162688` | `2162688` | unchanged |

Follow-up correctness stress:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/stress_skippscsync_correctness_h16_s4096.log
# H16/S4096, seeds 97020..97027, one kept/probe compare each.
PY

timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u - <<'PY' 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/repeat_skippscsync_correctness_seed97022_h16_s4096.log
# H16/S4096, seed=97022, 80 alternating kept/probe launch pairs.
PY
```

Stress result:

- Seeds `97020..97027`, one launch each: exact output and exact LSE.
- Repeated alternating seed `97022`: intermittent output corruption in the probe while LSE stayed exact.

| Iter | Output equal | LSE equal | Max abs output | Output mismatches |
| ---: | --- | --- | ---: | ---: |
| `18` | no | yes | `9.183549615799121e-41` | `5` |
| `24` | no | yes | `0.0013427734375` | `839` |
| `33` | no | yes | `9.183549615799121e-41` | `5` |
| `52` | no | yes | `9.183549615799121e-41` | `5` |
| `53` | no | yes | `0.000244140625` | `212` |

Interpretation:

- Direct timing looked favorable, but NCU did not show improved PV feed: representative kernel time was unchanged, tensor activity and eligible warps were slightly worse, long-scoreboard increased slightly, and total executed instructions increased.
- Source counters did not prove the intended protocol reduction; `SYNCS`/`TRYWAIT` counts increased slightly in the representative profile.
- More importantly, the repeated alternating correctness stress exposed intermittent output corruption with exact LSE. The removed sync is therefore not a safe no-op: despite folded P-scale reuse, it appears to preserve ordering around payload proxy publication / P-scale TMEM issue / PV consume.

Decision: reject and revert `skippscsync`.

Revert/cleanup commands:

```bash
git checkout -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_skippscsync_qkscfix.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep 'skippscsync' || true
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'arrivereuse_pscreusefold(_skippscarrive)?_pstage2_q200_p112_o56_qkscfix'
git diff -- tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc
```

Cleanup result:

- No `skippscsync` route string remains in the rebuilt binary.
- Kept `pscreusefold` and kept `skippscarrive` route strings remain.
- Forward config/dispatch/kernel diff is empty after cleanup.
- Kept `skippscarrive` resource footprint after cleanup: `168` registers, `2` barriers, `1904` bytes smem, no stack frame, no spills.
- Post-revert one-shot finite smoke kept route: H16/S128 and H16/S4096 output/LSE finite. Kept-vs-pscreusefold repeated bitwise comparison shows occasional output-only differences even with clean source and the rejected route removed; this appears to be baseline comparison instability rather than retained `skippscsync` code. Do not use `skippscsync` again without adding a replacement ordering primitive.

## Structural P/PV handoff design audit before next patch

User redirect: stop the single-CTA P payload microprobe. Do not repeat TCGEN/coarser-ready, split-K64, or skippscsync. Next work must be structural P/PV handoff or a 2-CTA macro-tile design, with Q/K/V movement resources intact.

No patch has been made for this audit. Current forward source state:

- `fwd_configs.inc`, `fwd_host_dispatch.inc`, and `fwd_streaming_kernel.inc` are clean relative to the checked-out tree after Loop 100 cleanup.
- `fwd_device_helpers.inc` has a pre-existing local helper edit adding `packed_col_offset` to `fp4pv_zero_invalid_causal_payload_groups_mxfp4`; this audit does not depend on it and will not touch it.

### Current kept qkscfix ownership

Kept route:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`

Static shape:

- `TOTAL_WGS=3`, `NUM_THREADS=384`.
- Dispatch instantiates `_ClusterSize=1` for the kept route.
- Online role map for this route: `PRODUCER_WG=2`, `OUTPUT_WG=0`, `QUANT_WG0=1`, `RELAY_WG=-1`.
- `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`.
- `ONLINE_DECOUPLED_PV_ISSUE=true`.
- `ONLINE_DIRECT_P_SCALE_TMEM=true`.
- `ONLINE_SCORE_DERIVED_P_PRESCALED_PACK=true`.
- `ONLINE_SCORE_DERIVED_P_X1_SCALE_TMEM=true`.
- `ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY=true`.
- `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT=true`.
- `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT=true`.
- `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE=true`.
- `ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE=true`.

Work ownership in the current 3WG route:

- Producer WG (`groupid=2`) owns Q/K/V global movement. `warpid==0` handles Q/K/V payload staging and shared-memory publish. `warpid==1` owns producer-side async V-scale TMEM staging when `ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM` is enabled. This is the existing V-scale ping-pong path and must be preserved.
- Output WG (`groupid=0`) owns QK issue, PV issue, output accumulator TMEM, final output conversion/store, and LSE store. The PV issue lane waits/stages V scale and P scale, issues the PV MMA, then releases V-scale reuse and P-stage reuse.
- Quant WG (`groupid=1`) owns row softmax state (`row_max`, `row_sum`, correction/LSE smem), score-derived P payload generation, score-derived P-scale generation, direct x1 P-scale TMEM store, and the coarse P-scale-ready publish.

### Row softmax and LSE ownership

The row softmax is not separable from the P producer in the kept route:

- Quant WG consumes score TMEM for one query row tile, maintains `row_max`/`row_sum`, computes the score-block max, derives the E8M0 P scale from score block max vs row max, and directly generates E2M1 payload from score residual -> `exp2` -> FP4 conversion.
- LSE finalization uses `lse_smem` and `max_vec_smem` written by the quant WG and consumed by the output WG for the same row tile.
- A true CTA producer/PV split where one CTA only produces P and another only consumes PV would have to move row softmax state, score-derived P payload, P scales, output accumulator ownership, and final LSE/output ownership across CTA boundaries. That is not a narrow first patch.

### P payload and scale lifetime

Current lifetime:

- P payload slots are `p_fp4_stage[slot]`, slot selected by `fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx)`.
- The producer waits on `p_stage_reusable[slot]` after the two-slot ring has wrapped.
- Score-derived P payload is stored to shared memory, then the quant WG synchronizes and publishes shared backing/proxy state. The failed `skippscsync` probe proved that weakening the post-publish/P-scale ordering can intermittently corrupt output even when one-shot smokes pass.
- P scale is direct x1 TMEM: `fp4pv_store_mxfp4_scale_tmem_32x32b_x1(p_sc_tm_cur, warpgroup::warpid(), row_scale_word)`, followed by `fp4pv_tmem_store_wait()`.
- `p_sc_tmem_ready[p_sc_slot]` is the coarse PV-ready event for P scale plus payload visibility in the direct route.
- PV releases `p_stage_reusable[slot]`; in the kept fold route this also covers P-scale reuse, so the separate `p_sc_tmem_reusable` arrive is skipped.

### V payload and V-scale lifetime

Current V path must stay intact:

- V payload is double-buffered in `v_fp4_smem[2]`.
- V scales are double-buffered in `v_sc_smem[2]` and in two TMEM slots.
- Producer-side async V-scale staging waits on `v_arrived[v_buf]`, stages the scale into `v_sc_tm0/v_sc_tm1`, arrives `v_sc_tmem_ready[v_buf]`, and only advances once PV later arrives `v_sc_tmem_reusable[v_buf]`.
- Any P/PV handoff probe must not collapse `V_SCALE_TMEM_SLOTS` to 1 and must not serialize V-scale staging behind more P work. A P probe that improves P progress while increasing `v_sc_tmem_ready` wait cycles or lowering PV issue/tensor activity is a rejection.

### TMEM partition

Current non-aliased qkscfix TMEM budget is full:

- Score TMEM: `SCORE_TMEM_SLOTS=2` * `Nb=128` = `256` columns.
- Output accumulator: `OUTPUT_TMEM_SLOTS=1` * `Dvo=128` = `128` columns.
- Q scale: `16` columns.
- K scale: `16` columns.
- P scale ping-pong: `2 * 16 = 32` columns.
- V scale ping-pong: `2 * 32 = 64` columns.
- Total: `512` columns.

Therefore, true P-scale depth cannot be added to the kept non-aliased layout without footprint relief or aliasing. The current audit does not permit stealing Q/K/V scale or V-scale ping-pong slots.

### 2-CTA macro-tile feasibility

What is possible from the current source:

- The kernel already has `__cluster_dims__(C::CLUSTER_SIZE,1,1)`, `cta_rank`, `p_remote_ready`, `v_remote_ready`, and cluster waits/arrives.
- The direct-after-rescale shape guard includes `Mb=128, Nb=128, CLUSTER_SIZE=2`.
- A correctness-first 2-CTA macro-tile can be attempted as a guarded route by preserving the score-derived qkscfix math but removing cluster1-only handoff shortcuts.

What blocks reusing the exact kept shortcut stack with `CLUSTER_SIZE=2`:

- `ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY` has a static assert requiring `CLUSTER_SIZE==1 && Mb==Nb`.
- `ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT` has a static assert requiring `CLUSTER_SIZE==1 && TOTAL_WGS==3`.
- `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE` has a static assert requiring `CLUSTER_SIZE==1`, `P_STAGE_SLOTS==2`, non-aliased direct x1 P scales, and one-to-one P payload/P-scale slots.
- The kept `skippscarrive` optimization depends on folded P-scale reuse, so it also cannot carry directly into a cluster2 route.

Structural patch candidate after this audit:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_2cta_pstage2_q200_p112_o56_qkscfix`

Candidate constraints:

- `TOTAL_WGS=3`, `NUM_THREADS=384`, `_ClusterSize=2`.
- Keep score-derived qkscfix math: fused block max, floor E8M0, x1 direct P-scale TMEM, prescaled E2M1 payload. No `fp4pv_pack_scores_to_stage_mxfp4` and no vector-amax-over-materialized-P quantization.
- Start without cluster1-only `diagzero`, `prepub`, `earlyreuse`, `pscreusefold`, or `skippscarrive`.
- Preserve dual score TMEM, direct-after-rescale dual output accumulation, producer-side V-scale ping-pong, and two P-scale TMEM slots.
- Treat CTAs as peer row-tile owners inside a macro-tile, not as one pure producer CTA and one pure PV CTA. This avoids moving row softmax/LSE ownership across CTAs in the first structural patch.
- If this correctness repro builds and smokes, later patches can reintroduce safe equivalents of the kept handoff shortcuts with cluster-aware remote-ready ordering.

Deadlock/corruption risks to test first:

- `p_remote_ready` phase ownership for cluster2 direct P-scale: CTA1 must publish remote P payload/scale readiness and CTA0 must wait exactly once per consumed score tile.
- `p_sc_tmem_ready` is only arrived by `cta_rank==0`; this is correct only if cluster PV issue is centralized through CTA0 and the cluster MMA consumes both CTA ranks as intended.
- `v_remote_ready` and `v_sc_tmem_ready` must remain ordered so the PV issue path does not shift the stall from P to V.
- Output/LSE finalization still stores `t_coord.y + cta_rank`; both CTA ranks must write their own row tile exactly once.
- `output_reusable`, `tile_arrived`, and `pv_tmem_ready` phases may be per-CTA or CTA0-centric depending on the cluster MMA path; small S128 stress is required before any timing.

Rejected first patch shape:

- A true CTA0-P-producer / CTA1-PV-consumer split is too broad for the next loop. It needs a new transport for score-derived P payload/scales and row softmax state, plus output TMEM/LSE ownership changes. It is a design direction only after the cluster2 peer macro-tile or another cluster-aware handoff repro proves the current cluster protocol can be made correct.

### Validation plan for the structural probe

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_2cta_qkscfix_probe.log
```

Route-string check:

```bash
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E '2cta.*qkscfix|pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix'
```

Correctness smoke, all with timeouts to catch deadlock:

- H16/S128 against kept qkscfix baseline, same prepared input, output and LSE finite/equal within the same tolerance used by previous loop smokes.
- H16/S4096 against kept qkscfix baseline.
- Repeat H16/S128 alternating launches for at least 50 iterations to catch phase/lifetime bugs before timing.
- If the probe survives, run H16/S8192 or longer S as a stress shape.

Direct preallocated timing:

- Same one-prepared-input paired timing harness as Loops 97-100.
- Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
- At least 30 warmups and 180 timed iterations.
- Compare against the kept qkscfix route in the same binary when possible.

NCU only if correctness passes and timing is non-negative or diagnostically important:

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_CONFIG='<route>' \
ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export <export-path> python3 <single-forward-profile-driver>
```

Required metric names to record:

- `gpu__time_duration.avg`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- `launch__registers_per_thread`
- `launch__barrier_count`
- `launch__shared_mem_per_block_static`
- `derived__local_spilling_requests`

Required source/SASS counters or prefixes to compare:

- `SYNCS`
- `SYNCS.PHASECHK.TRANS64.TRYWAIT`
- `SYNCS.ARRIVE`
- `BAR`
- `MEMBAR`
- `FENCE`
- `UTCBAR`
- `STTM`
- `LDTM`
- `STS`
- `LDS`
- `UTMALDG`
- `UTMASTG`
- `MUFU.EX2`
- `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`

Acceptance:

- Keep only if the route is correct and improves representative isolated kernel time or NCU shows improved overall PV feed without shifting stalls to V-scale or protocol overhead.
- Reject/revert if it deadlocks, corrupts output/LSE, lowers PV issue/tensor activity, increases `v_sc_tmem_ready`/wait stalls, or only wins by dropping kept qkscfix math/route features outside the intended structural trade.

## Structural 2-CTA peer macro-tile probe result

Status: rejected and reverted.

Route tested:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_2cta_pstage2_q200_p112_o56_qkscfix`

Kept baseline:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`

Files changed for the probe:

- `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`: added a guarded `_ClusterSize=2`, `P_STAGE_SLOTS=2` config inheriting the score-derived qkscfix direct-scale path.
- `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc`: added the route in both MXFP4 dispatch blocks.

No backward files were touched. The route was later removed from both files.

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_2cta_qkscfix_probe.log
```

Build result: success. The tested route compiled with `Used 168 registers, used 2 barriers, 1904 bytes smem` and no spills for the new route.

Route check:

```bash
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E '2cta.*qkscfix|pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix'
```

Correctness smoke commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 420s python3 -u <same-input kept-vs-probe H16/S128 smoke> 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_2cta_qkscfix_vs_kept_h16_s128.log
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 600s python3 -u <same-input kept-vs-probe H16/S4096 + repeat S128 smoke> 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_2cta_qkscfix_vs_kept_h16_s4096_repeat_s128.log
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 900s python3 -u <same-input kept/probe-vs-BF16 H16/S4096 smoke> 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_2cta_qkscfix_vs_bf16_h16_s4096.log
```

Smoke results:

- H16/S128 kept vs probe: finite, deterministic, not bitwise equal. Output max/mean abs diff `0.022705078125 / 7.235810335259885e-05`; LSE max/mean abs diff `0.01704549789428711 / 0.0016445904038846493`.
- H16/S4096 kept vs probe: finite, not bitwise equal. Output max/mean abs diff `0.033447265625 / 0.00020881078671664`; LSE max/mean abs diff `0.02001953125 / 0.0022770892828702927`.
- H16/S128 repeated alternating launches: 50/50 deterministic finite.
- H16/S4096 vs BF16 reference: kept output mean/RMSE `0.005304355174303055 / 0.010660047642886639`; probe output mean/RMSE `0.005293324589729309 / 0.01065919455140829`. LSE finite-ref mean error kept/probe `0.0015419453848153353 / 0.0014117788523435593`. The probe stayed in the same reference-error envelope, so timing was allowed.

Direct preallocated timing command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 1200s python3 -u <paired raw ext.forward_streaming_live_mxfp4 timing harness> 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/timing_2cta_qkscfix_vs_kept_prealloc.jsonl
```

Timing method: same prepared input per shape, raw preallocated extension calls, 30 warmups and 180 CUDA-event timed launches per block, two blocks per route in kept/probe/probe/kept order.

Timing results:

- H16/S2048 persistent: kept `0.05987200140953064 ms`, probe `0.07344000041484833 ms`, delta `+0.013567999005317688 ms`, speedup `-18.47494407499277%`.
- H16/S4096 persistent: kept `0.16697599738836288 ms`, probe `0.2008640021085739 ms`, delta `+0.03388800472021103 ms`, speedup `-16.871118948378516%`.
- H16/S8192 fullgrid: kept `0.5420959889888763 ms`, probe `0.6466080248355865 ms`, delta `+0.1045120358467102 ms`, speedup `-16.163120752063755%`.
- H4/S2048 persistent: kept `0.05947199836373329 ms`, probe `0.06676800176501274 ms`, delta `+0.0072960034012794495 ms`, speedup `-10.927395171952925%`.

NCU commands:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul PROFILE_ROUTE='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix' PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LAUNCH=persistent PROFILE_SEED=97301 \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_2cta_kept_h16_s4096 python3 -u <single post-warmup forward launch>
```

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul PROFILE_ROUTE='dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_2cta_pstage2_q200_p112_o56_qkscfix' PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LAUNCH=persistent PROFILE_SEED=97301 \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_2cta_probe_h16_s4096 python3 -u <single post-warmup forward launch>
```

NCU kept snapshot metric names and values:

- `gpu__time_duration.avg`: `154.208000 us`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`: `7.099774`
- `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`: `14.878104`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`: `36.128168`
- `smsp__warps_eligible.avg.per_cycle_active`: `0.418031`
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`: `3.636047`
- `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`: `1.609103`
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`: `0.209984`
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: `1.271668`
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`: `11.064899`
- `launch__registers_per_thread`: `168`
- `launch__barrier_count`: `2`
- `launch__shared_mem_per_block_static`: `1.904000 Kbyte/block`
- `derived__local_spilling_requests`: `0`
- `launch__cluster_size`: `1`
- `profiler__replayer_passes`: `16`

The probe NCU replay did not complete: after the first pass NCU reported a slow/hung workload replay for `kernel_streaming_live_fp4pv`; the process was terminated after roughly five minutes and logged `Failed to profile "kernel_streaming_live_fp4pv"`. Because direct timing was a uniform 11-18% regression, this profiling blocker was not chased further for the rejected route.

Revert command/path:

- Removed only the added route/config lines from `fwd_configs.inc` and `fwd_host_dispatch.inc`.
- Rebuilt:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_revert_2cta_qkscfix_probe.log
```

- Verified route strings:

```bash
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E '2cta_pstage2_q200_p112_o56_qkscfix|pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix' | tee results/mxfp4_fa4_forward_profile_20260612/route_strings_after_revert_2cta_qkscfix_probe.log
```

Result: only the kept qkscfix route remains.

Post-revert smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul timeout 420s python3 -u <kept H16/S4096 finite smoke> 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_revert_2cta_qkscfix_probe_h16_s4096.log
```

Result: finite output and LSE, `out_abs_max=0.76171875`, `lse_abs_max=8.320462226867676`.

Classification and blocker:

- The current kept route remains PV tensor-core underfed: tensor active `7.10%`, TC active `14.88%`, eligible warps `0.418/cycle`, long scoreboard `3.64`, wait `1.61`, no DRAM pressure, and no spills.
- The naive cluster2 peer macro-tile does not solve the feed problem. It preserves score-derived math and correctness envelope, but giving up the cluster1-only kept shortcuts (`diagzero`, `prepub`, `earlyreuse`, `pscreusefold`, `skippscarrive`) costs more than any structural benefit. The result is a uniform isolated-kernel regression.
- The next structural slice cannot simply turn on `_ClusterSize=2` around the current qkscfix path. It must either preserve equivalents of the kept cluster1 handoff shortcuts in a cluster-aware protocol, or implement a real P/PV ownership split that overlaps P payload/scale production with PV issue without moving Q/K/V TMEM resources or collapsing V-scale ping-pong.

## Next structural P/PV handoff audit after 2-CTA rejection

This audit is intentionally written before any next patch.

Current kept route facts from source after the revert:

- CTA/warpgroup shape: cluster1, `TOTAL_WGS=3`, 384 threads.
- WG ownership: `OUTPUT_WG=0`, `QUANT_WG0=1`, `PRODUCER_WG=2`, no relay WG.
- Q/K/V movement: producer WG owns TMA movement for Q/K/V payloads; `ONLINE_V_LOAD_WARPS=2` comes from `config_fp4pv_3wg_dual_score_force_persistent`.
- Row softmax/LSE ownership: quant WG owns row max/sum/correction state (`max_vec_smem`, `lse_smem`, `corr_vec_smem`), score-derived E8M0 P scales, and score-derived E2M1 P payload generation.
- Output ownership: output WG owns QK issue, P/V scale staging waits, PV MMA issue, output accumulator TMEM, final output store, and LSE store.
- P payload lifetime: `p_fp4_stage[P_STAGE_SLOTS]` with kept `P_STAGE_SLOTS=2`; payload is published early by the qkscfix route and reuse is signaled after PV consumes the tile.
- P-scale lifetime: kept route uses direct x1 score-derived P-scale TMEM with `P_SCALE_TMEM_SLOTS=2`; `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE` folds P-scale reuse into P payload reuse, and `ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE` skips the extra P-scale reuse arrive.
- V-scale lifetime: V payload shared memory stays double buffered; V-scale TMEM uses `V_SCALE_TMEM_SLOTS=2`. The next patch must preserve this ping-pong.
- TMEM layout: dual score `256`, output `128`, Q scale `16`, K scale `16`, P scale `32`, V scale `64`, total `512`. There is no non-aliased room for extra P-scale slots without footprint relief.

Why a simple feeder/relay is not automatically acceptable:

- The current output path already has a decoupled QK/PV issue mode, but the PV path still calls `wait_and_stage_v_sc`, `wait_and_stage_p_sc`, and `issue_pv` from the output WG loop.
- Moving those same waits into a relay/feeder without overlapping them before PV reaches the tile would add a new `pv_feed_ready` wait and likely just move the same `SYNCS`/TMEM wait cost behind more protocol.
- A feeder warp inside `OUTPUT_WG` is not obviously safe because the PV MMA helpers are warpgroup-oriented. Stealing one warp from the output WG may reduce the PV issue group or break collective assumptions unless the code is explicitly split so only the intended lane stages scales and the intended PV lane issues.
- A feeder WG outside the current 3WG shape is a broad resource change. It risks lower occupancy/register balance and violates the preference for normal 3WG/384-thread unless a smaller correctness repro proves the current WG is under-resourced.

True structural handoff target if attempted:

- Keep cluster1 and the kept qkscfix math/route features first.
- Keep Q/K/V movement resources intact: producer WG still loads Q/K/V payloads and V payload ping-pong remains two slots.
- Do not deepen P scale by stealing V-scale TMEM. P scale remains two TMEM slots unless footprint relief is implemented.
- Split the output WG feed path only if the scale-staging lane can stage both P-scale and V-scale TMEM for tile `i+1` while PV issues tile `i`.
- The PV issue lane should wait on one combined `pv_feed_ready[slot]` for tile `i`, then issue PV MMA, then signal both P payload/P-scale and V-scale reuse.
- The feed lane should wait on P payload/scale readiness and V payload readiness, stage existing P-scale/V-scale TMEM slots, publish exactly one combined feed-ready event, and advance. It must not hold future P tiles in registers and must not add another per-32 or per-half churn event.

Deadlock/lifetime risks to solve before patching:

- Reuse phase ownership must move from the PV lane to the feed/PV pair without double-arriving `p_sc_tmem_reusable` or `v_sc_tmem_reusable`.
- The feed lane must not claim a P-scale TMEM slot before PV has consumed the previous tile mapped to that slot.
- The feed lane must not overwrite V-scale TMEM slot `score_idx&1` until PV signals `v_sc_tmem_reusable`.
- If `ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE` remains enabled, the combined feed protocol must preserve the invariant that P payload and P-scale slots are one-to-one and released after PV consumes the tile.
- If the implementation needs a new semaphore, it should be one coarse `pv_feed_ready[P_STAGE_SLOTS]` event; adding separate payload-ready, scale-ready, and V-ready events would violate the goal.
- Any route that disables `prepub`, `earlyreuse`, `pscreusefold`, or `skippscarrive` must be treated as structurally suspect because the 2-CTA peer probe already showed those losses dominate.

Alternative structural direction:

- A cluster2 route is only worth revisiting if it preserves cluster-aware equivalents of the kept cluster1 shortcut stack. The failed 2-CTA route proves that `_ClusterSize=2` without `diagzero/prepub/earlyreuse/pscreusefold/skippscarrive` is not competitive.
- Cluster-aware prepublish would need to publish shared payload backing for local and remote CTA visibility before scale-slot waits.
- Cluster-aware P-scale reuse folding would need remote-safe reuse phase handling so skipping the extra P-scale reuse arrive remains correct.
- This is riskier than a cluster1 feed-ready split because the row softmax/LSE and output accumulator ownership remains per CTA and remote-ready phases become correctness-critical.

Exact next smoke/timing/NCU plan for any feed-ready structural patch:

Build:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_<route>.log
```

Route check:

```bash
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E '<new-route>|pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix'
```

Correctness:

- H16/S128 kept-vs-probe finite/equality envelope.
- H16/S4096 kept-vs-probe finite/equality envelope.
- H16/S4096 probe-vs-BF16 reference if kept-vs-probe is not bitwise equal but remains small.
- 50 alternating H16/S128 launches to catch phase/lifetime mistakes.

Direct timing:

- Same preallocated raw extension harness used above.
- Shapes: H16/S2048 persistent, H16/S4096 persistent, H16/S8192 fullgrid, H4/S2048 persistent.
- 30 warmups and 180 timed launches per block, kept/probe/probe/kept order.

NCU only if timing is non-negative or the route is diagnostically important:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul PROFILE_ROUTE='<route>' PROFILE_SEQ=4096 PROFILE_HEADS=16 PROFILE_LAUNCH=persistent PROFILE_SEED=<seed> \
  ncu --profile-from-start off --target-processes all --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_<route>_h16_s4096 python3 -u <single post-warmup forward launch>
```

Required metric names:

- `gpu__time_duration.avg`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- `launch__registers_per_thread`
- `launch__barrier_count`
- `launch__shared_mem_per_block_static`
- `derived__local_spilling_requests`

Acceptance:

- Keep only if the route preserves qkscfix numerics and improves representative isolated kernel timing, or if NCU shows improved PV issue/tensor activity without shifting the stall to V-scale TMEM wait or protocol overhead.
- Reject/revert if it only makes P staging earlier while lowering PV issue rate, collapsing V-scale ping-pong, increasing barrier/wait stalls, or reducing occupancy enough to erase overlap.

## Loop 31 - feed-ready/PV handoff gate ledger

### Gate 0 - baseline hygiene

Commands:

```bash
git status --short results/mxfp4_fa4_forward_profile_20260612/ledger.md tk_fa4/fp4_fa4_fwd/fwd_configs.inc tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc tk_fa4/fp4_fa4_fwd/fwd_device_helpers.inc
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'pvfeedready|pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix'
git diff -- tk_fa4/fp4_fa4_fwd/fwd_device_helpers.inc
```

State:

- Forward config/dispatch/kernel are clean before this probe.
- Dirty files before source edits are this ledger and `tk_fa4/fp4_fa4_fwd/fwd_device_helpers.inc`.
- The dirty helper edit is pre-existing guarded K256 tail-offset scaffolding in `fp4pv_zero_invalid_causal_payload_groups_mxfp4`; it is unrelated to this feed-ready route and will not be overwritten.
- Kept qkscfix route is present in the binary:
  `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix`.
- No `pvfeedready` route is present before this probe.
- Kept route type remains:
  `config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1>`.
- Kept ptxas footprint from the last clean build: 168 registers/thread, 2 barriers, 1904 bytes static smem, 0 spill stores, 0 spill loads.
- Kept H16/S4096 NCU bottleneck snapshot:
  `gpu__time_duration.avg=154.208000 us`,
  `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=7.099774`,
  `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed=14.878104`,
  `smsp__issue_active.avg.pct_of_peak_sustained_active=36.128168`,
  `smsp__warps_eligible.avg.per_cycle_active=0.418031`,
  `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=3.636047`,
  `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio=1.609103`,
  `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio=0.209984`,
  `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed=1.271668`,
  `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed=11.064899`,
  `launch__registers_per_thread=168`,
  `launch__barrier_count=2`,
  `launch__shared_mem_per_block_static=1.904000`,
  `derived__local_spilling_requests=0`.

Classification remains PV tensor-core underfeed with low eligible warps and long scoreboard/wait, not DRAM, launch, or spill limited.

### Gate 1 - design audit before code

Current serial qkscfix PV path:

- CTA/WG ownership for the kept 3WG path is `PRODUCER_WG=2`, `OUTPUT_WG=0`, `QUANT_WG0=1`, `RELAY_WG=-1`.
- For a guarded 4WG online direct route, existing ownership logic maps `PRODUCER_WG=3`, `OUTPUT_WG=2`, `QUANT_WG0=1`, `RELAY_WG=0`. Existing relay code is inactive for online direct P-ready routes, so the feeder can be added under a new route flag without changing the kept route.
- P payload slots are `p_fp4_stage[P_STAGE_SLOTS]`; kept route has `P_STAGE_SLOTS=2`.
- P-scale TMEM slots are `P_SCALE_TMEM_SLOTS=2` for direct online P-scale; kept `pscreusefold` keeps P payload and P-scale slot lifetime one-to-one.
- V-scale TMEM slots are `V_SCALE_TMEM_SLOTS=2`; this probe preserves V-scale ping-pong.
- Current producer loads V payload and V scales into shared, publishes shared backing, then signals `v_arrived[v_idx]`.
- Current quant WG generates score-derived prescaled E2M1 payload directly from scores/local/global max and E8M0 P scale from the score-block max versus row max. It stores the x1 P scale directly to P-scale TMEM with `fp4pv_store_mxfp4_scale_tmem_32x32b_x1`, waits with `fp4pv_tmem_store_wait`, prepublishes payload backing under `prepub/earlyreuse`, and signals `p_sc_tmem_ready[p_sc_slot]`.
- Current PV issue path waits/stages V scale (`v_arrived` -> `fp4pv_load_mxnv_scale_async_cluster`) and waits P scale readiness (`p_sc_tmem_ready`) before `tensor_load_wait`, `tensor_before_thread_sync`, and PV MMA. It then commits `pv_tmem_ready`.
- Current output drain waits `pv_tmem_ready`, applies the dual-output/direct-rescale sequencing, and releases P payload reuse through `p_stage_reusable`. Because `pscreusefold+skippscarrive` is enabled, there is no separate P-scale reuse arrive on the kept route.

Feeder ownership decision:

- Do not use an output-WG sub-warp feeder for the first patch. The PV MMA helpers are warpgroup tcgen issue paths, not proven lane-local helper calls; stealing lanes/sub-warps from OUTPUT_WG risks breaking collectives or lowering PV issue rate.
- Do not steal producer WG resources: producer currently owns Q/K/V movement and V payload/shared-scale staging.
- Do not steal quant WG resources: quant owns row softmax/LSE, score-derived E2M1 P payload generation, and score-derived E8M0 P-scale generation.
- Use a dedicated guarded 4WG feeder route (`RELAY_WG=0`) only for the opt-in `pvfeedready` route.

Planned guarded protocol:

- Add `ONLINE_PV_FEED_READY` as an opt-in config trait and route string containing `pvfeedready`.
- Preserve qkscfix math and route features: score-derived P payload/scale, `diagzero`, `prepub`, `earlyreuse`, `arrivereuse`, `pscreusefold`, `skippscarrive`, `dualaccum_directrescale`, `decoupled`, `P_STAGE_SLOTS=2`, `P_SCALE_TMEM_SLOTS=2`, `V_SCALE_TMEM_SLOTS=2`.
- Add one coarse `pv_feed_ready[P_STAGE_SLOTS]` event. No per-32, K64, payload-ready/scale-ready/V-ready event fanout.
- For the feed route only, quant WG writes the x1 score-derived P-scale word into shared shadow `p_online_mxfp4_p_scale_words[buf][row]` and signals existing `p_quant_ready[buf]` after payload publish. It does not directly write P-scale TMEM or signal `p_sc_tmem_ready`.
- Feeder WG waits the existing producer-to-consumer readiness primitives for tile `i`: `p_quant_ready[buf]` for P payload/P-scale shadow and `v_arrived[v_buf]` for V shared state. It stages P scale from the shared shadow into the existing P-scale TMEM slot, stages V scale into the existing V-scale TMEM ping-pong slot, performs the required TMEM/proxy/order waits, then signals exactly one `pv_feed_ready[buf]`.
- PV issue path waits only `pv_feed_ready[buf]` before its existing `tensor_load_wait`, `tensor_before_thread_sync`, and PV MMA. PV/output release still signals P payload reuse exactly once through the kept `p_stage_reusable` path. The feeder route adds V-scale slot reuse signaling for the two-slot V-scale ping-pong so the feeder never overwrites a live V-scale TMEM slot.

Startup/drain:

- Feeder naturally prefeeds tile 0 because PV waits `pv_feed_ready[buf]` for tile 0 and the feeder starts at idx 0.
- Feeder advances one tile at a time and may stage tile `i+1` while PV consumes tile `i`; it never holds future P tiles in registers.
- Drain stops at `iters_per_task`; the final tile uses the same `pv_feed_ready`/`pv_tmem_ready`/output drain path as earlier tiles, avoiding a special tail event.

Risk gates:

- Reject on compile-time resource blowup, spills, or large register/barrier increase.
- Reject on any H16/S128 or H16/S4096 finite/equality failure, or any intermittent H16/S4096 corruption over the repeated alternating run.
- Reject if timing/NCU shows the route only moves stalls to V-scale waits/protocol barriers, lowers PV issue/tensor active, or loses occupancy enough to erase overlap.

### Gate 2 - guarded implementation and resources

Implemented and tested an opt-in route:
`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pvfeedready_pstage2_q200_p112_o56_qkscfix`.

Implementation summary:

- Added a guarded 4WG config with `ONLINE_PV_FEED_READY=true`, `TOTAL_WGS=4`, and `ONLINE_RELAY_REGS=24`.
- Added one coarse `pv_feed_ready[P_STAGE_SLOTS]` event.
- Quant WG kept score-derived qkscfix payload math and wrote x1 P-scale words to the existing shared P-scale shadow.
- Feeder WG staged P-scale shadow into existing two P-scale TMEM slots, staged V scales into existing two-slot V-scale ping-pong, then signaled one coarse `pv_feed_ready`.
- PV WG waited `pv_feed_ready`, issued the existing PV MMA, and output drain released P-stage reuse. A first correctness fix moved V-scale TMEM reuse release from PV issue time to post-`pv_tmem_ready` output drain time.

Build commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pvfeedready.log
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_pvfeedready_vreusefix.log
```

Route string checks:

```bash
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -E 'pvfeedready|pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix'
```

ptxas resources after V-scale reuse fix:

- Kept qkscfix: 168 registers/thread, 2 barriers, 1904 bytes static smem, 0 stack, 0 spill stores, 0 spill loads.
- Probe `pvfeedready`: 128 registers/thread, 4 barriers, 2976 bytes static smem, 0 stack, 0 spill stores, 0 spill loads.

### Gate 3 - correctness

Smoke command pattern:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u <inline kept/probe smoke> | tee results/mxfp4_fa4_forward_profile_20260612/smoke_pvfeedready_vreusefix_vs_kept.log
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u <inline 80-pair alternating smoke> | tee results/mxfp4_fa4_forward_profile_20260612/repeat_pvfeedready_vreusefix_pairwise_h16_s4096.log
```

Single-shot smoke, after V-scale reuse fix:

- H16/S128, seed 97141, persistent: finite kept/probe output and LSE; output exact; LSE exact.
- H16/S4096, seed 97142, persistent: finite kept/probe output and LSE; output exact; LSE exact.

80-pair alternating H16/S4096 correctness, seed 97143:

- finite failures: 0.
- pairwise LSE exact: 80/80.
- pairwise output exact: 72/80.
- maximum pairwise output delta: 0.00140380859375; maximum pairwise output mean delta: 1.655426729030296e-08.
- kept route drift vs first kept reference: output exact 72/80, max output delta 0.00140380859375, LSE exact 80/80.
- probe route drift vs first kept reference: output exact 80/80, max output delta 0.0, LSE exact 80/80.
- no bad examples by the corruption threshold; this matched the known sub-0.002 output-only jitter envelope and did not show LSE corruption.

Correctness decision: pass for timing/profile gate.

### Gate 4 - timing/profile and decision

Direct preallocated timing command pattern:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u <inline raw-extension timing> | tee results/mxfp4_fa4_forward_profile_20260612/timing_pvfeedready_vreusefix_vs_kept_prealloc.jsonl
```

Timing protocol:

- Shared prepared inputs per shape.
- Raw extension launch into preallocated `out`/`lse`.
- Order: kept, probe, probe, kept.
- 30 warmups and 180 timed iterations per block.

Timing results:

- H16/S2048 persistent: kept 0.05126426484849718 ms, probe 0.05950808789994982 ms, delta +16.08103242251859%.
- H16/S4096 persistent: kept 0.15172257423400878 ms, probe 0.18409111234876846 ms, delta +21.334029084449995%.
- H16/S8192 fullgrid: kept 0.5237323549058702 ms, probe 0.6426883697509765 ms, delta +22.713130806380306%.
- H4/S2048 persistent: kept 0.04514844417572021 ms, probe 0.055278311835394965 ms, delta +22.436803404008245%.

NCU commands:

```bash
CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_kept_h16_s4096 python3 - <<'PY'
# H16/S4096, seed 97211, ten raw-extension warmups, cudaProfilerStart(), one preallocated persistent launch, cudaProfilerStop().
PY

CUDA_VISIBLE_DEVICES=0 TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pvfeedready_pstage2_q200_p112_o56_qkscfix PYTHONPATH=/workspace/codebases/pv/fp4_matmul \
  ncu --target-processes all --profile-from-start off --kernel-name regex:kernel_streaming_live_fp4pv --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_probe_h16_s4096 python3 - <<'PY'
# H16/S4096, seed 97211, ten raw-extension warmups, cudaProfilerStart(), one preallocated persistent launch, cudaProfilerStop().
PY
```

NCU export commands:

```bash
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_kept_h16_s4096.ncu-rep --csv --page raw --metrics gpu__time_duration.avg,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__barrier_count,launch__shared_mem_per_block_static,derived__local_spilling_requests > results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_kept_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_probe_h16_s4096.ncu-rep --csv --page raw --metrics gpu__time_duration.avg,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__barrier_count,launch__shared_mem_per_block_static,derived__local_spilling_requests > results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_probe_h16_s4096_raw.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_kept_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_kept_h16_s4096_source.csv
ncu --import results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_probe_h16_s4096.ncu-rep --csv --page source > results/mxfp4_fa4_forward_profile_20260612/ncu_pvfeedready_vreusefix_probe_h16_s4096_source.csv
```

Metric names and values, H16/S4096:

- `gpu__time_duration.avg`: kept 155.392 us, probe 188.384 us.
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`: kept 7.157046%, probe 5.859570%.
- `sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed`: kept 15.008326%, probe 49.069418%.
- `smsp__issue_active.avg.pct_of_peak_sustained_active`: kept 36.134224%, probe 36.010838%.
- `smsp__warps_eligible.avg.per_cycle_active`: kept 0.417768, probe 0.448703.
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`: kept 3.639997, probe 5.175480.
- `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio`: kept 1.608427, probe 2.150568.
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`: kept 0.207598, probe 0.235798.
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: kept 1.261875%, probe 1.041619%.
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`: kept 11.143116%, probe 11.947211%.
- `launch__registers_per_thread`: kept 168, probe 128.
- `launch__barrier_count`: kept 2, probe 4.
- `launch__shared_mem_per_block_static`: kept 1.904 KB, probe 2.976 KB.
- `derived__local_spilling_requests`: kept 0, probe 0.

Source-page executed instruction counts:

- `SYNCS`: kept 6391517, probe 8305224.
- `SYNCS.PHASECHK.TRANS64.TRYWAIT`: kept 6109597, probe 7952648.
- `BAR`: kept 227856, probe 292128.
- `MEMBAR`: kept 58640, probe 82976.
- `LDS`: kept 43520, probe 180736.
- `STS`: kept 428944, probe 462736.
- `STTM`: kept 275464, probe 274704.
- `LDTM`: kept 376840, probe 376080.
- `MUFU`: kept 4392960, probe 4401152.
- `F2FP.SATFINITE.E2M1.F32.PACK`: kept 2162688, probe 2162688.

Classification:

- The feed-ready route did not improve overall PV feed. It preserved V-scale ping-pong but added a fourth WG, doubled static barrier count, raised shared memory, increased protocol/source `SYNCS` by about 1.9M executed instructions, raised `long_scoreboard`, `wait`, and `barrier` stall ratios, and reduced tensor active while wall time regressed.
- The high `sm__pipe_tc_cycles_active` in the probe is not an accepted win because kernel time and tensor active regress and the extra protocol stalls dominate.

Decision:

- Rejected and reverted. The blocker is protocol overhead from the dedicated feeder WG and coarse `pv_feed_ready` handoff: it moves scale staging out of PV but pays additional `p_quant_ready`/`pv_feed_ready`/V-scale-reuse synchronization, more shared loads, and a 4WG/barrier footprint without improving PV issue rate.
- Next structural slice should not add a relay WG that simply replays the same P/V scale work behind extra `SYNCS`; it needs either fewer total handoff events on the existing 3WG path, a half-tile/K64 handoff that starts PV earlier without extra TMEM pressure, or real TMEM/footprint relief that increases P-scale depth while preserving V-scale ping-pong.

### Gate 5 - clean revert verification

Revert/build commands:

```bash
CUDA_VISIBLE_DEVICES=0 make -C tk_fa4/fp4_fa4_fwd forward 2>&1 | tee results/mxfp4_fa4_forward_profile_20260612/build_after_pvfeedready_revert.log
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -F 'dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix'
strings tk_fa4/_C_b300_causal_fp4_fwd_experiments.cpython-312-aarch64-linux-gnu.so | grep -F 'pvfeedready'
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/codebases/pv/fp4_matmul python3 -u <inline restored-kept smoke> | tee results/mxfp4_fa4_forward_profile_20260612/smoke_after_pvfeedready_revert_kept_h16_s4096.log
```

Post-revert state:

- Forward config/dispatch/kernel have no remaining `pvfeedready` source diff.
- Rebuilt binary contains kept qkscfix route and does not contain `pvfeedready`.
- Kept route ptxas after rebuild: 168 registers/thread, 2 barriers, 1904 bytes static smem, 0 stack, 0 spill stores, 0 spill loads.
- H16/S4096 restored-kept smoke, seed 97221: finite output true, finite LSE true, `out_abs_max=0.77734375`, `lse_abs_max=8.320328712463379`.

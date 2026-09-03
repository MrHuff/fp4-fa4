# Stage2 Mixed ALU EX2 Emulation And FP4 Packing Report

Date: 2026-07-11

Task: `session6_stage2_ex2_alu_pack_20260711.md`

Hardware: NVIDIA GB200, GPU2, SM100a. All wall-time runs used the persistent MXFP4 forward route with timeline and sparse stamps disabled unless a stamp run is explicitly identified.

## Result

The exact local FA4 paired degree-3 EX2 emulation was ported, validated in an isolated SM100a probe, and integrated only into the Stage2 score-derived prescaled P payload loop. A compile-time 4-of-16 pair cadence (`e16`) was the best independent cadence: it reduced `MUFU.EX2` from 257 to 193, added 96 packed `FFMA2.FTZ` instructions, retained 168 registers and zero spills, shortened the measured P interval, and repeated wall-time gains on all three required shapes.

The existing and interleaved FP4 packing schedules were both implemented. Interleaving changed SASS but did not repeat an incremental gain across shapes, so its selectors and behavior were removed. The optional fused inline-PTX pack helper was not justified because the compiler did honor the tested schedule and that schedule did not win.

Combining 4-of-16 emulation with retained `pchainc` produced the final explicit/default-off route:

`dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix`

`e16pc` repeated p50 gains versus matched Stage2 on h4/s2048, h8/s1024, h8/s4096, and h16/s4096. It retained Stage2-level high-head determinism, while standalone `e16` did not, so only the combined route remains selectable. Stage2 remains the global default.

## Implementation

- Added `stage2_ex2_alu_helpers.cuh`, included by the forward-only experiment translation unit and tracked by its Makefile dependency list.
- `fp4pv_ex2_alu_emulation_f32x2` is a literal CUDA inline-PTX port of local `flash-attention/flash_attn/cute/utils.py::e2e_asm2`: FTZ clamp to -127, packed round-down with `0x4B400000`, two packed RN subtracts, coefficients `0x3D9DF09D`, `0x3E6906A4`, `0x3F31F519`, three `fma.rn.ftz.f32x2`, and integer exponent reconstruction.
- Added a compile-time static loop and `fp4pv_stage2_exp2_pair<Period, PairIndex>`. The selected four pairs are the final four positions in each period; there is no runtime cadence branch.
- Restricted emulation with static assertions to the online MXFP4 score-derived prescaled P payload route. `acc_scale` and unrelated exponentials remain native.
- Applied the helper in both nonzero-scale and zero-scale/causal paths. The zero-scale path retains `-inf`; the helper's clamp/reconstruction maps causal `-inf` to zero.
- The final config is `config_fp4pv_stage2_ex2_alu_pchain_c<16,...>`, which also enables the previously retained early asynchronous P-scale store and deferred row-sum update.

Source anchors after cleanup:

- helper: `tk_fa4/fp4_fa4_fwd/stage2_ex2_alu_helpers.cuh:7`
- config: `tk_fa4/fp4_fa4_fwd/fwd_configs.inc:675`
- nonzero/zero payload loops: `tk_fa4/fp4_fa4_fwd/fwd_streaming_kernel.inc:21869` and `:22026`
- explicit dispatch: `tk_fa4/fp4_fa4_fwd/fwd_host_dispatch.inc:2654` and `:3351`

## Probe Validation

The standalone probe used the repository's required `-gencode arch=compute_100a,code=sm_100a`. A first plain `-arch=sm_100a` attempt correctly failed because plain SM100 PTX does not expose the E2M1 conversion target; no numerical result from that attempt was used.

SASS isolation proved the intended pipeline split:

| Probe kernel | `MUFU.EX2` | `FFMA2.FTZ` |
|---|---:|---:|
| paired emulation only | 0 | 3 |
| paired native EX2 only | 2 | 0 |

Over 1,000,037 pairs (2,000,074 scalar values), including 74 threshold-adjacent directed values:

| Metric | Result |
|---|---:|
| max absolute EX2 error vs native | 4.625320e-4 |
| max relative EX2 error vs native | 8.778174e-5 |
| all-emulated scalar E2M1 mismatches | 39 / 2,000,074 = 1.949928e-5 |
| directed threshold-adjacent mismatches | 19 / 74 |
| all-emulated 8-value payload-word mismatches | 26, rate 1.039963e-4 |
| 4-of-16 scalar mismatches | 11, global rate 5.499797e-6 |
| 4-of-16 payload-word mismatches | 5, rate 1.999928e-5 |
| causal `-inf` maps to zero | yes |

The directed set intentionally straddles quantization boundaries; its higher local mismatch count explains where the small global mismatch rate comes from.

Artifacts: `forward_stage2_ex2_alu_probe.cu`, `forward_stage2_ex2_alu_probe_build.log`, `forward_stage2_ex2_alu_probe_gpu2.json`, and `forward_stage2_ex2_alu_probe_{emulated_only,native_only}.sass`.

## Cadence Sweep

Static SASS counts and ptxas resources:

| Route | Emulated pairs | MUFU | FFMA2 | F2FP | E2M1 | BRA | ptxas |
|---|---:|---:|---:|---:|---:|---:|---|
| Stage2 | 0% | 257 | 0 | 144 | 128 | 263 | 168 regs, 2 barriers, 1904 B smem, no stack/spills |
| e16 | 25% | 193 | 96 | 144 | 128 | 263 | same, no stack/spills |
| e12 | 33% | 177 | 120 | 144 | 128 | 263 | same, no stack/spills |
| e10 | 40% | 161 | 144 | 144 | 128 | 263 | 8 B stack, 8 B stores/8 B loads |
| e8 | 50% | 129 | 192 | 144 | 128 | 263 | 8 B stack, 16 B stores/16 B loads |
| all emulated | 100% | 1 | 384 | 144 | 128 | 263 | 168 regs, no stack/spills |

The unchanged branch and conversion counts prove static convergent selection. The one remaining MUFU in the all-emulated route belongs to an unrelated exponential intentionally outside this experiment.

First-pass p50/min ms, with p50 delta versus the Stage2 in the same run:

| Shape | Stage2 | e16 | e12 | e10 | e8 | all emulated |
|---|---:|---:|---:|---:|---:|---:|
| h4/s2048 | .067104/.065152 | .064800/.061952 (-3.43%) | .065280/.063040 (-2.72%) | .069056/.066624 (+2.91%) | .070752/.068864 (+5.44%) | .082816/.080704 (+23.41%) |
| h8/s1024 | .050080/.047904 | .049280/.047136 (-1.60%) | .050304/.046368 (+0.45%) | .050720/.048032 (+1.28%) | .051712/.049312 (+3.26%) | .058176/.055040 (+16.17%) |
| h8/s4096 | .104448/.100192 | .098560/.095872 (-5.64%) | .099168/.096864 (-5.06%) | .104800/.102496 (+0.34%) | .111552/.108896 (+6.80%) | .133408/.131776 (+27.73%) |

Reverse-order repeat p50/min ms:

| Shape | Stage2 | e16 | e12 |
|---|---:|---:|---:|
| h4/s2048 | .068208/.066176 | .065808/.063904 (-3.52%) | .066224/.064160 (-2.91%) |
| h8/s1024 | .049312/.046368 | .048336/.046368 (-1.98%) | .048288/.045920 (-2.08%) |
| h8/s4096 | .102496/.099296 | .097248/.095200 (-5.12%) | .097984/.095136 (-4.40%) |

`e10` and `e8` fail the no-spill gate and were slower; all-emulated is an ALU/FMA ceiling and was substantially slower. `e12` was valid but consistently weaker than e16 at the important long shape and has a higher payload mismatch rate, so it was not retained.

## Sparse P-Chain Attribution

Median cycles from uncontended stamp runs:

| Shape/route | scale -> exp/pack | exp/pack -> payload bytes | scale -> P-ready |
|---|---:|---:|---:|
| h4/s2048 Stage2 | 1354 | 227 | 1907 |
| h4/s2048 e16 | 1189 (-165) | 226 | 1739 (-168) |
| h4/s2048 e16pc | 1288 (-66) | 146 (-81) | 1696 (-211) |
| h8/s1024 Stage2 | 1327 | 228 | 1880 |
| h8/s1024 e16 | 1187 (-140) | 218 (-10) | 1738 (-142) |
| h8/s1024 e16pc | 1284 (-43) | 141 (-87) | 1687 (-193) |
| h8/s4096 Stage2 | 1350 | 229 | 1908 |
| h8/s4096 e16pc | 1404 (+54) | 192 (-37) | 1857 (-51) |

The independent e16 cadence directly shortens the target interval on both supporting shapes. `pchainc` changes marker ordering by publishing P-ready before deferred row-sum correction; in the combined route the more meaningful end-to-ready interval improves by 193-211 cycles on the short/supporting shapes and by 51 cycles on h8/s4096. Only uncontended stamp artifacts are used here.

## Packing Schedules

For e16 and e12, the existing per-pack `produce -> row-sum/pack` schedule was compared with an interleaved producer/consumer schedule. The interleaved kernels had different SASS hashes and the same 168-register, no-spill resource use, so the compiler did preserve the scheduling change.

The result was not repeatable as an incremental improvement over the same cadence. In reverse-order timing, e16 interleaving was effectively tied/slower at h4/s2048 (.066592 vs .066560 ms), slower at h8/s1024 (.047824 vs .047600), and faster only at h8/s4096 (.101296 vs .101952). e12 interleaving regressed h4/s2048 and did not produce a consistent cross-shape result. Both interleaved routes were removed.

## Retained Combination Timing

Clean-build repeated timing used 30 samples per route. Each cell is first-order p50/min followed by reverse-order p50/min and its p50 delta versus matched Stage2:

| Shape | Stage2 | standalone e16 | retained e16pc |
|---|---:|---:|---:|
| h4/s2048 | .068688/.064672; .067600/.064608 | .065616/.062880; .065392/.058880 | .065152/.063136 (-5.15%); .065024/.058464 (-3.81%) |
| h8/s1024 | .047232/.045088; .045216/.040736 | .046496/.043904; .044480/.042688 | .046368/.042496 (-1.83%); .044144/.042112 (-2.37%) |
| h8/s4096 | .105232/.102144; .104464/.102944 | .100496/.096224; .099040/.097152 | .100400/.094624 (-4.59%); .099056/.095872 (-5.18%) |
| h16/s4096 | .181488/.178656; .182512/.180288 | .174704/.171680; .176208/.173184 | .174672/.172288 (-3.76%); .174192/.171200 (-4.56%) |

The combo also improves incrementally over standalone e16 on h4/s2048 and h8/s1024 in both orders, is neutral on h8/s4096, and is better on the reverse h16/s4096 run.

## Correctness And Determinism

All smoke and timing runs were finite. The final h4/s1024 winner smoke was finite and repeated; candidate max-absolute output difference versus Stage2 was 4.882813e-4.

Reverse-order `e16pc` numerical deltas versus Stage2:

| Shape | output max abs | output relative L2 | LSE max abs | BF16 RMSE | BF16 LSE max abs |
|---|---:|---:|---:|---:|---:|
| h4/s2048 | 1.098633e-3 | 3.471298e-4 | 1.764297e-5 | .013786814 | .029491946 |
| h8/s1024 | 7.324219e-4 | 2.347278e-4 | 1.764297e-5 | .019235828 | .029491946 |
| h8/s4096 | 4.882813e-4 | 2.266276e-4 | 1.859665e-5 | .010798289 | .019126892 |
| h16/s4096 | 4.882813e-4 | 2.252934e-4 | 1.931190e-5 | .010649993 | .021505542 |

The BF16 envelope is effectively unchanged from Stage2. On reverse high-head repeats, `e16pc` output run-to-run max abs was 6.103516e-5 and LSE max abs 9.536743e-7, within the matched Stage2 envelope. Standalone e16 reached 2.441406e-4/6.752014e-4 on h8/s4096 and 3.479004e-3/4.787445e-4 on h16/s4096 (output/LSE), so its selector was removed despite its speed.

## Winner NCU Profile

Compact h8/s4096 NCU results, timeline/stamps off:

| Metric | Stage2 | e16 | e16pc |
|---|---:|---:|---:|
| duration (us) | 87.904 | 82.528 | 82.208 (-6.48%) |
| ALU active (%) | 20.49 | 23.56 | 23.66 |
| FMA active (%) | 14.77 | 22.56 | 22.66 |
| FMA-lite active (%) | 12.63 | 19.18 | 19.32 |
| issue active (%) | 30.57 | 36.31 | 36.41 |
| eligible warps/cycle | .40 | .48 | .48 |
| long-scoreboard ratio | 3.47 | 2.78 | 2.75 |
| wait ratio | 1.75 | 1.27 | 1.26 |
| TC active (%) | 13.52 | 14.70 | 14.76 |
| tensor active (%) | 6.31 | 6.73 | 6.76 |

One-replay dynamic pipe counts:

| Pipe instructions | Stage2 | e16 | e16pc |
|---|---:|---:|---:|
| ALU | 6,143,770 | 6,624,591 | 6,623,965 |
| FMA | 5,646,247 | 7,853,407 | 7,853,322 |
| FMA-lite | 3,786,624 | 5,391,744 | 5,408,639 |
| XU | 2,292,224 | 1,751,552 | 1,751,552 (-23.6%) |

The profile shows the intended SFU/XU-to-FMA redistribution, shorter duration, better issue/eligibility, and lower scoreboard/wait pressure. FMA active remains only 22.66%, so the winner does not saturate that pipe. NCU did not expose a useful dynamic F2FP metric in the compact pass; static SASS proves its count remains 144. An initial broad conversion-metric pass failed to emit a usable row and was stopped rather than used; the compact and pipe-count passes above completed normally.

## FP4-Native Addendum

The follow-up task `session6_fp4_native_exp_pack_addendum_20260711.md` was executed after the mixed-cadence matrix above. It tested:

- F1: degree-3 FA4 cubic for the row sum plus a direct nibble derived from the same polynomial.
- F2 degree-3: the same cubic row sum plus a direct nibble derived from the original log-domain region/fraction.
- F2 degree-2: the FA4 degree-2 cubic for the row sum plus the same F2 direct nibble.
- F3: direct F2 nibble classification for every pair plus a dequantized E2M1 denominator accumulated with PRMT and DP4A; no production P-loop EX2 or E2M1 conversion.

All classifiers used at most two fractional comparisons after FA4 range reduction. An initial C++ implementation generated data-dependent branches, so it was replaced before kernel integration with explicit PTX predicate/select sequences. Final microprobe SASS has only the common kernel bounds branch.

### Addendum Microprobe

The focused SM100a probe covered 1,000,025 pairs (2,000,050 scalar inputs) from -127 through log2(6), every E2M1 midpoint, two neighboring FP32 values on each side, and causal `-inf`.

| Candidate | max abs exp error | max relative exp error | nibble mismatches vs native EX2+cvt | directed mismatches |
|---|---:|---:|---:|---:|
| F1 degree-3 | 4.625320e-4 | 8.778174e-5 | 90 / 2,000,050 = 4.499888e-5 | 10 |
| F2 degree-3 | 4.625320e-4 | 8.778174e-5 | 7 / 2,000,050 = 3.499913e-6 | 7 |
| F2 degree-2 | 9.662628e-3 | 2.075664e-3 | 7 / 2,000,050 = 3.499913e-6 | 7 |

F1 direct classification had zero mismatches versus hardware conversion of its own cubic result. Its additional mismatches versus native Stage2 are real polynomial threshold shifts. F2 degree-3 and degree-2 produced identical direct payload bytes; all seven mismatches versus native were in the constructed threshold-adjacent set. F3 bytes exactly matched F2 degree-3, the PRMT/DP4A half-unit sum had zero mismatches, and all causal `-inf` paths produced code/value zero.

Per-pair probe SASS:

| Probe | MUFU | packed FFMA2 | F2FP | float compares | integer compares | data branches |
|---|---:|---:|---:|---:|---:|---:|
| native EX2+cvt | 2 | 0 | 1 | 0 | 1 | 0 |
| F1 | 0 | 3 | 0 | 4 | 17 | 0 |
| F2 degree-3 | 0 | 3 | 0 | 4 | 17 | 0 |
| F2 degree-2 | 0 | 2 | 0 | 4 | 17 | 0 |
| F3 classification | 0 | 0 | 0 | 4 | 17 | 0 |
| F3 x8 sum | 0 | 0 | 0 | 0 | 1 | 0; 2 PRMT + 2 IDP.4A |

Artifacts: `forward_stage2_fp4_native_probe.cu`, `forward_stage2_fp4_native_probe_gpu2.json`, `forward_stage2_fp4_native_probe_{native,f1,f2d3,f2d2,f3,f3_qsum}.sass`, and `forward_stage2_fp4_native_probe_resources.txt`.

### Kernel Resources And SASS

F1/F2 were integrated at the winning 4-of-16 cadence. Pair bytes were combined directly into the existing 32-bit payload order; unselected native pairs retained one hardware pair conversion. Two register-pressure formulations were tested: immediate row-sum accumulation and a fused range/cubic/classifier PTX block.

Best ptxas result observed for each route:

| Route | Registers/barriers/smem | Stack | spill stores/loads | Gate |
|---|---|---:|---:|---|
| F1 e16 | 168 / 2 / 1904 B | 40 B | 64/68 B | fail |
| F2 degree-3 e16 | 168 / 2 / 1904 B | 24 B | 40/36 B | fail |
| F2 degree-2 e16 | 168 / 2 / 1904 B | 40 B | 68/72 B | fail |
| F3 all-direct | 168 / 2 / 1904 B | 0 B | 0/0 B | pass |

The fused F2 formulation did not help the full kernel and increased its spill traffic to 64/68 B; it was the formulation used in the final timing matrix below. All F1/F2 formulations fail the addendum's no-spill retention gate.

Static whole-kernel instruction counts from that timing binary:

| Route | MUFU | FFMA2 | F2FP | E2M1 | ISETP | FSEL | SEL | IDP.4A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage2 | 257 | 0 | 144 | 128 | - | - | - | 0 |
| F1 e16 | 193 | 96 | 112 | 96 | 794 | 531 | 795 | 0 |
| F2 degree-3 e16 | 193 | 96 | 112 | 96 | 730 | 531 | 795 | 0 |
| F2 degree-2 e16 | 193 | 64 | 112 | 96 | 794 | 531 | 795 | 0 |
| F3 | 1 | 0 | 16 | 0 | 2330 | 1299 | 2139 | 64 |

F3 therefore did remove the intended P-loop EX2 and conversion work. The residual one MUFU and 16 F2FP instructions are unrelated operations outside that payload loop. Its replacement predicate/select network is much larger than the specialized SFU/conversion path.

### Addendum Timing

Clean instrumentation-off repeated timing used 30 samples per route. Each cell gives first-order p50 and reverse-order p50, with the corresponding delta from matched Stage2:

| Shape | Stage2 | F1 e16 | F2 degree-3 e16 | F2 degree-2 e16 | F3 |
|---|---:|---:|---:|---:|---:|
| h4/s2048 | .065632; .064256 | .114592 (+74.6%); .113152 (+76.1%) | .111072 (+69.2%); .110656 (+72.2%) | .111968 (+70.6%); .111728 (+73.9%) | .235568 (+258.9%); .235120 (+265.9%) |
| h8/s1024 | .049728; .046560 | .071104 (+43.0%); .070896 (+52.3%) | .069280 (+39.3%); .070064 (+50.5%) | .070096 (+41.0%); .070832 (+52.1%) | .131648 (+164.7%); .132512 (+184.6%) |
| h8/s4096 | .101728; .103888 | .209440 (+105.9%); .212352 (+104.4%) | .204400 (+100.9%); .207296 (+99.5%) | .204992 (+101.5%); .207936 (+100.2%) | .464816 (+356.9%); .468080 (+350.6%) |

All routes were finite. Addendum run-to-run output deltas were at most 1.220703e-4 across the required matrix, below the 4.882813e-4 matched Stage2 delta observed at h8/s4096. No neighboring cadence was justified: e16 already adds 39-106% for F1/F2, and the allowed neighboring e12 cadence would execute more of the expensive direct classifier. No route won independently, so the task gates correctly skipped h16 winner timing, pchainc combination, and winner NCU profiling.

### Addendum Interval Attribution

Median sparse P-chain cycles:

| Shape/route | scale -> exp/pack | exp/pack -> payload bytes | scale -> P-ready |
|---|---:|---:|---:|
| h4/s2048 Stage2 | 1460 | 227 | 2010 |
| h4/s2048 F1 / F2d3 / F2d2 | 2629 / 2659 / 2651 | 270 / 321 / 267 | 3234 / 3332 / 3251 |
| h4/s2048 F3 | 12628 | 444 | 13445 |
| h8/s1024 Stage2 | 1452 | 229 | 2004 |
| h8/s1024 F1 / F2d3 / F2d2 | 2630 / 2599 / 2657 | 273 / 320 / 270 | 3242 / 3254 / 3262 |
| h8/s1024 F3 | 13061 | 472 | 13851 |
| h8/s4096 Stage2 | 1458 | 231 | 2011 |
| h8/s4096 F1 / F2d3 / F2d2 | 3351 / 3300 / 3334 | 312 / 327 / 316 | 4003 / 3967 / 3985 |
| h8/s4096 F3 | 12551 | 463 | 13343 |

This is not timing noise: direct classification roughly doubles the F1/F2 target interval, while F3 expands it by about 8.6-9.0x despite eliminating EX2 and F2FP. Specialized SM100a MUFU plus E2M1 conversion is materially cheaper here than the convergent integer/predicate replacement.

### Addendum Numerics And Decision

F1 retained the same Stage2 comparison envelope as standalone e16. F2 degree-3 reduced candidate-vs-Stage2 relative L2 to about 1.63e-4 to 1.78e-4 while keeping LSE max abs at 1.86e-5 or below. Degree-2 increased relative L2 to about 1.06e-3 to 1.16e-3 and LSE max abs to about 4.77e-4.

F3's changed denominator semantics produced output max abs 0.2383, relative L2 0.165-0.171, and LSE max abs 0.286-0.398. Its BF16 output RMSE changed little, but BF16 LSE max abs degraded to 0.286-0.398; it is not an acceptable accuracy trade even before its large slowdown.

Reject F1, both F2 degrees, and F3. Their selectors, configs, and kernel behavior were removed. The probe-only helper primitives remain uninstantiated in `stage2_ex2_alu_helpers.cuh` so the focused validation artifact is reproducible; no production dispatch references them.

## Cleanup And Final State

- Removed cadence-only e16/e12/e10/e8/all-emulated selectors after the sweep.
- Removed both interleaved packing selectors and behavior.
- Removed the standalone e16 selector because it failed the high-head determinism gate.
- Removed all addendum F1/F2/F3 selectors, configs, and static kernel behavior after they failed the interval/performance gates; F1/F2 additionally failed the no-spill gate.
- Retained the shared helper only because `e16pc` uses it.
- Retained prior `pchainc` independently and retained one new explicit `e16pc` selector in both dispatch paths.
- `forward_stage2_pchain_driver.py` now reports relative L2 and exposes only `stage2`, `c`, and `e16pc`; `py_compile` passes.

The final forced rebuild succeeded with:

`MXFP4_FWD_TIMELINE=0 MXFP4_FWD_PCHAIN_STAMPS=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 KPIPE_SELECTIVE_POLICY=0 POLICY126_COUNTERS=0`

The final post-addendum restored h4/s1024 smoke selected ordinary Stage2, was finite, and measured p50/min .041872/.039744 ms. Timeline raw/decoded counts were 0/0, every policy counter was zero, and a separate P-chain check returned an empty raw run with no nonzero intervals. The retained specialization still compiles at 168 registers, two barriers, 1904 B smem, and zero stack/spills in the default/off build.

Primary final artifacts:

- `forward_stage2_ex2_alu_winner_{timing,repeat}_{h4_s2048,h8_s1024,h8_s4096,h16_s4096}_gpu2.json`
- `forward_stage2_ex2_alu_pack_combo_stamps_{h4_s2048,h8_s1024}_gpu2.json`
- `forward_stage2_ex2_alu_combo_stamps_h8_s4096_gpu2.json`
- `forward_stage2_ex2_alu_winner_{stage2,e16,e16pc}_kernel.sass`
- `forward_stage2_ex2_alu_ncu_{stage2,e16,e16pc}_h8_s4096_gpu2_{compact,pipecount}.csv`
- `forward_stage2_ex2_alu_restore_default_off_build.log`
- `forward_stage2_ex2_alu_restore_default_off_h4_s1024_gpu2.json`
- `forward_stage2_ex2_alu_restore_default_off_pchain_h4_s1024_gpu2.json`
- `forward_stage2_fp4_native_{timing,repeat,stamps}_{h4_s2048,h8_s1024,h8_s4096}_gpu2.json`
- `forward_stage2_fp4_native_{f1e16,f2d3e16,f2d2e16,f3}_kernel.sass`
- `forward_stage2_fp4_native_restore_default_off_build.log`
- `forward_stage2_fp4_native_restore_default_off_h4_s1024_gpu2.json`
- `forward_stage2_fp4_native_restore_default_off_pchain_h4_s1024_gpu2.json`

## Decision

Keep `e16pc` as the sole new explicit/default-off candidate. It meets the stated interval, repeatability, resource, numerical, determinism, and NCU redistribution gates. F1/F2/F3 do not challenge that decision: direct nibble production costs more than the SM100a specialized path, and F3 also changes denominator semantics unacceptably. Do not promote `e16pc` globally without a broader model-level shape sweep, but it is a real Stage2 P-chain winner rather than a timing-only fluctuation.

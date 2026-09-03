# Session 6 Task: Stage2 Mixed ALU EX2 Emulation And FP4 Packing

Work in `/workspace/codebases/pv/fp4_matmul`.

## Goal

Reduce the dominant Stage2 P scale-to-exp/pack interval by porting and tuning the mixed ALU/SFU exp2 strategy used by local Cute FA4, then scheduling FP4 E2M1 packing around it.

Current measured facts:

- Stage2 scale-to-exp/pack is about 1.35k cycles and is the largest local P block.
- Exact two-pack source restructuring did not alter the generated instruction schedule.
- Stage2 SASS contains about 257 `MUFU.EX2`, 144 `F2FP`, and 128 E2M1 conversions.
- Candidate `pchainc` is retained default-off and gives a small long-shape win; Stage2 remains the global reference.

Do not retry log-domain compare ladders, quantized-denominator ladders, four-way sums, K64-half readiness, or generic scheduler policies.

## Source Of Truth

Use the local FA4 implementation, not a newly invented approximation:

- `flash-attention/flash_attn/cute/utils.py:24` for `POLY_EX2` coefficients.
- `flash-attention/flash_attn/cute/utils.py:681` for `ex2_emulation_2`.
- `flash-attention/flash_attn/cute/utils.py:702` for the explicit paired PTX sequence `e2e_asm2`.
- `flash-attention/flash_attn/cute/softmax.py:238` for the mixed native/emulated cadence.
- `flash-attention/flash_attn/cute/flash_fwd_sm100.py:358` for FA4 frequency selection.

Degree-3 coefficients are:

```text
c0 = 1.0
c1 = 0.695146143436431884765625
c2 = 0.227564394474029541015625
c3 = 0.077119089663028717041015625
```

Important: FA4 does not emulate every exponent. It replaces selected pairs so SFU and ALU/FMA pipelines are both used. The local D=192 causal Cute route normally disables emulation, so do not assume FA4's frequency is optimal for this MXFP4 kernel; measure it.

## Step 1: Port And Validate The Paired Emulation

Add a route-local device helper equivalent to FA4 `e2e_asm2` for one `float2`:

1. Clamp each log2 input to `-127.0f` using FTZ max.
2. Use the `2^23 + 2^22` round-down trick with packed `f32x2` operations.
3. Extract the fractional part.
4. Evaluate the cubic with three packed `fma.rn.ftz.f32x2` instructions.
5. Reconstruct the power-of-two exponent bits with shift plus integer add.

Preserve FA4's rounding/FTZ semantics. Handle `-inf` causal-mask inputs correctly. Inputs are max-subtracted/prescaled and satisfy the upper-domain assumption.

Before integrating it broadly:

- compile a tiny SM100a probe or one route-local kernel instantiation;
- inspect SASS and prove emulated pairs contain no `MUFU.EX2`;
- compare emulation against native `ex2.approx.ftz.f32` over representative softmax log2 inputs, including threshold-adjacent values that could change E2M1 codes;
- report max absolute/relative exp error and FP4 code mismatch rate.

## Step 2: Mixed Cadence Sweep

Add explicit default-off Stage2 configs that apply emulation only in the P payload exp/pack loop. Keep `acc_scale` and unrelated exponentials unchanged.

The selection must be compile-time and convergent: no runtime branch or warp divergence. Map the static `(qid, pack, kk)` position to a linear pair index.

Test at least these degree-3 cadences:

- emulate 4 of every 16 pairs (25%);
- emulate 4 of every 12 pairs (33%);
- emulate 4 of every 10 pairs (40%, FA4 causal cadence where enabled);
- emulate 4 of every 8 pairs (50%);
- all native as reference;
- all emulated as a diagnostic ceiling, not an assumed winner.

If compile time is excessive, use a single integer trait and instantiate the smallest matrix needed to cover these ratios. Do not change the global default selector.

For every cadence, inspect SASS counts for `MUFU.EX2`, packed FMA/ALU reconstruction instructions, and E2M1 `F2FP`. Verify ptxas registers, stack/spills, barriers, and smem.

## Step 3: Packing Schedules

For the best one or two mixed cadences, test bounded packing schedules:

1. Existing pack: produce four `float2` exponent results, accumulate row sum, then call the current 8-value E2M1 pack.
2. Interleaved pack: after enough independent native/emulated pairs have been issued, pack a completed four-pair group while later emulated pairs use ALU/FMA. The intent is to overlap F2FP with remaining SFU/ALU work.
3. Optional fused inline-PTX group helper only if SASS proves the compiler will not interleave the existing form. It must return both exponent values for row-sum semantics and the exact E2M1 payload word.

Do not bypass exponent values with a compare ladder. Do not change denominator semantics. Preserve current accumulation order initially; if any reordering is tested, treat it as a separate approximate candidate and report it clearly.

## Step 4: Correctness And Timing

Use Stage2 as the primary baseline. Also combine the winning exp/pack cadence with retained `pchainc` only after it wins independently.

Validate:

- finite h4/s1024 smoke;
- deterministic repeated output;
- candidate-vs-Stage2 output max abs, relative L2, and LSE max abs;
- candidate-vs-BF16 reference numerical envelope;
- payload-word mismatch rate versus native Stage2 for representative tiles;
- h4/s2048, h8/s1024, h8/s4096 repeated/interleaved p50 and min;
- add h16/s4096 for a higher-head read on any winner.

Use the existing sparse P-chain stamps to prove scale-to-exp/pack moved. A wall-time result without interval movement is noise until repeated.

## Step 5: Profile A Winner

Only profile a cadence that repeats a wall-time gain. Compare against Stage2 with stamps/timeline disabled:

- duration;
- MUFU/SFU utilization and instruction count;
- FMA/ALU utilization;
- F2FP instruction count;
- issue active and eligible warps;
- long scoreboard and wait;
- TC/tensor-pipe activity.

The expected success signature is lower MUFU pressure/latency and shorter scale-to-exp/pack without creating FMA-pipe saturation, spills, or a longer pack dependency chain.

## Acceptance

Keep a route only if:

- the intended interval decreases;
- wall time repeats on at least two supporting shapes or has a defensible shape gate;
- no stack/spills are introduced;
- numerical error remains acceptable and deterministic;
- NCU confirms real pipeline redistribution for the winner.

Reject and remove non-winning route selectors/behavior. Keep useful helpers only if they are used by a retained explicit candidate.

## Deliverables

Write:

`results/mxfp4_fa4_forward_recover_20260617/forward_stage2_ex2_alu_pack_20260711_report.md`

Append a concise ledger entry to:

`results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_ledger.md`

Restore default/off at the end, run a finite smoke, and leave this Codex session open.

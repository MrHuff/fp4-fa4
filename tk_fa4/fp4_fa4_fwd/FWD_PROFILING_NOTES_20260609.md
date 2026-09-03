================================================================================
!! RETRACTION + CORRECTED STATUS (2026-06-10, session 27) - READ THIS VERSION
!!
!! The earlier banner here claimed production FP4 QK quantization is
!! "effectively global-scale-only" with a ready fix cluster. THAT CLAIM IS
!! RETRACTED: it rested on a consumption model that hardware experiments have
!! now shown to be inconsistent. DO NOT act on the previous fix cluster
!! (d2 h*3->h, k-depth t->2t) without the resolution below.
!!
!! WHAT REMAINS BYTE-VERIFIED (real, unexplained):
!!   (a) The k_sc/q_sc TMA with d2 = h*QK_SCALE_CHUNKS loads ALL-ZERO tiles
!!       for heads >= 4 (verified across the whole 1536B tile) and head-3h's
!!       pages for h = 1..3, while the host tensor is fully populated.
!!   (b) Zero e4m3 scale-factor bytes are TRUE MULTIPLICATIVE ZERO in the
!!       MMA (sfa_k192 mode-2: D == 0 exactly, both N).
!!   (c) Production and the bit-identical fused nb128 match torch refs at
!!       the 2.4e-4..2.9e-3 floor for ALL heads.
!! (a)+(b)+(c) are mutually inconsistent with "the QK MMA consumes the staged
!! k_sc_tm as its SFB" - at least one link of the staging/consumption model
!! is wrong (what the MMA's SFB actually reads in the real kernel, or what
!! the staging places where). Until that is resolved with a real-kernel
!! marker experiment, NO claim about production scale-consumption quality
!! can be made in either direction.
!!
!! Status of the related findings: the nb64 numeric gap (dLSE ~2e-2) and its
!! suspects (K-nibbles / 64-granular recurrence) are tracked in sessions
!! 26-27; the SF consumption-map micros remain valid AS micro results.
================================================================================

# FP4 FA4 Forward: Profiling + Optimization Notes (2026-06-09)

Scope: `kernel_streaming_live_fp4pv`, production config `dualaccum_directrescale_q208_p96_o48`
(H>4 lane), shape S=4096/B=1/H=12/Dqk=192/Dv=128 causal, fullgrid, GB200 (sm_100a), cuda:3.
Pre-session baseline (this session, harness): FP4 0.95–0.77x BF16 across S=2048..8192; 1.12x (slower) at S=16384.

## Where the cycles go (ncu, full set, S=4096 fullgrid)

- Duration 115.6 us, 130,685 elapsed cycles @1.12 GHz, 25.15M instructions.
- **Tensor pipe 3.4–7.3% of peak** (metric family dependent). The FP4 MMAs are idle >90%
  of the time; the kernel achieves ~13% of FP4 tensor peak (~670 TFLOPS at S=4096).
- No pipe saturated: ALU 23–26%, XU/MUFU ~16%, LSU low, DRAM 1.2%. Issue 0.37/cycle/scheduler,
  2.65 active warps/scheduler (1 CTA/SM, 12 warps: producer/output/quant 1 warp each per scheduler).
- Stall mix per issued instruction (7.14 cy): long_scoreboard 3.45 (mbarrier spin-waits classify
  here on Blackwell, plus TMEM/L1 deps), wait (fixed-latency chains) 1.30, short_sb 0.44.
- Top stall sites (PC sampling): OUTPUT WG waiting `corr_arrived` (17.6%) = idle, not critical path;
  QUANT `wait_for_scores`; per-CTA prologue wait + `CCTL.IVALL`; P-payload `STS.128`; producer TMA waits.
- 2.3-way shared bank conflicts on 48% of shared-load wavefronts (est. 6%); partial-wave tail
  flagged up to 33% (2.53 waves, block-limit 1 CTA/SM due to regs+smem).

**Conclusion: latency/serialization-bound.** The per-iteration software pipeline is one stage deep
(QUANT(i) || OUTPUT-rescale(i-1)+PV(i-1)), and the longest stage is the QUANT warpgroup's serial
chain: 4x `tcgen05.ld` 32x32b.x32 (one full 128-col row per lane) -> 36-op max tree -> 64 FFMA2 +
128 MUFU.EX2 + 64 FADD2 + pack + 4 STS.128 per thread per tile (~440 instr). Proof of
latency-boundness: the coeff fold below cut instructions 19% but cycles only ~2%.

## Changes kept

1. **Fixed-coeff fold into exp2 bias** (`STATIC_ONLINE_MXFP4_FOLDED_P_COEFF`, plain online
   fixed-scale path only; streamscore/localmax/split-2WG keep the old mul+cvt):
   - `neg_max_scaled += log2(COEFF)` (COEFF = 6*2688/4096 = 3.9375); ZERO_SCALE branch biases
     its exp argument the same way.
   - P pack is now a pure `cvt.rn.satfinite.e2m1x2.f32` x4 (`fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte`);
     bf16 alt path passes coeff 1.0; EARLY_CORR group helper converted likewise.
   - row_sum then carries COEFF; compensated via `MXFP4_ALPHA = COEFF/36` in both epilogues and
     `log2_row_sum -= log2(COEFF)` in both LSE stores.
   - Validation: vs pre-fold .so refs max|dO| <= 3.2e-3 (kernel's own run-to-run nondeterminism
     floor is ~5e-3), max|dLSE| <= 2e-6. ncu: instructions 25.15M -> 20.38M (-19%),
     cycles 130,685 -> ~127.9-128.7k (-1.5..-2.1%).
2. **Four new runtime configs registered** (dualaccum_directrescale_earlycorr[_splitk64][_pstage3]
   _q208_p96_o48) enabling EARLY_P_READY + PRE_P_QUANT_CORR_PUBLISH (+ first genuinely active SPLIT_P_READY_K64)
   on the dualaccum family; static assert at fwd_streaming_kernel.inc:774 relaxed to allow
   direct-after-rescale dual mode. **Outcome: slower** (+2..8% vs base at S=4096/8192) — kept as
   registered options/documentation; production default unchanged. Reason: the overlap they add
   already existed one pipeline stage deep; they add a per-iteration QUANT-side reuse wait.

## Measured negative results (do not redo)

- `earlycorr`/`PRE_P_QUANT` on dualaccum: +2..8% slower (above).
- `streamscore` family (`dualaccum[_forcespare]_streamscore[_half]_q208`): +3..6% slower.
- `localmax_split2wg` at H=12: +15% slower (stays the H<=4 lane).
- `mb64_nb64` / `nb64` variants: 2-6x slower at H=12.
- `decoupled_pstage4_stagev`: parity with base (within noise).
- Small but real wins available in the registered space: `q224` (-2.1%) and `pstage4`
  (-1.8..-2.5% at S=8192); `decoupled_pstage4_q224_p96_o48` untested in combination.

## Bugs found

1. **Persistent-mode deadlock (production q208 family), root-caused**: the QUANT task-start
   `wait(rescale_finished[0], rescale_phase)` toggles parity once per task, but OUTPUT arrives that
   semaphore (iters-1) times per task. A persistent CTA that picks up a second task spins forever
   when the leftover phase count is even. Repro: any tasks > grid (304), e.g. S=4096 H=12 (384),
   q208 H=4 S=16384 (512). S=2048 H=12 (192) fine. The H<=4 `localmax_split2wg` lane is immune
   (EARLY_P_READY skips that wait). NOTE: auto launch-mode selection currently picks persistent for
   H>4 & S<8192, so **auto at H=12 S=4096 hangs today**; use fullgrid explicitly.
   A cumulative-parity drain fix at the wait removed this deadlock but exposed a second,
   *intermittent* task-boundary hang elsewhere in the dualaccum/directrescale family (April-era
   fp32pack persistent multi-task worked, so the regression is family-specific). Fix reverted;
   comment left at the wait site. Needs a dedicated session auditing all per-task phase state in
   PRODUCER/OUTPUT lanes for this family.
2. **Output nondeterminism**: production config differs run-to-run by max|dO| ~5e-3 (LSE exact);
   `localmax_split2wg` differs in LSE by ~4.5e-2 run-to-run (~0.6% of rows). Pre-existing.
3. `dualaccum_directrescale_pstage3_q208_p96_o48_splitk64` registers SPLIT_P_READY_K64 without
   EARLY_P_READY, so the flag is compile-time dead -> identical to plain pstage3.
4. `benchmark_fp4_vs_bf16_canonical` is broken (routes `forward_streaming_live_localcta` to the
   bwd experiments module).
5. GPUs 1/2 stuck at 100% util with no processes (likely wedged persistent kernels from earlier
   runs; see bug 1). May need reset; also makes wall-clock benchmarking noisy (use ncu cycles or
   same-process ratios).

## Ranked next steps toward the 4x gap (tensor pipe is >90% idle)

1. **Deepen the pipeline beyond one stage** so the QUANT stage stops bounding the period:
   the running-max recurrence serializes tiles, but exp/pack(i) can overlap LDTM+masktree(i+1)
   if score regs are halved (Nb=64 sub-tiles within the Nb=128 handshake, not the nb64 config) —
   register budget is the constraint (64 float2 already live at q208).
2. **Persistent multi-task fix** for the dualaccum family (kills the 2.53-wave prologue/tail, the
   33% ncu tail estimate, and the S=16384 cliff; also un-breaks auto mode).
3. **Latency diet in QUANT**: 4 sum accumulators instead of 2 (halves the FADD2 chain depth),
   earlier first-EX2 (compute max of cols 0..63 while LDTM of 64..127 lands), bank-conflict fix for
   the 2.3-way shared-load conflicts.
4. **Second CTA per SM** (needs Nb=64-regs quant or split quant across two WGs without the localmax
   accuracy/handshake cost) for TLP-based latency hiding — largest structural lever, hardest.
5. Re-test `q224` + `pstage4` (+ maybe `decoupled`) as the new default once measured on quiet GPUs.

---

# Session 2 (2026-06-10): persistent multi-task fix attempt + pipelined quant body

Both work items landed as gated-off code with full forensics; the shipped kernel is
behaviorally identical to the 2026-06-09 coeff-fold build (verified: ncu 129.0k cycles /
20.38M instructions at S=4096 fullgrid; dO vs pre-fold refs at the ~5e-3 nondeterminism
floor; LSE <= 2e-6).

## Persistent multi-task (item 2) — status: root-caused deeper, still gated off

Debug tooling that worked well (reuse this):
- cuda-gdb attach to the hung pid: `info cuda warps` per SM gives each warpgroup's stuck
  PC; `info line *ADDR` decodes the spin loops back to fwd_streaming_kernel.inc lines
  (build has line info). Spin loops are `SYNCS.PHASECHK.TRANS64.TRYWAIT` + `@!P0 BRA`;
  the phase argument register holds parity<<31; `print *(@shared unsigned long long*)ADDR`
  reads mbarrier words (bit63 = completed-phase parity; pending arrives visible as
  field deltas vs the pristine word 0x7ffff800001ffffe; one warp::arrive = -2 low / -0x800 high).
- LSE side-channel: call ext.forward_streaming_live_mxfp4 directly with a NaN-prefilled
  LSE tensor, then read per-(head, m_tile) completion from the hung process via cuda-gdb
  `print *(@global unsigned int*)(lse_ptr + 4*(h*S + m*128))`. This produced the decisive
  completion map.

What was established:
1. The legacy QUANT task-start `wait(rescale_finished[0])` parity bug (session 1) is real.
2. `p_stage_reusable` gating for !EARLY was per-task (`idx >= P_STAGE_SLOTS`) with no
   tail compensation -> cumulative drift; FIXED via a per-slot first-use mask in
   `wait_for_p_stage_reusable` (kept in tree; currently dead code for production since
   the !EARLY generic body never calls the lambda - payload safety in production is
   pure timing slack, see item-1 notes).
3. `p_copy_done` (score-slot reuse) is NOT broken: the task tail advances
   `p_copy_phase_mask` for the final SCORE_TMEM_SLOTS iterations without waiting
   (transitively ordered through the PV issue path), so the per-task skip is correct
   across persistent tasks. A "wait unconditionally from task 2" strengthening is
   WRONG (double-books phases) and also wrong as a per-slot-first-use mask
   (the tail-flips already account for those arrives). Leave it alone.
4. A complete persistent corr protocol was implemented (OUTPUT pre-loop seed +
   per-rescale + post-epilogue arrives; PV waits every issue; QUANT lag-1 per-publish
   waits; count-aligned for any task-size mix including iters==1). It is gated off by
   `STATIC_ONLINE_MXFP4_PERSISTENT_CORR_PROTOCOL = false && ...` because a THIRD,
   pre-existing bug remains: for an iters==1 task, corr_arrived[0] was observed stuck
   at 7-of-8 arrives (phase 1 at 3/4 warp arrives) while all four quant warps had
   advanced to the next task's scores wait - i.e. one quant warp skipped one of its
   two t1 corr publishes. The LSE completion map showed every small task (m<=6)
   wedging this way at S=4096 H=12 persistent. This needs an instrumented build to
   localize (suspect warp-divergent control flow between the per-iteration publish
   and the end-of-task stats arrive).

## Pipelined quant body (item 1) — status: implemented, gated off pending a TMEM question

`STATIC_ONLINE_MXFP4_PREFETCH_SCORES` body (self-contained variant block in the QUANT
loop): loads + causal-masks tile idx+1's scores during tile idx's pack phase, with the
explicit `wait_for_p_stage_reusable` backpressure (needed: the legacy path's payload
safety is timing slack that the pipeline compresses away - removing the wait corrupts
outputs at max|dO| ~7).

Unresolved: with prefetch active, small tiles (iters 2..7) produce deterministic
max|dO| ~0.08-0.12 on even-bid CTAs (bit-exact across runs; LSE exact). Bisects ruled
out pack-loop WAR on tcgen05.ld destinations (tail-batched loads corrupt identically),
early copy-done release (consume-time release identical), and P-stage backpressure
(fixes the ~7 -> ~0.08 component only). iters==1 tasks corrupt at ~4.8 even though
their runtime path is the same as a clean bisect build - still unexplained; they are
routed to the legacy body as a workaround. Remaining suspect: the early tcgen05.ld of
the next score slot overlapping in-flight tensor-pipe work (commit/visibility ordering
for cross-warpgroup TMEM reads issued one iteration early). Worth asking the TK authors
about tcgen05.ld vs mma commit ordering guarantees here.

Expected gain if fixed: the score-load + mask + maxtree segment (~200-300 cy of the
~1100-1500 cy iteration) overlaps the previous tile's exp/pack -> est. 10-20% on the
QUANT-bound steady state.

---

# Session 3 (2026-06-10, cont.): pipelined-body corruption ROOT-CAUSED

**compute-sanitizer racecheck found it in one run** (use this tool first next time):

    Error: Potential RAW hazard at __shared__ 0x484:
      Write Thread (161) stats_store  (fwd_bf16_baseline.inc:490)  <- quant WG corr publish
      Read  Thread (33)  stats_load   (fwd_bf16_baseline.inc:480)  <- correction WG rescale read

i.e. **corr_vec_smem[0] is a single slot protected only by timing slack**: the quant WG's
publish of corr(idx+1) races the correction WG's read of corr(idx). The legacy body has the
same latent race (one full quant-iteration of slack hides it); the pipelined body shortens
the iteration and loses. This also explains why LSE stayed bit-exact while O corrupted
(corrections are consumed only by the output rescale), why magnitudes were small
multiplicative (~0.08-0.16), and why it was deterministic per shape/data.

Disproven along the way: V-scale TMEM race (constant-v_sc A/B corrupts identically),
ZERO_SCALE head parity (no heads have sg product 1.0), tcgen05.ld WAR/commit-ordering
theories. TK source facts confirmed: init_semaphore(bar, tc, tx) -> expected = tc + tx;
warp::/warpgroup::arrive = ONE elected-lane mbarrier.arrive; wait(sem, p) spins while
try_wait.parity(p) fails; SASS parity operand = p << 31.

Fix attempt: per-publish `wait(rescale_finished[0])` for idx >= 2 (continuing the
task-start wait's phase; count-aligned with the (iters-1) arrives, incidentally also
parity-correct across persistent tasks for iters >= 2). It eliminates the race window by
construction but **deadlocks at runtime** (QUANT at a scores wait, lane at a
p-ready wait; cycle not identified - every paper trace of the
publish->rescale->arrive chain grounds out, so the actual blocking edge is in a
producer-side wait whose pairing differs from my model). Two candidate paths forward:
1. Double-buffer corr_vec instead of waiting: set STATIC_ONLINE_MXFP4_CORR_SLOTS=2 for the
   prefetch family and reuse the existing corr_slot = idx&1 + corr_reuse machinery
   (the EARLY_P_READY family already runs exactly this protocol successfully).
2. Instrument the deadlock (one printf-free progress word per warpgroup in global memory)
   and identify the blocking edge of the per-publish wait.

State shipped: prefetch body gated off (`false &&`), kernel re-validated at the
pre-fold-reference noise floor. The corr-race fix should also be considered for the
LEGACY body (it is a real race; the ~5e-3 output nondeterminism documented earlier may
be this exact race occasionally winning by a hair).

---

# Session 4 (2026-06-10): corr_vec double-buffering SHIPPED; persistent mode FIXED

## What shipped (STATIC_ONLINE_MXFP4_DUAL_CORR_SLOT, enabled for the production family)

Double-buffered correction slots for the non-early fixed-scale dual direct-after-rescale
family, borrowing exactly the EARLY_P_READY corr machinery: corr_slot = idx & 1,
count-gated rescale_finished reuse waits before each publish, direct_rescale_finished[2]
for the PV gate, stats_arrived for the end-of-task stats handshake, and the legacy
task-start rescale wait compiled out. Correction VALUES and all scale/amax math unchanged.

Validation on cuda:3:
- **Run-to-run output nondeterminism collapsed to exactly 0** (6 runs, S=2048 and S=4096,
  bit-identical O and LSE; was ~5e-3). This confirms the racecheck diagnosis: the
  production nondeterminism WAS the corr_vec single-slot race.
- vs pre-fold refs: dO <= 2.9e-3 (inside the old racy envelope), dLSE <= 1.9e-6.
- **Persistent mode fixed as a side effect**: S=4096 H=12 (384 tasks > 304 grid) and
  q208 H=4 S=16384 (512 tasks) now complete reliably (15 hammered launches), are
  bit-identical to fullgrid AND self-deterministic. The two root causes are both gone:
  the parity-broken task-start wait is compiled out for this family, and the end-of-task
  corr_arrived[0] handshake (the 7-of-8-arrives wedge) moved to stats_arrived.
  Auto launch-mode no longer hangs at H=12 S<8192.
- Cost: ncu S=4096 fullgrid 129.0k -> 133.4k cycles (+3.4%), instructions 20.4M -> 21.3M.
  CORR_SLOTS=3 might trim the publish-wait depth if this ever matters.

Persistent vs fullgrid timing (H=12): -5.5% at 4K, +6.4% at 8K, +2.5% at 16K -> fullgrid
remains the right default at 8K+; the 16K ratio cliff is NOT a tail-wave problem
(get_tile_idx already reverses m so long tiles go first).

## Pipelined quant body: status after this session

The corr race fix made the body race-free and bit-deterministic, and a new empirical
constraint emerged from a clean bisection matrix:
- next-tile tcgen05.ld issued AFTER this iteration's payload st.shared -> payload PV
  consumes corrupts deterministically (interleaved, tail-batched, and drained-before-
  proxy-fence variants all corrupt identically; LSE stays bit-exact).
- loads issued BEFORE the stores (iteration top) -> clean.
- consuming scores_arrived(idx+1) early -> harmless (clean).
With per-chunk interleaving therefore off the table, the remaining "A-form" body
(early try_wait capture + top loads) measured **+32% elapsed cycles** vs the generic
body at identical math (176.1k vs 133.4k; same spills; independent of the p_stage
backpressure wait and blocking-vs-try waits) - cause not isolated (suspect: union
register pressure from carrying both bodies behind the iters>1 runtime branch).
Gated off again. To make this pay, rebuild the body around the store/load ordering
constraint (e.g., split pf registers so chunk loads precede all stores of the SAME
chunk set, or get an authoritative answer on why tcgen05.ld after st.shared corrupts
the staged payload - this looks like a real tooling/ISA question for the TK authors).

## Config stack re-test on the dual-corr build (fullgrid, in-process)

q224 / pstage3 / pstage4 / decoupled_pstage4[_stagev][_q224]: all within +/-1.4% of
q208_p96_o48 at S=4096/8192/16384 - the earlier q224/pstage4 ~2% edges do not reproduce
under the new protocol. No default change recommended.

## PC sampling on the shipped build (S=4096 fullgrid, 133.2k cycles)

Stall profile shape unchanged from session 1: top = correction-WG corr wait (idle, 17%),
QUANT scores wait, per-CTA prologue (CCTL.IVALL + first-tile latency, ~7%), payload
STS.128 + 2-way shared bank conflicts (~4%), producer TMA waits; 'wait'-class stalls
spread across the exp/pack math. Tensor pipe still ~3-7% of peak. Next levers unchanged
in priority: (1) legal load/pack overlap in QUANT, (2) prologue trim, (3) shared-store
conflict fix + 4-way sum accumulators, (4) second CTA/SM.

---

# Session 5 (2026-06-10): throughput agenda (A)-(E)

## (A) Auto launch mode - banked
Persistent-vs-fullgrid measured per shape (H=12): -6.4% @1K, +1.7% @2K, -2.5% @4K,
+7.7% @8K, +0.2% @16K. The existing auto rule (H>4: persistent <8192, fullgrid >=8192)
is already the empirically right one; it was simply unusable before the deadlock fix.
Auto-mode sweep confirms correct selection and ratios ~0.9-1.13 vs bf16 (wall clock).
No code change needed.

## (B) A-form +32% regression - ROOT-CAUSED: blocking try_wait
Isolation build (prefetch body unconditional, generic body dead-code-eliminated,
iters==1 routed in-body): regression unchanged (177.5k cycles, same 168 regs/thread)
-> NOT register union pressure. The cause is PTX semantics: `mbarrier.try_wait` is
*potentially blocking* (suspends up to a system-dependent time limit before returning
false); the per-iteration opportunistic probe therefore stalled every iteration whose
QK(idx+1) lagged. Replacing it with `mbarrier.test_wait` (truly non-blocking, inline
asm; TK's try_wait wraps try_wait.parity) recovered 177.5k -> 139.7k. Residual +4.7%
vs the generic 133.4k buys nothing without the load interleave, so the body stays
gated off - but it is now structurally sound, deterministic, handles iters==1
in-body, and the runtime iters>1 branch is unnecessary.

## (C) ld-after-st corruption - SASS fence audit CLEAN; queued for TK authors
Rebuilt the corrupting interleave variant (repro: dO ~3.8, even-head pattern, slight
run2run jitter) and audited the production kernel SASS (cuobjdump -fun):
- all payload STS.128 precede the (tail-clustered) LDTM.x32 group - ptxas preserved
  program order, no store sinking across the loads;
- publish chain after the loads is corr STS -> SYNCS.ARRIVE -> BAR.SYNC ->
  FENCE.VIEW.ASYNC.S -> UTCBAR, as in source;
- the drained variant (tensor_load_wait before the proxy fence) corrupts identically,
  so outstanding-loads-at-fence is also excluded.
Minimal statement for the TK authors: in a Blackwell warpgroup that stores an FP4
tile to shared (generic proxy), fences with fence.proxy.async, and signals a consumer
tcgen05.mma reading that tile through an SMEM descriptor, issuing an unrelated
tcgen05.ld (different TMEM region) AFTER the st.shared but before/after the fence
makes the MMA consume stale/garbled payload bytes - register-side values stay
bit-exact (LSE unchanged), order preserved in SASS, repro deterministic per
shape/data. Loads issued before the stores are clean.

## (D) Latency diet - exhausted, kernel confirmed latency-bound
- 4-way sum accumulators: 133.1k vs 133.4k cycles (no win; FADD chain hides under
  EX2 latency). REVERTED to keep outputs byte-identical.
- Shared bank conflicts: the session-1 hotspot (35k excessive wavefronts) no longer
  exists on the dual-corr build - one residual site, 4.6k excessive wavefronts
  (~0.4%), once per task. Not worth touching.
- CCTL.IVALL is the kernel-EXIT cache control (before EXIT), i.e. teardown drain,
  not a prologue cost; the per-task first-wait is pipeline fill, amortized by
  persistent mode at <8K. Nothing further here.

## (E) Dual-QUANT ping-pong - paper scope (next structural lever)
Design: 4 warpgroups (512 thr): PRODUCER(96) + OUTPUT(48) + QUANT_A + QUANT_B, tiles
ping-pong by idx parity; each QUANT WG does the full 128-col row work (LDTM, mask,
local maxtree, exp, pack, stores) for its tiles; only the running (m, row_sum)
recurrence is exchanged per tile through SMEM vectors (publish+consume, ~2 extra
handshakes/tile, +2KB SMEM). Numerics EXACT: identical m_i / row_sum_i / corr_i
values in consumption order; scale/amax math untouched (the localmax-style
fold-into-scales shortcut is explicitly rejected as it changes scale math).
Register budget: 2x176 + 48 + 96 = 496 <= 512 regs/thread-budget -> fits at q176
per QUANT WG WITHOUT the Nb=64 register diet (April's q176 family proves 64xfloat2
fits in 176); Nb=64 diet remains the fallback and is shared groundwork for 2 CTAs/SM.
Natural fit: the freshly landed dual-corr machinery maps 1:1 (corr_slot = idx & 1
<-> publishing WG identity; per-slot rescale_finished reuse waits already exist).
Expected: QUANT stage halves-ish minus handoff -> est. +20-35% kernel; raises tensor
pipe toward ~10-15%.

## Shipped state after session 5
Identical binary behavior to the session-4 dual-corr build (133.3k cycles @4K,
bit-deterministic, persistent-safe). Auto mode is the recommended default.

---

# Session 6 (2026-06-10): dual-QUANT ping-pong built; compute-bound model FALSIFIED

## What was built (registered config: dualaccum_directrescale_pingpong_q176_p96_o48)

STATIC_ONLINE_MXFP4_PINGPONG_QUANT_2WG: 4 warpgroups (512 thr; OUTPUT=0, QUANT_A=1,
QUANT_B=2, PRODUCER=3; regs 176/176/48/96 = 496), tiles alternate by idx parity; each
QUANT WG owns a fixed score/P-stage/corr slot (= its parity, mapping 1:1 onto the
dual-corr machinery); the exact (running max, running row_sum) recurrence crosses WGs
via two SMEM vectors + pp_m_ready/pp_sum_ready semaphores, one publish+consume per
tile, publish skipped on each task's last tile so per-slot phase pairing is exact
across persistent tasks. m published right after the local max tree (before exp) so
the neighbor overlaps fully; row_sum exchanged after the exp pass.

**Correctness: bit-exact vs the production config on the first build** (dO=0, dLSE=0,
deterministic, S=2048/4096) - the exact-recurrence handoff works as designed, and
scale/amax math is untouched by construction.

**Performance: 144.3k cycles vs 133.3k default at S=4096 fullgrid (+8%).** Stacking
ONLINE_QK_TWO_AHEAD (the qk2 knob, to feed two consumers) made it far worse (274.8k -
qk2's copy-done bookkeeping appears incompatible with the 2-consumer cadence; not
pursued). Config kept registered for experiments; NOT the default.

## The conclusion that matters: the QUANT-compute-bound model is falsified

Four independent attacks on QUANT compute all failed to move elapsed cycles:
- coeff fold: -19% instructions -> -2% cycles
- 4-way sum accumulators: -0%
- A-form restructure (scores-wait off critical path): +5% (after the try_wait fix)
- 2x QUANT warpgroups (this session): +8%

while every PC-sampling profile shows >60% of stall samples in mbarrier PHASECHK spins
across ALL warpgroups and no execution pipe above ~26%. The per-iteration period
(~1100 cy at S=4096) is bounded by the **per-tile synchronization round-trip**:
scores_arrived -> quant -> corr_arrived -> rescale -> rescale/direct_rescale_finished
-> PV issue -> pv_tmem_ready -> slot releases, ~7-9 mbarrier hops per tile, each
costing wakeup+spin latency on 3-4-warps/scheduler occupancy, plus the irreducible
serial segments between hops (LDTM latency, max tree, exp not overlappable with its
own tile's loads). Adding QUANT parallelism adds hops (+2/tile) instead of removing
them - hence +8%.

## Re-ranked path to 2.5x+ (attack the handshake chain, not the math)

1. **Hop elimination / fusion** on the per-tile chain (no numerics impact):
   e.g. fold corr_arrived into the payload-ready publish (one publish, OUTPUT reads
   corr behind the same phase), collapse rescale_finished + direct_rescale_finished
   (PV can key off one), revisit whether OUTPUT's wait_pv -> releases can ride the
   PV tensor-commit directly. Each removed hop is worth ~50-150 cy/tile (~5-15%).
2. **Deeper rings so hops overlap across tiles**: 3 score slots requires the Nb=64
   TMEM economics (3x64 scores + 128 out + scales fits; 3x128 does not with
   dual-accum); same groundwork as 2 CTAs/SM.
3. The TK-authors ld-after-st answer still unlocks ~150 cy/tile of load overlap.
4. 2 CTAs/SM (TLP over the latency) - now likely the strongest single lever since
   it amortizes hop latency with a co-resident pipeline; needs the register diet.

## Shipped state
Default unchanged and verified: dual-corr q208, 133.7k cycles @4K, bit-deterministic,
refs clean, persistent-safe. Ping-pong available via
TK_FA4_FP4PV_FWD_CONFIG=dualaccum_directrescale_pingpong_q176_p96_o48.

---

# Session 7 (2026-06-10): hop-fusion campaign - measurements and the structural decision

Goal per plan: fuse numerics-free hops one at a time, measure each, and use the
cumulative result to decide deeper-rings vs 2 CTAs/SM.

## Fusion 1: corr publish moved off the QUANT tail (earlycorr / PRE_P_QUANT)
Runtime config `dualaccum_directrescale_earlycorr_q208_p96_o48` (composes with
DUAL_CORR; the PRE publish block already carries the per-slot reuse wait).
Result: 135.5k cycles vs 133.3k - **no gain**, and the config is NON-deterministic
(det ~1e-2; the PRE path has a residual race - do not ship). Information value:
corr arrival is NOT what gates the rescale; the OUTPUT loop's wait_pv(idx-1) is.
This pinpointed the true critical cycle:
  PV(idx-1) MMA -> OUTPUT wakeup -> in-place rescale (~300-400cy, chunked through
  o48 regs) -> arrive -> lane wakeup -> PV(idx) -> ...
The rescale is sandwiched between consecutive PV MMAs on the same accumulator.

## Fusion probe set: is the rescale-in-the-gap the period?
- OUTPUT register width A/B (runtime): o40 = 136.9k, o48 = 133.3k, o56 = 133.0k.
  Narrowing the rescale batch clearly hurts; widening is near-saturated.
- **Spare mode** (`dualaccum_q208`, PV accumulates into the score-TMEM spare while
  the rescale happens, merge afterwards - the rescale leaves the PV-PV gap):
  **128.7k cycles (-3.5%)** and wall-clock -1.8%..-4.8% across S=1024-16384.
  This is the direct measurement of the gap hop budget: ~350cy/tile of the ~1100cy
  period is rescale-positioning + its two handshakes.
  CAVEAT: dualaccum_q208 runs the OLD corr protocol (DUAL_CORR requires
  DIRECT_AFTER) - it is non-deterministic (det 6-9e-3) and dLSE vs base up to 8e-4;
  NOT shippable as-is.

## Attempt to make spare mode shippable: DUAL_CORR extension - REVERTED
Widened the DUAL_CORR gate to all dual-accum TMEM families + seed/per-iteration
rescale_finished arrives in the spare-merge OUTPUT loop. Bit-exact when it ran, but
**intermittent deadlock** (cold-context repro ~50%): cuda-gdb forensics (S=1024
block 0) show a triangle through the legacy spare accounting - issue lane waiting
dual_output_spare_reusable[0] parity 1 (one completion present) <- OUTPUT merge <-
OUTPUT stuck at a pv-family wait parity 0 <- PV <- lane. The pre-existing spare /
wait_pv phase bookkeeping interacts with the new arrives in a way that needs the
author's protocol knowledge. All edits REVERTED; shipped state re-verified
(bit-deterministic, refs clean, persistent 5/5, 133.8k cycles, full auto sweep
healthy: 0.90-1.13 vs bf16).

## The decision the campaign was run for
Hop/positioning budget measured at ~350cy/tile (~25-30% ceiling if everything
fusible is fused) out of a ~1100cy period; the remaining ~750cy/tile is
irreducible serial segments (LDTM latency, max tree, exp chain, MMA latency) that
single-pipeline fusion cannot touch. The 2.5x gap therefore cannot be closed by
hop fusion alone -> **2 CTAs/SM is the right next structural lever** (a co-resident
pipeline amortizes BOTH the hops and the serial segments), with the spare-family
protocol fix (author-assisted, worth its ~3.5% on top) and the Nb=64 register diet
as its groundwork. Deeper rings alone rank below it.

## For the TK authors (two precise questions now)
1. ld-after-st payload corruption (session 5 notes - SASS-verified ordering).
2. The spare-family (non-DIRECT dual accum) wait_pv/spare_reusable phase protocol:
   what pairing do the seed and per-iteration rescale_finished arrives need so the
   dual corr-slot machinery can be enabled there? (Triangle forensics above.)

---

# Session 8 (2026-06-10): 2 CTAs/SM - full three-resource budget and verdict

## Paper budget (numbers from code + ncu, production config as baseline)

**TMEM (the gating resource).** tcgen05 pool = 512 cols/SM; TK's
tensor_allocator<nblocks_per_sm,...> already divides it (cols = 512/nblocks,
static_assert-bounded), so a 2-CTA build is allocator-parameterized, 256 cols/CTA.
Production map (Nb=128): scores 2x128 = 256 [0,256) + output 128 [256,384) +
scales 128 [384,512) (q_sc 16, k_sc 16, p_sc 2-3x16, v_sc 2x32) = **exactly 512**.
-> 2 CTAs at Nb=128: impossible, full stop.
Nb=64 candidate: scores 64 (1 slot) + output 128 (Dvo fixed) + scales:
with today's hardcoded widths (p 16/slot, v 32/slot, 2 slots each) = 320 > 256.
Fits only with ALL of: 1 score slot, Nb-aware scale widths (p 16->8, v 32->16 at
Nb=64 - currently hardcoded), and v_sc single-buffered -> exactly 256.
Feasible but requires scale-width code changes; compile-time verifiable.

**Registers.** 64K regs/SM. Production: 168/thread x 384 = ~64.5K (the whole file;
register-limited to 1 CTA today regardless of anything else). 2 CTAs need
q+o+p <= 256 (avg <= 85/thread). Diet configs landed this session:
nb64_q112_p96_o64 (272, 1-CTA tuning) and nb64_diet_q112_p80_o64 (256, the 2-CTA
budget); ncu reports 112/thread padded -> the 85-avg bar likely needs
__launch_bounds__(384, 2) to force ptxas allocation, plus spill watch
(currently 8-48B spills at q112).

**SMEM.** Production 103.8 KB/block (100.35 dynamic + 2.42 static + 1.02 driver).
Nb=64 measured: 77.8 KB dynamic -> 2x ~160 KB <= 228 KB SM budget. Fits; with
1 score slot and single v_sc it shrinks further. Not a blocker.

## The number that kills the lever as scoped

`dualaccum_directrescale_nb64_q112_p96_o64` at 1 CTA, S=4096 fullgrid:
**317.5k cycles vs 133.3k default = 2.38x slower.** This is the hop-bound model's
fifth confirmation: halving Nb doubles the tile count at near-constant per-tile
handshake cost (the period is hops, not work). Even perfect 2x TLP from a second
CTA projects 317.5/2 = ~159k > 133.3k. **2 CTAs/SM via the Nb=64 diet cannot beat
today's default unless the per-tile hop cost is first cut ~2x** - which is the same
hop-fusion wall, now load-bearing for the structural lever too.

ALSO: both nb64 configs are NON-deterministic (det ~3-5e-2, dLSE ~2e-2,
far beyond legitimate 64-col quantization regrouping) - the Nb=64 lane has its own
latent race(s), untouched by dual-corr; needs its own debugging session before any
Nb=64-based work. Registered for experiments; clearly not shippable.

## Standing after sessions 6-8 (the strategic picture)

- Per-tile period ~1100cy = ~350cy hops/rescale-positioning + ~750cy serial
  segments; tensor pipe single-digit %.
- Measured wins available: spare mode -3.5% (needs author's phase protocol to make
  deterministic); o-reg width is saturated at ~48-56.
- Falsified/blocked: QUANT compute diet (x4), ping-pong 2WG (+8%), in-WG load
  interleave (ld-after-st, TK question), earlycorr (racy + no gain), 2 CTAs/SM at
  Nb=64 (2.38x standalone deficit > 2x TLP ceiling), 2 CTAs at Nb=128 (TMEM).
- The remaining big-win shapes all run through ONE design question: get the
  rescale/merge out of the PV->PV serial gap without per-tile rescale_finished
  round-trips. The spare/merge dataflow does exactly this (and its deferred-merge
  generalization would amortize the merge over k tiles); what's missing is the
  author-blessed phase protocol (deadlock triangle documented in session 7).
  Recommendation: take the two queued TK-author questions forward before more
  solo protocol surgery; meanwhile the shipped dual-corr default (bit-deterministic,
  persistent-safe, 0.88-1.13x bf16 wall-clock) is the stable production state.

---

# Session 9 (2026-06-10): spare-triangle solo attempt (root cause #1 found, #2 remains); tcgen05 A-from-TMEM CONFIRMED

## (1) The spare-family deadlock - full forensics

Roles pinned by code-anchored SASS patterns (not layout guesses): issue lane =
thread 256 (the LDS p_nonfirst_mm2_ok guard + spare_reusable wait inside
issue_next_qk); thread 352 = K-stage helper at the k_arrived expect/wait pair;
warp 320 = k_sc stager; QUANT at wait_for_scores (WARPSYNC+R2UR signature);
OUTPUT in the pv/corr wait family (count fingerprints: corr/stats = count-4,
everything else count-1; pristine-word fingerprint table in this file's session-2
notes still applies).

**Root cause #1 (proven):** `pv_tmem_ready[0]` is a single-semaphore one-phase-per-
iteration parity chain. mbarrier parity waits fail permanently on EVEN overshoot:
if PV(k) and PV(k+1) both commit before OUTPUT polls phase k, parity returns to the
waited value and the poll spins forever. The legacy !DIRECT family survives only by
timing slack (QUANT task-start rescale throttle + publish cadence); the dual-corr
extension removed that throttle, letting the pipeline front sprint 2 PVs ahead at
task start -> observed exactly: pv word parity0 with 2 commits, OUTPUT at arg0,
spare[1] zero arrives, lane parked in issue_next_qk(3)'s spare wait, QUANT starved
at scores(3). General protocol law for this codebase: **every single-sem phase
chain needs count-enforced overshoot <= 1; slack is not a protocol.**

**Count-exact pacing built** (publish(idx) waits the rescale_finished arrive posted
at OUTPUT loop entry idx-1 on slot (idx-1)&1; seed posted after wait_pv(0);
last-tile and single-tile tasks post no arrive so per-slot pairing is exact across
persistent tasks; acyclicity and overshoot<=1 derivations in session notes).
Verified active at runtime (QUANT halts at the new pace wait; seed lands). 
**A second cycle remains**: OUTPUT progresses past loop(1) while the entry-arrive
is absent from the expected rescale_finished word and QUANT stays pace-blocked -
a contradiction that requires ground-truth shared-memory layout (the compiler
reorders __shared__ arrays; remote fingerprinting leaves corr/rescale slot order
ambiguous). One serious solo attempt spent; REVERTED to shipped state (verified:
bit-deterministic, refs clean, persistent 5/5). For the author session: the
hung-state word tables, role map, and the pacing design are all in this file;
nd_trap repro is /tmp/nd_trap.py (S=1024 H=12 fullgrid, cold context, ~50%).

## (2) Spare-as-default + deferred k-tile merge: blocked on (1)'s second cycle.

## (3) tcgen05.mma A-from-TMEM for block-scaled FP4: **CONFIRMED LEGAL on sm_100a**

ptxas (CUDA 13.x, sm_100a) accepts and assembles:
  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X
      [d-tmem], [a-tmem], b-desc, idesc, [sa-tmem], [sb-tmem], p;
emitting SASS `UTCOMMA tmem[d], gdesc[b], tmem[a], ..., tmem[scales]`.
Negative controls reject precisely (scale_vec::3X -> unknown modifier; b from
tmem -> argument mismatch), so this is real validation, not rubber-stamping.
TK's tcgen05.cuh has A-from-TMEM wrappers only for f16/f8f6f4 - the block-scaled
A-tmem form simply has no wrapper yet.

Implication: QUANT can tcgen05.st the packed P payload straight to TMEM and PV can
consume it as the A operand - deleting the entire P-through-SMEM roundtrip: the
STS (and its bank conflicts), the proxy fence, the SMEM descriptor, AND the
ld-after-st hazard class that blocks the load/pack interleave. TMEM cost: P in
fp4 = 16 columns per slot (2 slots = 32; current map has exactly 0 free at Nb=128,
so it needs v_sc single-buffering or the p_sc third slot dropped). Open questions
for the prototype: tcgen05.st lane/column layout vs the A-operand packing format
for 4-bit kinds (PTX 9.7.17.10.4), scale_vec::2X vs 4X match with the current
E8M0 scale path, and the QUANT-side fence/commit sequence (tcgen05.wait::st +
fence semantics instead of fence.proxy.async). This is now the top prototype
candidate: it attacks the same ~350cy/tile hop component as the spare/deferred-
merge line, without the author-blocked spare protocol.

---

# Session 10 (2026-06-10): P-in-TMEM prototype - built, bit-identical, measured

## Micro-test first (/tmp/p_tmem_micro.cu, standalone nvcc, ~30s builds)

Settled the open questions on hardware before touching the kernel:
- `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X
   [d],[a-tmem],b-desc,idesc,[sfa],[sfb]` produces CORRECT results on sm_100a
  (reference desc-A path == host simulation exactly; TS path == reference
  **0/16384 bit mismatches with random payloads**). CUTLASS simply has no
  `_TS` atom for block-scaled kinds; the hardware supports it.
- TMEM-A layout for dense fp4 = lane r holds row r as 16 dense 32-bit words,
  bytes identical to the SMEM payload ((y<<4)|x nibbles). 16 columns per
  128x128 P slot. K64 chunks at +0/+8 columns with scale_factor_id 0/2.
- Ordering contract (worked first try, per ISA): producer per thread
  tcgen05.st -> tcgen05.wait::st -> tcgen05.fence::before_thread_sync ->
  warpgroup barrier -> elected mbarrier arrive; consumer: mbarrier wait ->
  tcgen05.fence::after_thread_sync -> MMA.

## Kernel integration (STATIC_ONLINE_MXFP4_P_TMEM_A, config dualaccum_directrescale_ptmem_q208_p96_o48)

TMEM map: v_sc single-buffered via the existing STATIC_SINGLE_V_SCALE_TMEM mode
(extended to the flag; lane-staged cp is tensor-queue-ordered against the
consuming MMA, so single-slot is hazard-free; producer-async excluded by
static_assert) -> P payload slots at columns 480-512 (2 x 16). QUANT stores the
packed words via tcgen05.st (replacing the payload STS), drains with wait::st +
fence::before + one warpgroup barrier, and the PV path consumes A from TMEM
(new helper fp4pv_mm_p_tmem_ABt, two K64-chunk MMAs + commit) after
fence::after on the lane. tcgen05.st into a slot is guarded by the existing
p_stage_reusable tensor-commit backpressure (wait_for_p_stage_reusable).

**Validation: bit-identical to the shipped default** - dO = 0.0, dLSE = 0.0,
deterministic, S=2048/4096, both launch modes spot-checked. The bar held.

## Honest hop accounting (the user's directive 3)

ncu S=4096 fullgrid: default 133.3k -> ptmem 136.0-136.5k (+2.0-2.4%).
Wall-clock deltas +0.6% to +2.7% across 1K-16K (table below). PC sampling:
stall profile shape UNCHANGED (top = correction-WG corr wait ~15%, spin-
dominated, Compute ~33%). Exactly as predicted: the deleted hops (payload STS,
fence.proxy.async, SMEM descriptor) were never on the per-tile critical cycle
(PV -> rescale -> PV), while the added costs (wait::st + fence + one extra
warpgroup barrier + the TMEM-slot backpressure wait) land near it.
P-in-TMEM as a pure swap is ~neutral (+2%): NOT a default change.

## The stack experiment - and a reframing of the old corruption

P_TMEM_A + interleaved prefetch (loads behind each packed chunk): LSE exact,
deterministic, dO ~8 = payload garbage. With P never in SMEM, the SMEM/proxy
ordering theories are eliminated for good: the corruption follows the
interleaved tcgen05.ld itself, pointing at async-ld DESTINATION REGISTER
semantics (ptxas reusing/aliasing in-flight ld dest registers with live pack
values between issue and wait::ld). The session-5 SMEM-era corruption was very
likely the same thing - st.shared was never the culprit. TK-author question #1
is hereby REFRAMED: "what is the register contract for in-flight tcgen05.ld
destinations, and does ptxas honor it across long unrolled regions?"
Interleave reverted; the A-form scaffolding stays in-tree gated off.

## Sweep (vs default, same data, both modes; clocks were boosted this round
## so absolute ms differ from earlier tables - deltas are the signal)

  1024 fullgrid +2.1% / persistent +1.7%; 2048 fullgrid +0.8% / persistent
  +0.6%; 4096 fullgrid +2.7% (ncu 136.0k vs 133.3k). All measured rows
  bit-identical (dO=0, dLSE=0). CAVEAT: ptmem PERSISTENT mode is FLAKY - one
  guarded probe at S=4096 passed, a later unguarded run hung with GPU pegged
  (intermittent; fullgrid never hung). Suspect the tcgen05.st slot-reuse guard
  pairing across task boundaries; needs a trap+gdb pass before persistent use.
  ptmem is registered-not-default, so no production exposure.

## State

Default untouched and re-verified (bit-deterministic, refs clean). ptmem
registered for experiments: correct, bit-identical, +2% at fullgrid (persistent
flaky - see caveat), 16KB SMEM freed (p_fp4_stage now unused on that path -
available for deeper K/V stages later), and the ld-after-st hazard class
structurally removed for any future in-WG pipelining work.

---

# Session 11 (2026-06-10): both walls attacked solo - one fell, one cornered; memo written

**(1) Spare deadlock SOLVED.** Ground-truth SMEM map via init-time printf
(scores c58/c60, corr c68/c70, stats c78, resc c88/c90, pv c98, out_re ca8,
spare cb0/cb8, v_arr cd0, v_fin ce0, p_copy d00/d08, p_quant d20/d28,
p_stage d40/d48, corr_vec 400/600 - note: 0xce0 from earlier rounds was
v_finished[0], NOT k_arrived; the lane identifications stand) + a __device__
debug ring read from the hung process showed OUTPUT never entered the loop:
its corr[0]-arg1 spin was MY SEED, double-consuming corr_arrived[0] because the
non-direct family already has its own pre-loop seed (wait_corr(0)+arrive at
~3974, active when !DIRECT_AFTER). Fix: exclude the paced family from that
pre-block (seed must sit after the first pv poll or publish(1) reintroduces the
parity miss). 10/10 cold repros clean. BUT: count-paced spare = 135.6k vs
133.5k default - the racy -3.5% was under-synchronization, not dataflow.
Spare line CLOSED on merits; deferred k-merge survives as design question Q1.
Residual race in the family (det ~5e-3, dLSE ~5e-4) left undiagnosed.

**(2) Interleave corruption cornered as a COMPILER MISCOMPILE.** Full falsification
matrix (all deterministic, LSE exact, dO byte-identical 8.367/7.996):
- in-loop loads + no fences: corrupt (SMEM payload AND TMEM payload)
- + per-chunk wait::ld (ld-dest theory): corrupt -> falsified
- + per-chunk wait::st too (st-source theory, fully synchronous): corrupt -> falsified
- identical arithmetic, loads at iteration top (A-form + ptmem): bit-identical clean
- SASS order verified correct (session 5)
=> the corruption is keyed to the LOOP SHAPE (uniformly-branched tcgen05.ld inside
the unrolled pack loop), invariant to residence and fencing: ptxas/nvcc miscompile.
Q2 in the decision memo, with the micro harness as the reduction vehicle.

**Decision memo for Robert: DECISION_MEMO_20260610.md** (shipped state, scoreboard,
the two live external questions Q1/Q2, ranked plan). Final tree state validated:
default bit-deterministic at reference floor, ptmem bit-identical, persistent 5/5,
prefetch body gated off in A-form, spare family deadlock-free (registered, not
default).

---

# Session 12 (2026-06-10): Q1 and Q2 ANSWERED LOCALLY - no authors needed

## Q2: the 4-session "ld-after-st" mystery is a ptxas REGISTER MISALLOCATION - root-caused, reproduced standalone, worked around, prize path unblocked

- Standalone repro (/tmp/shape_micro.cu, 30s builds): the production cvt helper
  (4x cvt.rn.satfinite.e2m1x2.f32 into .b8 regs + mov.b32 {b0..b3}) lowers to a
  fused F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C chain; under certain shapes ptxas
  feeds a STALE REGISTER into a chain head (observed literally as R1 with
  STACK:0 - satfinite(garbage-as-float) -> 0 nibble). Host-validated: payload
  byte wrong, register sums right, bit-deterministic - the exact kernel
  signature. Any source perturbation moves/hides it (explains 4 sessions of
  shape-roulette: dO 8.37 in-loop, 1.54 tail-batch, clean at top).
- WORKAROUND (principled de-fusion): fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte_defused
  - one cvt per asm block with cvt.u32.u8 materialization + C-level byte
  combine; avoids the PACK_AB_MERGE_C lowering entirely.
- RESULT: per-chunk interleaved loads + TMEM payload + defused cvt =
  **BIT-IDENTICAL to the shipped default** (dO=0, dLSE=0, det=0). The
  correctness blocker on the pipelined quant body is DEAD.
- Perf status (S=4096 fullgrid): interleave+probe 142.0k vs plain ptmem 136.0k
  vs default 133.5k. Correct but not yet winning: remaining overheads are the
  defused cvt (+instructions), probe plumbing, and probe hit-rate (loads fall
  back to the top when QK(idx+1) has not committed by mid-iteration). This is
  now ordinary tuning, not a mystery. The blocking-wait variant measures 193k -
  the session-5 law (probe must be non-blocking) re-confirmed.
- Tree state: prefetch gate ON for the ptmem config only (registered,
  non-default, bit-identical); production default untouched.

## Q1: spare/deferred-merge - answered by measurement (sessions 9/11 + tonight)

The deadlock fell to local instrumentation (address dump + debug ring); the
count-exact pacing is correct (10/10 cold repros) but costs more than the gap
saves (135.6k vs 133.5k). The racy -3.5% was under-synchronization. The
deferred-k-merge would need its k-fold amortization to beat the pacing tax
plus fix one residual race - de-prioritized below the now-unblocked interleave
line. No author input required for any of this; the memo's "external
questions" are hereby retired (Q2 may still be worth an NVIDIA bug report as a
community service - the repro is ready - but nothing blocks on it).

## Next session (pure tuning, no unknowns)

1. Interleave overhead diet: measure defused-cvt cost in isolation (swap into
   the generic body for one ncu run); probe hit-rate counter; consider probing
   LATER in the iteration (after the first pack chunk) to raise hit rate;
   consider keeping fused cvt for the top-load path and defused only in-loop.
2. If interleave beats 133.5k: promote ptmem+interleave config toward default
   (full 1K-16K sweep both modes + persistent hammering first).
3. Residual ideas parked: freed-SMEM K/V stage deepening, registered-config
   lottery (splitk64/qk2 on the dual-corr build).

---

# Session 13 (2026-06-10): PHASE 1 CLOSED - three measurements, prefetch line ends

M1 (defused-cvt cost, generic body, equal structure): 137.8k vs 136.0k fused
   = +1.8k cycles (+1.9M instructions, mostly hidden).
M2 (probe hit-rate, pre-pack placement): 0% (0/31 block0, 0/24 block95).
M3 (placement variants): in-pack probe after qid0 = 7% hits (2/31, 2/24) and
   153.9k cycles (in-loop plumbing without data); QK_TWO_AHEAD to advance score
   production = 239.2k (+97k - the copy-done lookahead serializes the lane,
   same pathology as the pingpong-era qk2 at 274.8k).

VERDICT (numbers-backed): the 128-130k target is unreachable via the prefetch
interleave because the dependency is data, not synchronization: under the
production issue cadence, QK(idx+1) drains the in-order tensor queue (behind
PV(idx-1)) at approximately QUANT(idx)'s END, so there is nothing to prefetch
during the pack. The 142.0k-vs-136.0k gap decomposes exactly as
+1.8k cvt + ~4.2k plumbing with 0% benefit. Prefetch gated off with the verdict
in the gate comment; the de-fused cvt helper and the ptxas-bug knowledge stay.

Kept config: the production default (unchanged). Phase-1 compliance sweep
(auto modes, fresh run): ratios 0.73/0.73/0.94/0.96/1.29 across 1K-16K
(clock-noisy wall clock; ncu ground truth 133.6k @4K). Full validation green:
default bit-deterministic at reference floor, ptmem bit-identical, persistent 5/5.

PHASE 2 (next): pv_tmem_ready as TWO alternating semaphores (commit target =
score_idx & 1; per-slot phase masks at every consumer), gated to the spare
family only (DUAL_CORR && !DIRECT_AFTER) to bound blast radius; then relax the
QUANT pace from depth-1 (entry idx-1) to depth-2 (entry idx-2), which the
doubled parity tolerance makes safe (overshoot <= 2 cannot wrap past an unseen
phase with two alternating semaphores). Phase-2 baseline to beat: spare-paced
135.6k; target: recover toward the free-running 128.7k; ship bar: beat 133.5k
across the full sweep, bit-identical O/LSE vs the spare-correct reference,
cold-trap 10/10, persistent 5/5.

---

# Session 14 (2026-06-10): PHASE 2 executed and closed - the pacing ladder is complete

Implementation (STATIC_PV_DUAL_PHASE, spare family only): pv_tmem_ready[2] with
commit target = score_idx & 1, per-slot phase masks in the dual-block
wait_pv_and_release, QUANT pace relaxed to depth-2 (publish(idx) waits the
OUTPUT loop entry for idx-2, own corr slot), seed gated to N>2 and entries to
idx+2<N for exact per-slot cross-task pairing. Builds clean, no new hangs
(cold-start traps 10/10; persistent task-seam hammering with mixed task sizes
incl. iters==1: S=4096 H=12 5/5, plus S=8192/S=1024 runs).

## SEAM BUG caught by the mandated edge-case hammering (and fixed)

First depth-2 build wedged at S=8192 persistent (768 tasks, 2.5 tasks/CTA;
S=4096 at 1.26 tasks/CTA passed): pv_dual_phase_mask was declared in per-task
scope and reset each task while the semaphore parities are cumulative - any
task leaving an odd per-slot PV count desyncs the next task. Hoisted next to
pv_phase (warpgroup scope). After the fix: cold traps 10/10; persistent seams
S=4096 5/5, S=8192 3/3 (the former wedge), S=1024 5/5. The recurring failure
class of the night (per-task vs cumulative phase state at task seams) strikes
again - and the mixed-task-size hammering mandate is what caught it.

## The measured pacing ladder (S=4096 H=12 fullgrid, ncu cycles)

  free-running (racy, no pacing)        128.7k   non-deterministic, NOT shippable
  count-paced depth-1                   135.6k
  count-paced depth-2 + dual pv (fixed) 135.9k   (== depth-1 within noise)
  production default (direct)           133.7k   (same-session reference)

VERDICT: phase 2's hypothesis is falsified by measurement. The doubled parity
tolerance does not recover the pacing tax - depth-2 measures the same as
depth-1 once seam-correct (the pace was not the binding constraint; the
OUTPUT-loop round trip is). Every CORRECT variant of the spare family loses to
the direct default by ~2.2k; the 4.8k racy-vs-default gap is smaller than the
cheapest correct synchronization found. The spare/gap-restructure line is
closed on the merits at both depths. The family's pre-existing residual race
(det ~5e-3, dLSE ~5e-4) remains and is moot. Dual-phase machinery stays in
tree (correct, family-gated, non-default). Per the ship bar, no full-sweep
promotion run is owed: there is no candidate (loses at the gating shape).

## Where this leaves the roadmap

Phase 1 (prefetch/interleave): closed - data-starved at production cadence.
Phase 2 (pv dual-phase + relaxed pace): closed - net-negative.
The per-tile period therefore stands at ~1100cy with the rescale-in-the-PV-gap
and the QUANT serial chain as its poles, and BOTH gap-restructure attempts
(spare depth-1/depth-2) plus ALL QUANT-side parallelism/overlap attempts are
measured losers on this dataflow. Production default (133.5-133.7k,
bit-deterministic, persistent-safe, 0.73-1.29x bf16 wall-clock across 1K-16K)
is the optimum found across 14 sessions and ~40 measured variants.

---

# Session 16 (2026-06-10): "optimal" RETRACTED - the BF16 comparator changes the picture

## The two numbers that should have been taken on day one

| | FP4 (shipped default) | BF16 (TK baseline, `_C_b300_causal_bf16_baseline.so`, `ext.forward(q,k,v,out,lse)`, out=(B,S,H,128) bf16, lse=(B,H,1,S) f32) |
|---|---|---|
| ncu cycles S=4096 H=12 B=1 fullgrid | 133.5k | **137.4k** |
| SM compute throughput | ~33% | **44.7%** |
| regs/thread | 168 | 128 |
| stall character | spin-dominated | **spin-dominated too** (top-3 mbarrier spin BRAs = 39% of samples; it is also a tcgen05 multi-WG pipeline, LDTM present) |

FP4 is 2.8% faster than the BF16 kernel it was meant to beat by 4x. Both run
~1100 cy/tile; both are pipeline-latency-bound; tensor pipes near idle in both.
The FP4 conversion swapped BF16's cheap in-register P handling for quant +
payload + cross-WG corr/rescale protocol (~350 cy/tile) which nearly exactly
cancels the 4x MMA saving (~50 cy/tile of tensor time). The 14-session variant
sweep only ever tuned that protocol at fixed tile shape - it proved a LOCAL
optimum of {Mb=128, Nb=128, sync variants}, nothing more. bf16 profiling
artifacts: /tmp/bf16full.ncu-rep, runner /tmp/profile_bf16.py (note: bf16
kernel symbol is plain `kernel<config<128,128,192,128,b>>`, ncu filter `-k kernel`,
needs out/lse preallocated).

## ACTIVE PLAN (approved): skip-rate build, then Nb=256-equivalent pairing

### Step 1 - rescale skip-rate measurement (one instrumented build, then revert)
Where: OUTPUT's `rescale_main_output_for_corr` (DIRECT family lambda, currently
~line 4000 region; computes `needs_rescale = __any_sync(corr < 0.999f)`).
Design: `__device__ unsigned long long fp4pv_dbg_ring[64];` in
fwd_device_helpers.inc (file scope - NOT inside the kernel-body .inc, that
fails with "automatic __device__ not allowed"); in the lambda, lane 0 of
blocks 0 and 95 increments ring[0]=calls, ring[1]=rescales-taken; print at the
OUTPUT end-of-task (or read via cuda-gdb `x/4gx &fp4pv_dbg_ring` on a live
process - symbol has no debug info, use the & form). Production config,
S=4096 fullgrid. Purpose: calibrate how much of the PV->PV gap is rescale
compute vs pure hop latency; informs the pairing design below (a pair shares
ONE rescale - if skips dominate, the win is hop-only ~350/2; if rescales
dominate, win is bigger).

### Step 2 - Nb=256-equivalent: PAIRED k-tiles, one protocol round per 256 cols
Key realization: no 256-wide score tile needed (TMEM/regs both prohibit it:
2x256 cols would blow TMEM; 128 float2 = 256 regs > q208). Keep Nb=128 score
tiles and PAIR them (tiles 2i, 2i+1) into one logical 256-col online-softmax
step, processed in two register passes:
  - pass A: wait scores(2i) -> LDTM+mask -> maxtree -> (registers freed);
            wait scores(2i+1) -> LDTM+mask -> maxtree -> pair_max
  - one running-max update + ONE acc_scale/corr for the pair
  - pass B: RELOAD tile 2i from TMEM (scores still resident in slot 2i%2 -
            do NOT arrive copy_done until after reload), exp/pack/store
            payload slot 0; reload 2i+1, exp/pack/store payload slot 1;
            sums accumulated across both -> ONE row_sum update
  - publish: ONE corr publish (corr slots/pacing per PAIR), payload publishes
            per tile as today (p_quant per slot), ONE rescale per pair on the
            OUTPUT side, then PV(2i) and PV(2i+1) issued BACK-TO-BACK with no
            rescale between (acc=0/1 then 1).
Cost/benefit: +1 extra LDTM pass per pair (~150-200cy) buys -1 protocol round
(~350cy hops + ~300-400cy rescale when taken) => projected >= 10% at S=4096.
The Nb=64 result (hop doubling => 2.38x) is the slope evidence in reverse;
per-tile fixed costs beyond hops also halve.
Numerics: max-combine exact; P payload of the first half uses the PAIR max
(same class of change as any Nb change - validate det==0 HARD, vs-ref at usual
envelope, LSE small shift legitimate); scale/amax FORMULAS byte-identical.
Protocol bookkeeping (all laws apply): scores copy_done stays PER TILE but
arrives only after the PASS-B reload of that tile; corr_arrived/rescale_finished
become per-pair (slot = pair&1 keeps the dual-corr machinery; counts per task
halve - re-derive seed/entry pairing for odd tile counts: last unpaired tile
runs as a singleton = today's path); pv waits per tile (2 commits per pair) or
per pair (single wait of the second commit - parity overshoot stays <= 1 since
both commits precede the single poll... NO: two commits before one poll = even
overshoot on a single sem - EITHER wait both commits (two polls) or commit
only the second MMA (one commit per pair, DO_COMMIT=false on the first - the
k64-chunk helpers already support exactly this via DO_COMMIT template arg!).
USE one-commit-per-pair: fp4pv_mma_p_stage_ABt_k64_chunk<...,DO_COMMIT> pattern
or plain mm/mma with the commit suppressed on the first PV (see
fp4pv_mma_p_stage_ABt_split_k64_view for the two-MMA one-commit shape).
K256-lane pointers for the port: STATIC_MXFP4_K256 (consumer modes 6/178,
EXTERNAL-only today), MXFP4_PV_MMA_PER_TILE=2, p_fp4_k256_tile =
st_fp4e2m1_2<Mb,Nb>, pair_buf machinery + wait_and_stage_v_sc_k256 in the lane,
K256 && DIRECT_AFTER currently asserted incompatible (the pairing above
sidesteps K256 entirely by staying at Nb=128 with paired protocol - likely
cheaper to build fresh under a new flag STATIC_ONLINE_MXFP4_PAIRED_KTILE than
to port K256).
EXIT CRITERION (falsifiable): >= 10% at S=4096 (<= 120k cycles) if the
hop/fixed-cost model is right; < 3% means the per-tile cost model is wrong
again and the period is something else. Bit-bar: det==0 mandatory; full 1K-16K
sweep both modes if it wins at 4K; default untouched until beaten across the
sweep; scale/amax math byte-identical.

### Backlog (after the above)
- exp/quant precision: fp32-exact EX2 feeds a 4-bit payload; ~600cy/tile
  (~19% of runtime) shared with bf16. Low priority until protocol amortized.
- bwd findings parked (from the aborted baseline): auto mode
  (TK_FA4_MXFP4_BWD_FULL_REWRITE=auto) WORKS, 0.40ms @ S=2048 H=12 = 1.6x
  SLOWER than cute-dsl bf16 bwd (0.25ms); DEFAULT mode (env unset) wedges
  (>5min at S=2048 - do not use); accuracy vs cute bf16: dq rel_l2 ~57 (!!),
  dk ~1.3, dv ~0.22 - dq structurally broken, not quant noise. Sweep script
  /tmp/bp5_sweep.py, probe /tmp/bp4.py.
- OPERATIONAL LAW (cost us a 5-min phantom wedge): check
  `nvidia-smi --query-compute-apps=pid` for zombie contexts BEFORE every
  measurement; two 3h-old gdb-orphaned trap processes (nd_trap2) sat resident
  all night - kill trap pids IMMEDIATELY after gdb detach.

## Session 16 step 1 RESULT - rescale skip rate (production config, fullgrid)

  S=1024:  block0  7/7   taken = 100%
  S=4096:  block0 26/31  taken =  84%
  S=16384: block0 56/127 = 44%; block95 32/53 = 60%

Rescales are mostly TAKEN at the shapes that matter (84-100% at <=4K), trending
~50% at 16K as the running max stabilizes. The PV->PV gap therefore contains
the ~300-400cy rescale compute most of the time, not just hop latency. Pairing
projection per pair: save ~350cy hops + ~0.84x rescale, pay ~150-200cy reload
=> net ~500-600cy per ~2200cy pair = the >=10% exit bar is well supported.
Instrumentation reverted after measurement (ring + printf + lambda counters).

## Session 16 step 2 RESULT - paired k-tiles: FALSIFIED by measurement

Built as designed (STATIC_ONLINE_MXFP4_PAIRED_KTILE, config
dualaccum_directrescale_pair_q208_p96_o48; pair-indexed dual-corr + dual-pv
phases; one corr/rescale/pv-commit per pair; two-pass quant with TMEM reload).
CORRECT on first build: det=0 at S=1024/2048/4096, dLSE<=1.9e-6, dO~1.4e-2
(legitimate pair-max payload regrouping).

Cycles @ S=4096: v1 (pair-end payload publishes) = 238.6k; v2 (per-tile
publishes restored) = 205.2k; default = 133.5k. Exit criterion (>=10% win /
<=120k) fires NEGATIVE by a wide margin.

WHY (the corrected model): the per-tile protocol round costs ~350cy ONLY as
non-overlapped residue - most of it hides under the 5-agent interleave. The
two-pass pairing lengthens QUANT's serial segment per pair (~2400cy: pass A
waits BOTH tiles' scores with no compute to hide behind; pass B reloads both
tiles; copy_done releases late so the producer lookahead compounds the lag)
and that serial growth costs ~3.5k/pair - 7x more than the protocol round it
saves. LAW (new): protocol savings only count at their non-overlapped residue;
any redesign that lengthens one agent's serial segment loses unless the
overlap structure is rebuilt around it (which is the qk2-class change that
measured catastrophic twice). Single-pass pairing is register-impossible at
Mb=128 (2x64 float2 = 256 regs > q208).

Config stays registered (correct, deterministic, non-default, marked slow).
Default untouched and re-verified after. The Nb-granularity axis is now
measured on BOTH sides: Nb=64 2.38x, paired-256-equiv 1.54x - the per-tile
shape Mb=128/Nb=128 sits at the overlap sweet spot of THIS pipeline design.
Remaining unexplored from session-16 list: exp/quant precision angle (shared
with bf16, ~19% of runtime); the memo's option-1/2 architecture shapes.

## Session 16 addendum - the THIRD comparator (cute-dsl FA4 bf16 fwd) and the venue map

cute-dsl bf16 fwd vs TK bf16 fwd vs our fp4 (wall ms, B=1 H=12, same harness):
  S=2048: 0.193 / 0.073 / 0.071  (fp4 = 0.37x cute, 0.97x tk)
  S=4096: 0.260 / 0.123 / 0.127  (0.49x / 1.03x)
  S=8192: 0.409 / 0.306 / 0.270  (0.66x / 0.88x)
=> The TK single-CTA pipeline is the FASTEST known architecture at these
shapes; the cute M256-cluster persistent design (= memo option 1's shape, per
the in-repo cute_dsl_mxfp4_forward_d192_port.py geometry: qk_mma (256,128,64),
pv_mma (256,128,128), cluster (2,1)) loses 1.3-2.6x at B=1 H=12. Cluster/M256
direction now has TWO negative data points (cta2 probe +22%, cute itself).

VENUE MAP (everything measured to date):
A. Two independent fused query-streams per CTA - the strongest revamp:
   1 warpgroup per stream does everything (softmax+quant+own PV issue+own
   rescale; same-WG ordering => protocol DELETED). 2 streams at Nb=64
   single-slot: TMEM = 2x(64 scores + 128 out) + ~64 shared scales = 448 <= 512
   FITS (Nb=64 hop penalty does not apply - no hops). Regs tight
   (~2x128x~180 + small producer). Stalls fill with sibling work; ceiling
   ~1.4-1.7x. Full consumer rewrite (~weeks) but conceptually simpler than
   today; the night's laws mostly become moot by construction.
B. Quantized-denominator softmax (math change - NEEDS ROBERT'S SIGN-OFF,
   violates the current row_sum semantics): P_fp4 = e2m1(exp2(t)) is a
   16-breakpoint step function of t - exp+cvt collapses to a compare ladder,
   and summing the QUANTIZED P makes the denominator consistent with the
   quantized numerator PV computes. Saves ~400-600cy/tile of the SHARED
   softmax cost (the 19% both kernels pay) - the only venue that attacks the
   floor itself. Cheap to build; numerics evaluation required.
C. Ship fwd + move to bwd: fwd already 0.88-1.03x vs the best bf16; bwd
   auto-mode is 1.6x SLOWER than cute bf16 bwd with structurally broken dq
   (rel_l2 ~57) - both more headroom and a correctness MUST-FIX, at 2-2.5x of
   training FLOPs.
D. Measured dead: cute-architecture adoption, cluster/M256, both granularity
   directions, protocol micro-variants, 2-CTA at Nb=128, prefetch.

Recommendation: C as the program move; A as the fwd revamp if fwd stays the
priority; B as the cheap outsized win pending the numerics decision.

---

# Session 17 (2026-06-10): VENUE B executed and closed - quantized-denominator softmax

## What shipped into the tree (registered, non-default)
`dualaccum_directrescale_qsum_q208_p96_o48` (+ `qsumptmem` sibling): payload =
production exp2+cvt path UNTOUCHED; row_sum = sum of the QUANTIZED P values,
computed from the packed payload words via nibble-compress + PRMT weight-LUT
({0,1,2,3,4,6,8,12} half-units) + dp4a, exact integer accumulation, final
0.5f*(float). det=0/20 at S=2048; dLSE = 0.088-0.11 and dO ~4-5e-2 vs base =
exactly the designed denominator-semantics shift (the denominator is now
consistent with the quantized numerator the PV computes). Flag
STATIC_ONLINE_MXFP4_QSUM_LADDER; helper fp4pv_qsum_weights_from_word.
NOTE: PRMT reads selectors from c[15:0] - byte-spread codes must be
nibble-compressed first (a wrong-selector build measured dLSE 0.80).

## Why B is CLOSED on performance
134.9k cycles vs 133.5k base at S=4096 - no win. The exp/cvt/sum serial chain
was NEVER the binding constraint (third confirmation of the overlap law: in
this latency-bound pipeline, serial-segment op counts do not convert to
cycles). Also measured and rejected on the way: per-element compare ladders -
a 7-deep if/else ladder compiles to divergent branch chains (752k cycles, 63.9M
instr); a branchless predicate-sum ladder is 5x the op count of MUFU.EX2+F2FP
(533k incl. exp2 coexistence). On sm_100a the hardware transcendental+convert
units ARE the right per-element quantizer; do not re-attempt ladders.

## The determinism forensic (paid for in full, worth keeping)
The branchy-ladder builds wobbled (det up to 1.4e-1, 16/20 runs, ALWAYS warps
2-3 of iters==1 tasks, LSE exact, wild per-element ratios). Exonerated by
experiment: SMEM payload transport (racecheck clean), p_stage slot reuse
(drain-wait added, no change - reverted to ptmem-only gate... NOTE: the gate
extension was reverted; ptmem keeps it), TMEM payload transport (qsumptmem
wobbled identically), register pressure (q224 same), pure perturbation (inert
build CLEAN), tail speed (dp4a build is faster than base yet CLEAN).
Conclusion: the wobble tracked WARP DIVERGENCE in the quant body (the nested
ladder), not timing or transport. The dp4a/branchless forms are convergent and
det-clean. Production default remains det=0 (re-verified 0/20 and full refs).

## Remaining value of B (Robert's call, zero perf cost)
The quantized-denominator semantics are arguably BETTER numerics (exact
normalization of what PV actually sums) and cost nothing (134.9 vs 133.5 =
within 1%). If a model-quality eval ever wants it: the config is registered,
deterministic, and correct. Otherwise it idles harmlessly.

NEXT: VENUE A (two independent fused query-streams per CTA) - design frozen in
the session-16 addendum; full implementation plan to follow in this file.

---

# VENUE A - frozen implementation design (build next): fused dual query-streams

One CTA = TWO independent single-WG attention pipelines ("streams") + one
producer WG. A stream owns m-tile pair (m, m+1) of the SAME (b,h) so K/V and
their scales are shared (bandwidth halves, slot release = count-2 arrives).
No softmax recurrence between streams; every cross-WG semaphore in today's
design disappears - each stream's softmax, quant, payload store, PV issue, and
accumulator rescale are same-WG program-ordered.

## Resource map (all budgets verified)
- 3 warpgroups: WG0 = stream A, WG1 = stream B, WG2 = producer (TMA only).
  Registers: consumers ~q160 (scores 32 float2 + temps; O stays in TMEM),
  producer ~80: (160+160+80)*128 = 51.2k < 64k OK.
- TMEM (<= 512 cols): scores 2 x 64 (Nb=64, SINGLE slot per stream - the
  consumer LDTMs the whole tile immediately and releases, so its own QK(i+1)
  overlaps its compute; the paired-ktile failure does not apply because there
  is no cross-agent handoff) + out 2 x 128 + q_sc 2x16 + k_sc 32 + v_sc 32 +
  p_sc 16 = 496. FITS.
- SMEM ~65KB: Q 2x12KB, K 2x6KB slots (shared), V 2x4KB (shared), payload
  2 streams x 2 slots x 4KB, scales. Roomy.

## Per-stream consumer loop (zero cross-WG hops)
  wait scores_arrived (own QK commit) -> LDTM x2 + wait::ld -> issue QK(i+1)
  (slot free; gated on k_arrived(i+1)) -> mask/maxtree -> m,corr update ->
  if corr<0.999: rescale OWN out accum (LDTM/STTM + wait::st, no semaphore) ->
  exp/pack 64 cols -> STS payload[i%2] -> fence.proxy.async (same-thread order
  to own MMA issue) -> wait v_arrived(i) -> issue PV(i) into OWN out ->
  tensor_commit p_reusable[i%2] (slot self-pacing, drain-correct) -> sums.
  Epilogue: commit+wait last PV, LDTM out, normalize by own row_sum, STG.
Sync inventory per stream: scores_arrived[1], p_reusable[2] (self), shared
k_arrived[2]/v_arrived[2]/q_arrived (producer, k/v released by count-2
arrives from both streams). Every wait is on DATA, none on another WG's
compute phase. Today's ~7-9 hops/tile collapse to 2 data waits + self-pacing.

## Win model and exit criteria (falsifiable)
Per-stream serial ~650cy/64-col tile; two independent WGs cross-hide all MMA
and memory latency; zero protocol residue. Ceiling ~1.4-1.7x. EXIT: >= 15%
at S=4096 (<= 113k cycles equivalent work) else the latency-bound model has a
deeper floor (SM dual-WG issue contention - measurable via per-WG stall
sampling). det == 0 hard; LSE shift small (Nb=64 sum grouping) documented;
dO vs default ~1e-2 class (legitimate regrouping); cold-trap 10/10;
1K-16K sweep; default untouched until beaten across the sweep.

## Causal/scheduling notes
Per-stream iters differ by 1 (m vs m+1): the longer stream runs its tail tile
solo - fine, streams never sync. Grid: ceil(tiles_m/2) x heads x batch CTAs,
fullgrid first (persistent variant only after the protocol soaks). Odd
tiles_m: last CTA runs stream A only (stream B = empty iters=0 early-exit).

---

# Session 18 (2026-06-10): VENUE A milestone 1 LANDED - fused single-stream correct

`fwd_fused_stream_kernel.inc` (`kernel_fp4pv_fused_stream<C>`, dispatch
`fusedstream_v0`, Makefile dep added - REMEMBER: new .inc files must be added
to the Makefile or `make forward` silently does nothing). One warpgroup does
everything (TMA issue, V LDG loads, softmax, quant, payload STS, own QK/PV
issue, own in-place rescale); the only mbarriers are TMA arrivals + ONE
pv_drained tensor-commit per tile whose sequential waits cover payload-slot
reuse, V-slot reuse, rescale-vs-PV ordering, and the epilogue. Extra 2 WGs of
the inherited 3WG config park at a named barrier (bar 14).

VALIDATED at Nb=128 (production-proven tile layouts): vs production default
dO = 1.2e-4 (bf16 floor), dLSE = 1.4e-6 (reference floor), det = 0/all, at
S=1024/2048/4096. The fused concept is PROVEN correct.

Debug ledger (for the Nb=64 port and future kernels):
- k_arrived parity alternates [TMA arrive=1, QK-drain commit=0] strictly:
  QK waits parity 0, slot-refill waits parity 1, both constant; k_sc (no
  commits) needs the per-slot phase mask.
- v_sc staging is a 16-col SUBTILE cp (`subtile<full_tt_fp8e8m0<16>>(0)`);
  the 128x16 static assert fires if you pass the 32-wide tensor.
- THE MXFP4 OUTPUT DESCALE LIVES IN THE EPILOGUE, NOT THE PV: O = o_reg *
  (MXFP4_ALPHA / FP4PV_MXFP4_ONLINE_P_SCALE) * inv_norm with ALPHA =
  FIXED_P_QUANT_COEFF/36 (6x6 e2m1 payload grid), P_SCALE = 2688, p_sc word
  0x8b8b8b8b = 2^12. Bare inv_norm gives O exactly 6*2^12 too large.
- Nb=64 first attempt produced garbage O with near-right LSE: k_sc global is
  64-row granular (FP4PV_KSC_DEPTH_DIVISOR=64) but the per-tile coord/tile
  types need the production nb64 lane's handling (k_sc_depth_idx=global_idx
  for TRIPLE_SCORE Nb=64); payload/MMA geometry at Nb=64 unverified. The
  Nb=64 port now has a WORKING Nb=128 fused reference to debug against.

NEXT: (a) ncu the single-stream period at S=4096 (calibrates the dual-stream
projection: dual ~ tau if overlap is perfect, 2*tau if the SM serializes);
(b) the Nb=64 layout port (TMEM for dual: 2x(64+128)+scales=448<=512);
(c) second consumer WG + count-2 K/V slot releases; cold-start + task-seam +
iters==1 hammering per the approved plan; exit >=15% at 4K (<=113k) else
per-WG stall sampling to find the floor.

## Session 18 continued - milestone 1 EXCEEDED, solo period measured

Optimizations landed on the solo stream (all bit-validated after each):
1. V loads split into LDG-prefetch at loop top (latency under the tile
   compute) + STS-commit after the drain wait. PC sampling had shown the
   inline LDG->STS dependency chains were the largest real stall (~24% of
   working-WG samples).
2. Dual score slots (TMEM 496/512): QK(i+1) issues at loop top into the other
   slot, fully overlapping tile i. Two in-flight commits would even-overshoot
   one semaphore -> scores_arrived[2], per-slot parity (i>>1)&1.
3. Reverse-m wave packing (largest causal task first, as production does).

RESULT: solo 539.3k -> 479.1k -> 458.3k, and the kernel is now BIT-IDENTICAL
to the production default (dO=0, dLSE=0, det=0 at 1K/2K/4K) - stronger than
the refs-at-floor bar.

Solo accounting at S=4096: 458k / ~42.8 tiles/SM = ~10.7k cy/tile vs ~1.9k of
real serial work (exp/pack ~600, LDTM ~600, rescale ~700 avg at the 84% take
rate, mask/max ~150, STS/fences/syncs ~400) => ~80% of solo time is exposed
latency (TMEM round-trips, MMA drains, barrier latencies) - EXACTLY the class
a sibling stream hides. Dual projection: 2x real work + residual ~= 4.5-5k per
tile-pair => 0.75-0.85x of production => the >=15% exit (<=113k) is reachable
but tight; the solo floor (~81k if all latency hidden) says the concept has
headroom beyond it.

PENDING for dual: the Nb=64 port (TMEM needs 2x(64+128)+scales=448): known
issues from the first attempt - k_sc indexing at 64-row granularity (use
production nb64's k_sc_depth_idx=global_idx convention), payload/MMA geometry
at Nb=64 unverified (first attempt read P codes one position off: ratio
6=code7-vs-code6 signature), v_sc half-word select exists in the loader
already. Then: second consumer WG (WG1) mirroring WG0 with stream-indexed
slots/semaphores, shared K via count-2 releases (K slot freed when BOTH
streams' QK drained - two commits, count-2 mbarrier), per-stream m = (2*bid,
2*bid+1) pairing of the same head, cold-start + iters==1 + seam hammering.

## Session 18 final - milestone 2 step 1 (Nb=64 solo port) IN PROGRESS, blocked point identified

Done: kernel made Nb-generic; `fusedstream_nb64` registered; v_sc prefetch
gained the production half-word select (prepared global is 128-row granular).
The old "Nb=64 garbage O" mystery RESOLVED retroactively: ratio 24623 = 6*2^12
= the then-missing epilogue descale, not payload geometry. k_sc depth
convention (64-row units, depth_idx = tile index) appears correct: tile-0
scores match nb128 exactly.

REMAINING at Nb=64 (hard blockers for dual): dLSE ~2e-2 vs nb128 (uniform
across m-blocks, present already in the first k-tile pair => k-tile-1-class
error: k_sc unit/half handling or QK N=64 scale geometry) and det != 0 at
S=1024/2048 (2.5e-2, rows s>=32 = warps 1-3, stable row set across runs).
IMPORTANT CONTEXT: the production nb64 lane itself is documented as det-broken
(det ~3-5e-2 "pre-dual-corr bitrot") - the v_sc half-word machinery and nb64
MMA scale geometry may carry that original bug; do NOT assume the cribbed
pieces are correct at Nb=64 the way they are at Nb=128.

Debug plan for next session (controlled, solo): (1) dump k_sc smem bytes for
tiles 0/1 at nb64 vs the matching halves at nb128 (the <3,256> k_sc tile is a
64-ROW UNIT - verify what the QK helper stages for unit 1); (2) same for
v_sc consumed halves; (3) det anatomy: stable wobble rows at warps 1-3 suggest
a fill/staging read by warp 0 only reaching TMEM late - check whether
p_sc/q_sc/k_sc cp staging (leader-only) vs warp 1-3 timing has an unfenced
window the nb128 timing masks. nb128 solo remains BIT-IDENTICAL and is the
debugging reference. Solo nb128 cycles: 458.3k (3 opts in); dual projection
0.75-0.85x of production once nb64 is clean.

## Session 19 (2026-06-10): nb64 debug round 1 - bug split and head-correlation nailed

Fixed and kept (correct-by-construction even though it wasn't the nb64 cause):
k_sc TMEM is now DOUBLE-BUFFERED in the fused kernel (k_sc_tm0/tm1, TMEM at
exactly 512) - two QKs are in flight, and a single staging slot lets QK(i+1)'s
scale cp overwrite what QK(i)'s running MMA reads. nb128 re-verified
bit-identical after the change. The same single-slot-vs-lookahead window
exists wherever a non-aliased k_sc_tm meets qk lookahead - check the
production families for this pattern when time permits.

nb64 is TWO bugs, now separated:
(A) DETERMINISTIC k_sc-consumption geometry bug: dLSE ~1-2e-2 at ALL S incl
    det-clean 4096. Head-correlated, the decisive signature: heads 0-3 broken,
    heads 4-11 BIT-PERFECT. Byte dumps: k_sc smem tile-0 data[0..31] is
    nonzero for h<4 and ALL-ZERO for h>=4 (in nb128 too, whose output is
    nevertheless bit-correct!) => the consumed scale location is NOT data[0]
    for h>=4 (or the zero path is harmless), and nb64 mis-consumes exactly
    when the bytes ARE there. The bug lives in the
    fp4pv_issue_qk_chunked_qsc_tmem_cluster staging geometry at N=64
    (load_k_scale_chunk extracts 16*32-byte chunks assuming a row count that
    Nb=64 violates). NEXT: read that helper + load_k_scale_chunk fully;
    determine which issue arm production ONLINE actually takes (PRESTAGE is
    EXTERNAL-only; check STATIC_ALIAS_SCALE_TMEM for the dualaccum family);
    instrument the CONSUMED k_sc TMEM (LDTM the tensor) instead of smem
    data[0]; also explain the h>=4 zero-smem-yet-correct mystery (coord d2
    units? swizzle position?) - it will likely fall out of the same read.
(B) SMALL-S RACE: det 2.5e-2/7e-3 at S=1024/2048, det=0 at S=4096; LSE
    wobbles too (scores-side). Survived the k_sc double-buffer. Hunt AFTER
    (A) lands - (A)'s fix may change the timing anyway.

Production default + nb128 fused reference both verified bit-clean after all
edits. Frozen per directive. Solo nb128 = 458.3k unchanged.

---

# Session 20 (2026-06-10): TWO PRODUCTION BUGS FOUND - the (A) plan executed to root cause

## PRODUCTION BUG 1: QK scale TMA coordinate over-multiply (ALL lanes, ALL heads but h=0)
The q_sc/k_sc gl d2 coordinate is in 3-ROW TILE UNITS (<3,256> hf tiles), but
production passes `t_coord.z * C::QK_SCALE_CHUNKS` (= h*3, an element-row
index): heads 1-3 consume heads 3/6/9's scale pages; heads >= 4 consume OOB
TMA ZEROS. Decoded the host layout by single-group perturbation probes
(k[0,s,h,g*32:..] *= 7.3, requantize, diff K_sc): page d2 = 3h + g/2 in
element rows; the scales are PER-HEAD-PER-GROUP CONSTANTS replicated along S
(consecutive depth pages are true duplicates; ~16 real bytes per 512B page).
That constancy + the narrow e4m3 alphabet is why the bug hides inside the
~3e-3 vs-ref envelope - but 11/12 heads run with wrong or zero QK fine-scales.
FIX (2 coords: q_sc_coord/k_sc_coord, `h*CHUNKS` -> `h`): validated in the
fused kernel - deterministic, ~0.13 dLSE shift = the heads' true scales
finally landing. This is a NUMERICS change (outputs move), so shipping needs a
quality eval vs refs, not bit-comparison. Production default left untouched;
the fused kernel carries the production-compatible coord with a comment block
documenting the bug for bit-parity with the frozen reference.

## PRODUCTION BUG 2 (the ancient nb64 det-breakage) ISOLATED to the N=64 descriptor
Decisive experiment pair: with ZERO scale-factor content staged (correctly
sized 1536B zero region - NOTE the first attempt's 512B buffer fed garbage to
chunks 1-2 via the +512/+1024 extraction and exploded), Nb=64 == Nb=128 to the
regrouping floor with det=0 at ALL S. With NONZERO content, N=64 consumption
is both WRONG (dLSE ~3e-2) and NONDETERMINISTIC (det up to 5e-2, all S once
all heads have content via the coord fix). => the N=64 QK MMA's scale-factor
slot/lane consumption differs from N=128's and reads racy state. This exactly
reproduces the production nb64 lane's documented "det ~3-5e-2 pre-dual-corr
bitrot" (its heads 0-3 had nonzero content through the coord bug).
REMAINING WORK for nb64: derive the correct N=64 scale-factor idesc encoding
from the ISA tables (the PV K64-chunk helpers use scale_factor_id 0/2 for the
two halves - the QK N=64 analog needs the same treatment; the current
instruction_descriptor<...,M,N=64,false,0> is wrong for SF consumption).

## State at checkpoint
fused nb128: BIT-IDENTICAL to production, frozen, verified after every edit.
fused nb64: registered, blocked on the N=64 SF descriptor (det gate).
Dual-stream: blocked behind nb64 det==0 per the milestone-2 gate.
Production default: untouched, verified. Makefile gained $(FSK_EXTRA) hook.

---

# Session 21 (2026-06-10): SF-descriptor theory FALSIFIED by micro; nb64 = k_sc depth content

## THE SF CONSUMPTION MAP (permanent fact, measured on hardware)
New micro `/tmp/sf_micro.cu` (one-hot e4m3 SF bytes through the production
smem -> tcgen05.cp -> TMEM -> mxf4nvf4.scale_vec::4X path, A=B=1.0):
  smem byte (16*j + 4*s + c) -> B-row n = j + 32*s, scale-quad byte c
  (j in [0,32) = smem row, s in [0,4) = column quad, c in [0,4) = 16-elem group)
IDENTICAL at N=64 and N=128; at N=64 the bytes for n >= 64 are simply ignored.
The N=64 MMA consumption is sane and deterministic - the
instruction_descriptor/idesc was NEVER the bug. (Also: D=16 per hot byte
confirms 4X/e4m3 semantics work exactly as encoded.)

## Confirmed facts assembled this session
- Kernel h=0 k_sc smem bytes == host page (0,0) BYTE-EXACT => the TMA lands
  pages contiguously at data[0..]; the d2 coord IS in 3-row tile units
  (h=5/d2=15 -> pages 45-47 OOB -> zeros; page (0,15) is nonzero on host).
  PRODUCTION BUG 1 (coord over-multiply) STANDS.
- K_sc depth pages are NOT constant along S (the page(0,0)==page(1,0) match
  was a first-16-byte coincidence; torch .all() says False) => nb64's
  depth=i stages DIFFERENT content than nb128's depth=t for the same K rows
  => the deterministic nb64 dLSE (~2e-2, h<4 only because h>=4 reads OOB
  zeros in both). The perturbation probe showed (s=0..200, h=0, g=0) all
  touching depth-0 bytes 0-3 in the FIRST FOUR diff entries - the diffs
  likely continue into other depth pages (the [:4] truncation hid them);
  the per-(h,g) scale has a global component + per-row residuals.
- The small-S det race remains separate and OPEN (det 2.5e-2 at S<=2048
  pre-coord-fix; all-S with the coord fix when all heads have content).

## NEXT (bounded test matrix)
nb64 k_sc depth-coord candidates against dLSE(h<4): {i (current, broken),
i/2 (128-row pages: odd tiles should stay broken - diagnostic), 2i}. Also
re-run the host perturbation probe WITHOUT [:4] truncation for s in
{0, 70, 200} to map the depth-page row structure directly - that determines
the correct coord arithmetic in one shot. Then the in-order validation plan:
zero-content control, nonzero det==0 + floor match, production nb64 lane fix.

## Session 21 final - THE COMPLETE QK SCALE PLUMBING, DECODED TO THE BYTE

Host layouts (perturbation-probed, row-isolated via *=0.01 shrink so the
per-head global amax stays fixed):
- Row s's two e4m3 scale bytes live at page offset 16*j + 4*q + c where
  (j, q) = the consumption map coords of s%128 (n = j + 32*q) - i.e. the host
  pages are PRE-SWIZZLED into the exact <32,16> consumption layout.
- Q_sc (B, S/128, 36, 512): depth page d = 128-row block d. Production's
  depth coord (m_tile) is CORRECT.
- K_sc (B, S/64, 36, 512): depth pages come in DUPLICATE PAIRS - pages 2t and
  2t+1 BOTH hold 128-row block t. Production's depth coord (k_tile_idx = t)
  therefore reads BLOCK t/2: WRONG K-scale rows for every k-tile t >= 1.
- d2 for both: 3-row TILE units; production's h*QK_SCALE_CHUNKS over-multiplies
  (wrong heads h=1..3, OOB zeros h>=4). [Bug 1, confirmed again.]

NET PRODUCTION STATE: the QK per-row fine-scales are wrong-or-zero for nearly
every (head, k-tile) combination; only the per-head GLOBAL scale (folded via
qk_scale_log2) is reliably applied. Silently absorbed because the per-row
residual alphabet is tiny (~0x70-0x7E). FIX CLUSTER (for Robert's quality
eval, NOT shipped): q_sc d2 h*3->h; k_sc d2 h*3->h AND depth t->2t. Model
quality may measurably improve - the current FP4 QK quantization is
effectively global-scale-only.

nb64 ADDITIONAL REQUIREMENT (from the N-consumption micro): at N=64 the MMA
consumes only page-rows j+32q < 64; an odd 64-tile's rows (page-rows 64-127)
can NEVER be consumed from a correctly-depth'd page directly - nb64 needs a
shifted restage for odd tiles (read the page's upper-half bytes, repack to
positions 0-63 in a scratch, cp from there) or a host-side 64-row-paged
variant. The deterministic nb64 dLSE is fully explained (wrong pair-pages +
odd-tile truncation). The small-S det wobble remains the one open item -
re-test after the scale plumbing is fixed (it may be content-dependent
consumption of the OOB/garbage regions).

Artifacts: /tmp/sf_micro.cu (+ binary) - the SF consumption-map prober;
copy into the kernel dir alongside the other micros.

---

# Session 22 (2026-06-10): parity restage built; STANDING PARADOX at close

Built: nb64 k_sc parity restage - depth = i & ~1 (pair-base page; the first
attempt's i/2 was wrong: pages pair as (2t, 2t+1) so EVEN tiles were already
correct under depth=i), odd tiles repack upper 8 bytes of each 16-group into
the consumed lower half (verified FIRING via printf, in-place safe, warp 0 +
proxy fence + WG syncs). Zero-content control (FSK_ZERO_QKSC build hook +
1536B zero region): PASSES - nb64 == nb128 at the regrouping floor
(0 / 2.9e-3 / 3.4e-3), det=0 at all S.

THE PARADOX: with natural content, dO/dLSE nb64-vs-nb128 are DIGIT-IDENTICAL
(3.857e-2 / 2.284e-2 @1024) across THREE variants that provably change the
staged k_sc bytes (pre-restage depth=i; depth=i/2; depth=i&~1 + repack), yet
the zero-control shows scale content DOES drive the diff. q_sc is
byte-identical in both kernels (same coords, M=128 consumption both). So the
nb64-vs-nb128 divergence rides on k_sc content, but is invariant to WHICH
half/page we stage... Candidates not yet eliminated: (a) the consumed SFB at
N=64 in the REAL kernel (3-chunk K=192 QK, double-buffered k_sc_tm0/1) sits
outside the positions the single-chunk micro mapped; (b) something about the
e4m3 byte SEMANTICS at N=64 making distinct bytes equivalent post-softmax in
exactly this data (implausible); (c) a stale-measurement artifact (guarded
against: source verified, binary fresh, probe fired).

NEXT PROBE (specified): marker bytes through the REAL kernel - stage a
constant distinctive e4m3 (e.g. 0x40=2.0) as k_sc in the fused kernel under a
build hook at BOTH Nb; if nb64-vs-nb128 floor-match under constant content but
diverge under varying content, the consumed POSITIONS differ between N=64/128
in the real 3-chunk path (contradicting the micro -> probe chunks 1/2);
then one-hot markers per chunk to map the real-kernel consumption directly.

State: fused nb128 BIT-IDENTICAL to production (re-verified); production
default at reference floor (re-verified); nb64 det gate CLOSED; dual-stream
blocked. Tree carries the (correct-by-derivation) depth+repack restage.

---

# Session 23 (2026-06-10): PARADOX RESOLVED - N=64 SFA truncation is the mechanism

## Solid facts (hardware-measured)
1. ZERO-CONTROL VACUITY (method lesson): zero e4m3 scales annihilate the QK
   contribution on BOTH kernels, so the zero-content control compared
   uniform-attention against uniform-attention - it validated nothing upstream
   of the scales. Controls must use distinguishable nonzero content.
2. **SFA (Q-scale) consumption at N=64 truncates exactly like SFB** even
   though M=128: one-hot bytes for m=64/96 (smem bytes 8/12) are consumed at
   N=128 but UNREAD at N=64 (sfa_micro PROBE_SFA mode). The A-scales for rows
   64-127 at N=64 are read from TMEM positions OUTSIDE the staged window (the
   +16-col neighbor probe also came back empty - exact home still unmapped).
   THIS IS THE MECHANISM for the whole paradox: q_sc never varied across the
   k_sc experiments (invariance), and the stray positions hold neighbor/stale
   TMEM (k_sc double-buffers etc.) - timing-dependent content = the small-S
   det wobble. It is also, at ISA level, the likely root of the production
   nb64 lane's ancient det-breakage (same helper, same N=64 shape).
3. M=64 MMA is UNENCODABLE for kind::mxf4nvf4 (idesc M field = M>>7 -> 0 ->
   illegal instruction): splitting the QK by M-halves is dead.
4. STTM lane-probe attempt came back anomalous: tcgen05.st targets lanes via
   the ADDRESS lane field, not the issuing warp - the probe wrote the wrong
   lanes (rows 0-31 all lit, value 64 = a full 4-byte word landing at lane 0
   region). Needs re-doing with lane-offset addresses (two-arg tensor
   allocate / addr + (lane << 16)).

## Engineering options for nb64 (in preference order)
(i) Complete the SFA atlas at N=64 with a CORRECTED STTM probe (lane-offset
    addressing), then stage Q-scales for rows 64-127 at the found positions
    (an extra cp/STTM at issue time). Bounded: one micro iteration + kernel
    staging change.
(ii) N=128 QK at nb64 (pair-granular scores): kills the dual-stream TMEM
    economy (2x(128+128)+scales > 512) unless the score slot is shared
    single-buffered between streams - park unless (i) fails.
(iii) Robert-level: ask whether the dual-stream design should pivot given the
    N=64 SF behavior (e.g., 2-stream with one N=128 QK serving both streams'
    score halves - the K-pair shape from session 16, revisited stream-wise).

State: fused nb128 BIT-IDENTICAL + production at floor (re-verified after
debug scrub). nb64 det gate CLOSED pending (i). Dual-stream blocked.
Tree carries the depth=i&~1 + odd-repack restage (correct per derivation,
insufficient alone). All probe artifacts in /tmp/sf_micro.cu (multi-mode:
PROBE_SFA, PROBE_STTM_LANES, PROBE_M) - copy refreshed to kernel dir.

---

# Session 24 (2026-06-10): N=64 SFA READ LAYOUT PINNED - the nb64 fix is now mechanical

## Hardware facts (sfa_atlas.cu, warp-indexed STTM probes - NOTE: tcgen05.st
## targets lanes by the ISSUING WARP (warp g -> lanes 32g..32g+31), not the
## address lane field; single-warp probes can never reach lanes 32+)
- N=64 SFA (A/Q-scale) READ layout: row m <- TMEM (lane m, word floor(m/32)),
  bytes 0-3 of that word = the 4 quad scales. Verified: 3 lane groups
  (warp 1/2/3 writes lit rows 32-63/64-95/96-127 exactly), word-decode
  (lanes 32+ light ONLY via word 1), stripes (even/t<16/t%4) and exact
  one-hots (t==5 -> rows 37/69/101).
- The production cp staging (load_mxnv of the 512B page) does NOT populate
  (lanes 64-127, words 2-3): rows 64-127's A-scales at N=64 read whatever
  lives there. Confirmed column-independent (sfa at col 64 vs 128 identical).
- SFB at N=64 via the cp: CORRECT (rows n<64 consume the right page bytes) -
  only the A-side needs custom staging.
- N=128's SFA read layout uses a DIFFERENT interleave (the N=128 STTM atlas
  went state-anomalous after one probe - not needed for the fix; parked).

## THE NB64 FIX (mechanical, next build)
Replace the q_sc staging at Nb=64 with a direct STTM stage per chunk c:
  warp g (g=0..3), lane t: word_bytes = q_sc_page[16*t + 4*g .. +4)
  (the host pages are already in consumption-layout: row m's 4 quad bytes at
  page offset 16*(m%32) + 4*(m/32)); STTM that 32-bit word to the chunk
  subtile's word g: addr = q_sc_tm.addr + 16*c + g (lane implied by warp).
  Then tensor_store_wait + fences, then the QK issue (k_sc staging unchanged).
This stages EXACTLY the (lane m, word m/32) layout the N=64 MMA reads, from
the same page bytes nb128 consumes => bit-parity expected, det restored
(no more unstaged-TMEM reads). Validation ladder queued: distinguishable-
nonzero controls, det==0 with timing hammering at S=1024/2048/4096, nb128
floor match, then the production nb64 lane repair check.

Artifacts: sfa_atlas.cu (warp-indexed STTM atlas) + sf_micro.cu (cp-path
probes, PROBE_SFA/SFA_COL/PROBE_M modes) both in the kernel dir.

---

# Session 25 (2026-06-10): SESSION-24 CONCLUSION CORRECTED - nb64 failure is
# kernel-environmental, NOT an ISA layout issue

## The decisive reproducer (sfa_k192.cu, in kernel dir)
Exact production shape - K=192 as 3 chunked N=64 MMAs, per-chunk cp staging of
SFA subtiles + SFB, M=128, DISTINGUISHABLE per-row scale content (2^((m%2)+c))
covering ALL 128 rows: **bad=0/128 in all four modes** (cp/N=64, cp/N=128,
STTM/N=64, STTM/N=128). D[m] = 448*2^(m%2) exact everywhere.
=> The production cp staging computes ALL 128 rows' A-scales CORRECTLY at
N=64 in a clean environment. Session 24's "N=64 SFA truncation" is RETRACTED
as an ISA property: the PROBE_SFA bytes-8/12-dead result was almost certainly
the no-sem `load_mxnv_scale_async(sfa, ...)` racing the immediately-following
MMA in that probe (the only structural difference from this passing
reproducer). The warp-indexed STTM atlas results (lane m, word m/32) remain
valid AS a layout description, but the cp populates it fully.

## Implications
- The real fused kernel's nb64 breakage (dLSE ~2e-2 + small-S det) is
  ENVIRONMENTAL: something in the kernel's state/ordering corrupts the QK
  scale path at Nb=64 only - candidates: the no-sem cp staging inside
  fp4pv_issue_qk_chunked_qsc_tmem_cluster racing its own MMA (works at nb128
  by slack, races at nb64's tighter timing - SAME class as the probe race!),
  TMEM state interactions, or the k_sc repack/depth interplay.
  STRONG NEXT CANDIDATE: per-chunk cp -> MMA with NO wait inside the helper -
  the reproducer's cp path used SEMAPHORE-WAITED cps (mode 0 waits each)!!
  Production nb128 may simply have enough slack; nb64 does not; the
  PROBE_SFA race had the same shape. FIX TO TRY FIRST NEXT SESSION: stage
  k_sc/q_sc with semaphore-completed cps (or tensor-pipe fences) before the
  MMA issue at nb64 (the preloaded-ksc helper variant + pre-waited staging).
- My kernel's STTM stage exploding (while passing in the micro) is the same
  environmental class; parked under FSK_STTM_QSC.

## State
fused nb128 BIT-IDENTICAL + production floor: re-verified after all of
today's edits. nb64: cp-default, finite-broken (dLSE 2e-2), det gate CLOSED.
Dual-stream blocked. References frozen. The repair-kit completion now hinges
on the staging-race fix, which also likely explains the production nb64
lane's breakage (same helper, same no-sem cps).

---

# Session 26 (2026-06-10): cp-race theory FALSIFIED; suspect list narrowed by
# a structural argument to K-nibbles or the recurrence

Built: completion-waited scale staging at Nb=64 (one commit+wait per tile over
all 3 chunk cps, preloaded_ksc pure-MMA issue path, k_sc TMEM widened to
2x48-col double buffers, 448/512). KEPT (it is correct-by-law hardening;
timing slack is not a protocol) but it did NOT change the numbers: dLSE
2.284e-2 @1024 digit-identical AGAIN. The unfenced-cp race theory is
falsified for the deterministic component.

## The structural re-rank (why the next probes are K-nibbles/recurrence)
Q-scales (SFA) multiply score ROWS: any per-row factor CANCELS in softmax -
O is invariant, only LSE shifts by log(s_m). Therefore nb64's dO=3.857e-2
CANNOT originate in the SFA/q_sc path (every SFA theory chased this week was
structurally incapable of explaining dO). K-scales (SFB, per-column) do
reweight softmax - but every k_sc content variant (depth x3, repack, fenced)
left dO/dLSE digit-identical. Remaining suspects, never cleared:
  (1) K-NIBBLES: the <64,96> K-tile TMA. Verified identical only at row 0
      (raw data[0] dump); deeper rows unverifiable by raw offsets (swizzle).
      NEXT: swizzle-aware dump via the tile's idx() address math - compare
      nb64 tile-1 rows 0-63 against nb128 tile-0 rows 64-127 row-by-row.
  (2) THE 64-GRANULAR RECURRENCE: never validated with nontrivial maxes (the
      zero-control degenerated it). NEXT: python-side regroup simulation of
      expected dLSE from 64- vs 128-tile online softmax on REAL scores
      (expected ~1e-6 if pure regroup; the measured 2e-2 implies wrong inputs
      OR a recurrence bug - the simulation discriminates).

## Standing contradiction to resolve (one reproducer run)
Production + nb128 match torch at floor while (per the coord-bug decode)
consuming ZERO k_sc bytes for heads >= 4 - yet the k192 reproducer shows
exact multiplicative e4m3 scale behavior. Either zero SF bytes are special
(treated as 1.0? - test: SFB=0x00 everywhere + nonzero A/B in sfa_k192) or
the coord-bug decode has a flaw. This also gates the parked quality-fix
cluster's narrative - resolve before Robert acts on the banner.

State: references verified bit-clean after the fenced-staging build (nb128
bit-identical; production floor). nb64 det gate CLOSED. Dual blocked.

---

# Session 27 (2026-06-10): ROOT CAUSE OF nb64 FOUND - the K-tile TMA loads
# wrong nibbles; banner retracted; scale theories closed

Discriminator ladder results (in the ordered sequence):
1. ZERO-SF SEMANTICS (sfa_k192 mode 2): zero e4m3 SF bytes are TRUE
   MULTIPLICATIVE ZERO (D == 0 exactly, N=64 and N=128). NOT special-cased.
   => production matching torch at floor while the staged h>=4 k_sc tile is
   verified ALL-ZERO across its full extent is inconsistent with the
   consumption model => THE BANNER WAS RETRACTED (rewritten in place with
   equal prominence; only byte-verified facts retained; the quality-fix
   cluster is ON HOLD until a real-kernel marker experiment resolves what
   the QK MMA's SFB actually reads).
2. RECURRENCE SIMULATION: 64- vs 128-grouped online softmax on real-scale
   scores, kernel-faithful fp32 sums: worst diff 4.2e-7 vs measured 2.0e-2
   => recurrence exonerated, INPUTS implicated.
3. SWIZZLE-AWARE K DUMP (G::k_tile::idx() addressing): logical K row 64:
   nb128 tile-0 row 64 = 76da6658..., nb64 tile-1 row 0 = 6cccf4b3...
   => DIFFERENT BYTES FOR THE SAME LOGICAL K ROW. **The <64,96> K-tile TMA
   coordinate/descriptor mapping is wrong at Nb=64** - the week of scale
   archaeology was chasing a K-DATA bug. (Also explains why every scale
   variant was invariant and why h>=4 - whose k_sc is zero-loaded the same
   way in both kernels - still differed... wait, h>=4 was bit-perfect; the
   head-dependence remains to be re-explained once the K coord is fixed -
   possibly the wrong-row K data coincides for some heads. Verify after fix.)
NEXT (one run): dump nb128 logical row 32 - if it equals 6cccf4b3 the d2
coord unit is 32 rows and the fix is coord i -> 2i; otherwise scan the
candidate mappings. Then the full validation ladder as ordered.

State: references verified bit-clean after scrub (nb128 bit-identical,
production floor). det gate closed pending the K coord fix. Fenced staging
kept as hardening.

---

# Session 28 (2026-06-10): K-tile TMA coord FIXED (32-row d2 units confirmed);
# residual converges with the banner question onto ONE experiment

CONFIRMED: nb128 logical row 32 == the bytes nb64's coord-1 tile reads as its
row 0 => the <64,96> K-tile TMA d2 coordinate is in 32-ROW units. FIX APPLIED:
k_coord d2 = 2i at Nb=64 (prologue + refill). Effect: dLSE moved (2.284 ->
2.360e-2 @1024) = the fix took, K data now correct (pre-fix, tile i loaded
rows [32i, 32i+64) - HALF-OVERLAPPING WRONG WINDOWS, explaining the bulk
error class). KEY METHOD NOTE: kittens TMA tile coords are in TMA-BOX units,
not tile units - box height for fp4 <64,96> tiles is 32 rows; <128,96>
apparently boxes differently (production t works there). Audit any future
non-128-row fp4 tile TMA the same way.

RESIDUAL after the fix: dLSE ~1.6e-2 / dO ~3.9e-2, STILL heads 0-3 only
(heads 4-11 bit-perfect = zero-SFB-both-sides trivial agreement). With
K-nibbles fixed and the recurrence exonerated (4.2e-7), the residual can only
live in the consumed nonzero q_sc/k_sc content at nb64 vs nb128 - i.e., the
SAME unresolved consumption model the retracted banner is on hold for.
=> ONE EXPERIMENT RESOLVES BOTH (queued first for next session): the
real-kernel SF marker experiment - stage distinguishable marker bytes as
q_sc/k_sc through the REAL kernel at both Nb (build hook), read the effect on
scores/LSE per (row, head), and derive what the QK MMA actually consumes in
situ. Design note: markers must be per-position-unique (the sfa_k192 micro
proved exact multiplicative readout works); softmax-invariance splits SFA
(LSE-only) from SFB (O+LSE) signatures for free.

Also parked next to the banner per directive: the NVFP4-vs-MX design question
on the QK path (e4m3 16-elem vs ue8m0 32-elem; PV is true MX) - Robert's
call, no semantics changes without it.

State: nb128 BIT-IDENTICAL + production floor re-verified after the K fix
(frozen references intact - the fix is Nb==64-gated). det gate still closed
(residual stands). Dual-stream blocked. Fenced staging + K coord fix +
depth/repack all kept (each correct-by-derivation; the stack now has exactly
one unknown left).

---

# Session 29 (2026-06-10): marker experiment CONCLUSIVE on consumption;
# K-tile box mapping still wrong - the last decode, procedure pinned

## Marker experiment results (the banner-resolving experiment, run 6 ways)
Synthetic q_sc/k_sc content injected identically into both kernels at every
staging site (build hook FSK_SF_MARKER, modes: K-markers/Q-markers x periodic,
aperiodic-hashed, natural-magnitude 0x70-0x7E): **nb64 == nb128 at the 1e-3
floor in ALL six runs** (bf16-output rounding scale). CONCLUSIONS:
- SF CONSUMPTION IS EQUIVALENT between N=64 and N=128 in the REAL kernel for
  arbitrary content on either side. The consumption-model crack is CLOSED:
  staged tensors ARE what the MMA reads, identically at both N.
- The retracted banner can be partially un-held: the h>=4 zero-loads and
  wrong-head d2 pages are still REAL (and now provably CONSUMED, since
  consumption==staged) - BUT production correctness at torch floor with
  multiplicative-zero SFB remains unexplained => one residual contradiction:
  re-verify the h>=4 zero-load claim itself (the d2 box-unit question below
  may resolve it: if d2 coords are box-units like the K tile, h*3 may land
  in-bounds pages after all and the zero dump was the K-CLASS box confusion).
- The nb64 odd-tile REPACK was removed (built on the retracted truncation
  theory; markers proved consumption needs no repack).

## The remaining bug: the <64,96> K-tile TMA box mapping (non-trivial)
Coord 1 lands row-32 data at tile(0,0); coord 2 lands ac979c9d which is NOT
row 64 (76da6658) => the unit is not uniformly 32 rows; the box ORDER is
nontrivial (likely 32rx48B boxes in an interleaved order). DECODE PROCEDURE
(mechanical, ~2 dump builds): dump nb128 logical rows {16,32,48,64,80,96,112}
col 0 AND row 0 cols {48} via idx(); locate ac979c9d and 6cccf4b3 => derive
coord->(Δrow,Δcol) box function => set nb64 k_coord accordingly (or
switch the K tile to <128,96> loads shared per tile-pair - loads each 128-row
K block ONCE, both 64-tiles read halves via the descriptor offset - AVOIDS
the box question entirely and halves K TMA traffic; the st_descriptor
chunk/offset machinery supports row offsets via subtile views).
RECOMMENDED: the <128,96>-shared-load variant - one TMA per pair into a
128-row slot, QK(i) uses a 64-row subtile view of it. No box decode needed.

## State
references frozen+verified (nb128 bit-identical, production floor). det gate
closed pending the K mapping fix. All scale-path theories CLOSED by the
marker experiment - the K-tile data is the last unknown, with two concrete
fix paths written above.

---

# Session 30 (2026-06-10): THE KEYSTONE MISREAD FOUND - fp4 tile types count
# PAIRED rows; every N/coordinate conclusion needs re-derivation

THE DISCOVERY (from the gl TMA-types instantiation chain + k_tile def):
  `k_tile = st_fp4e2m1_2<C::Nb/2, C::Dqk/2>` - the K tile at Nb=128 is
  <64,96>, at Nb=64 it is <32,96>. The fp4e2m1_2 tile "rows" are PAIRED
  (2 logical rows per tile row, matching the _2 packing). Consequences:
  - The "32-row TMA box-unit" discovery (session 28) was just the tile being
    32 (paired) rows; the session-28 coord change i->2i OVERSHOOTS under pair
    semantics and was REVERTED. The kernel is back to coord i with the
    <Nb/2,*> tile - the production convention (k_sc depth = i/2 kept).
  - The helpers' `N = KTile::rows * CLUSTER` therefore encodes PAIRED N: at
    Nb=128, "N=64"-in-pairs = 128 logical. ALL the N=64-vs-N=128 micro
    framing (sfa_atlas, sf_micro, sfa_k192) used st<N, K/2> tiles whose
    LOGICAL row count is 2N - the micros are internally consistent (their
    host models matched) but their "N" labels mean paired-N; re-interpret
    before reusing any of their layout conclusions.
  - The shared-K fix path (b) as coded (k_ld_t <128,96>) was registering the
    WRONG alt type via a bad conditional and is reverted; if revisited, the
    128-LOGICAL-row shared tile = st_fp4e2m1_2<64, Dqk/2> = exactly the
    Nb=128 k_tile - i.e., fix path (b) is "use the Nb=128 tile type +
    coords at nb64, subtile the descriptor per half" and the gl ALREADY has
    that descriptor at... the nb64 config registers <32,96> only; the
    k_gl was extended with k_tile_alt (conditional currently WRONG: produces
    <64,96> only when rows==64; fix the conditional to rows==32 -> <64,96>).
  - The nb64 residual (dLSE 2.284e-2, restored numbers) remains, now with a
    CLEAN hypothesis: at nb64 the QK consumes <32,96>-PAIRED tiles = 64
    logical K rows per issue with coord i in 32-pair units = rows 64i ✓ -
    so K data may have been RIGHT all along (the session-28 "row 32" match
    was correct PAIR-semantics: coord 1 = pair-rows 32 = logical rows 64!!
    -> 6cccf4b3 = logical row 64?? but nb128's idx({32,0}) dump found it at
    PAIRED row 32 = logical 64 ✓✓ CONSISTENT - the K data was CORRECT, the
    "fix" broke it, now reverted) => the residual is NOT K data either.
    Next discriminator: the V path (the only never-cleared per-O component:
    my custom v_sc half-select + v LDG col_base at nb64) for dO, and for
    dLSE... re-run the per-row dLSE structure first with FRESH eyes on pair
    semantics throughout (the mask/col arithmetic in MY kernel uses LOGICAL
    cols - verify the scores LDTM col mapping at Nb=64 - tt_score <128,64>
    LOGICAL cols vs the MMA writing PAIRED-N=32... the D tensor may hold 64
    cols correctly; CHECK the scores LDTM chunk count: my SCORE_CHUNKS =
    Nb/32 = 2 at nb64 - if the MMA writes N-paired=... D cols = LOGICAL N
    (f32 accumulator, unpaired) = 64 ✓ probably fine).

State: kernel restored to production-convention K (verified: nb64 numbers
back to the canonical 2.284e-2 baseline; nb128 BIT-IDENTICAL; production at
floor). det gate closed. The k_gl alt-descriptor extension remains in
fwd_bf16_baseline.inc (harmless; fix its conditional if pursuing path b).

---

# Session 31 (2026-06-10): V-path/k_sc-parity cleared; THE IMPOSSIBLE ROW
# isolated - next probe is a direct single-row scores forensic

Step (1) V path: structurally cleared for dLSE (V cannot move LSE; the
residual HAS a dLSE component => the primary bug is scores-side; V remains a
candidate only for part of dO, re-check after the scores bug falls).
k_sc later-tile parity: VERIFIED EQUAL by dumps (m=1: nb128 tile-1 page ==
nb64 tiles-2,3 pages, byte-identical; the depth-pair duplication makes
nb128's page=t and nb64's page=i/2 conventions IDENTICAL for all tiles).
NOTE: production nb128 k_sc depth=t lands the pair-duplicate of block t/2 -
the wrong-block observation stands for PRODUCTION (absorbed by the dominant
per-(h,g) global component) but is PARITY-EQUAL between the two fused
kernels, so it cannot cause the residual.

THE IMPOSSIBLE ROW: m=0 rows 0-63 (e.g. row 45, dLSE 1.4e-2 at S=4096) see
ONLY tile 0, where every input is verified byte-equal between kernels
(Q tile, K tile/page semantics, q_sc page, k_sc page, staging covered by the
marker experiment at floor) - yet LSE differs. Under markers the same row
floor-matches => the difference rides NATURAL q_sc/k_sc CONTENT through a
mechanism that synthetic content (including natural-magnitude 0x70-0x7E
hashes) does not trigger. Remaining hypotheses: (a) the natural pages contain
specific byte values (e.g. e4m3 NaN 0x7F? denormals?) consumed differently by
the paired-N=32 vs paired-N=64 descriptors; (b) a marker-experiment blind
spot (the stomp covered prologue q_sc + prologue k_sc slot0 partial + loop
k_sc; nb64 PROLOGUE k_sc slot1 + the loop's FIRST iteration interplay...).
NEXT PROBE (definitive, 1 build): dump scores_reg[0..22] for row 45, tile 0,
h=0 from BOTH kernels post-LDTM pre-mask and diff element-wise - the
differing score columns identify the consumed-input difference directly;
with equal scores the bug moves to max/exp/sum (re-instrument those).

State: references verified bit-clean after scrub (nb128 bit-identical,
production floor). det gate closed. Dual blocked. All tree changes today:
none net (dumps scrubbed); k_gl alt-descriptor extension remains (inert).

---

# Session 32 (2026-06-10): row-45 forensic - nb64 scores half-filled (cols
# 32+ zero with full K data present); readout reliability caveat; STG probe
# designed

ROW-45 RESULTS (printf readout, reliability caveat below):
- nb64 tile-0: scores cols 0-31 IDENTICAL to nb128; cols 32+ = 0.0 while the
  K smem holds full correct data (pair-rows 16/31 byte-equal to nb128). The
  zero boundary at col 32 = KTile::rows = the <32,96> tile's pair-row count.
  This SHAPE matches the measured residual exactly (row 45 loses cols 32-45
  -> dLSE 1.4e-2; rows <32 unaffected... CHECK: per-row probe had row 96 and
  row 129 too - consistent with per-tile upper-half loss).
- nb128 row-100 cols 64-99 ALSO read zero - which CONTRADICTS nb128's
  bit-identical-to-production output (those rows consume those cols) =>
  PRINTF READOUT OF HIGH-INDEX scores_reg IS UNRELIABLE (vararg marshalling
  perturbation, F2FP-class). The nb64 zeros are corroborated by the output
  numerics; the nb128 zeros are contradicted by them. Neither printf
  observation is trustworthy alone.

WORKING HYPOTHESIS (one readout from proof): the QK MMA's D-column fill =
desc-N = KTile::rows LITERALLY (32 cols at nb64) while nb128's 64 fills its
full 128 via... unresolved - OR st_descriptor<st_fp4e2m1_2<32,96>> walks B
wrongly (matches the production nb64 lane's historical breakage - same tile
type). DEFINITIVE READOUT (next, 1 build): STG scores_reg to the O output
buffer (probe-only run, e.g. rows of o[b, s, h, 0..63] overwritten with
scores for the probe row) and read from python - no printf, no vararg
marshalling. Then: if nb64 truly fills 32 cols, the fix options are
(i) two QK issues per 64-tile (second with a k-subtile/coord into D cols
32-63 - check subtile descriptor validity for 16-pair-row views first!) or
(ii) pair-granular QK (one <64,96> issue filling scores for BOTH tiles of a
pair into a 128-col buffer - kills the dual-stream TMEM budget as designed;
needs the TMEM design rework => Robert/design decision per the standing
routing).

State: dumps scrubbed; nb128 BIT-IDENTICAL + production floor re-verified.
det gate closed. Dual blocked. The banner d2 re-verification remains queued
behind this verdict.

---

# Session 33 (2026-06-10): production EXONERATED by attention-mass test;
# the STG probe itself is now the suspect; exact next steps

THE DECISIVE CONTROL (python, no kernel code): row 100 h=0 S=1024 upper-col
(64..100) attention mass = 0.366; torch-full vs torch-upper-removed O differ
by 0.077. Production O sits 0.011 from torch-full and 0.074 from torch-half
=> PRODUCTION CONSUMES THE FULL ATTENTION. The half-of-K-ignored hypothesis
is FALSE; never record it. (Also: 0.011 = production's ordinary quant
envelope at this row.)

CONSEQUENCE: the STG probe's "nb128 upper score cols are zero" is an
ARTIFACT. The probe build (FSK_SCORE_STG) was never bit-validated - it can
perturb the kernel (registers/scheduling) or its O-buffer writes may corrupt
adjacent state. The nb64 "real-but-wrong upper cols" observation carries the
same caveat BUT aligns with nb64's independently-measured residual (cols 0-31
exact-match nb128 is at least probe-credible: matching garbage twice is
unlikely).

NEXT STEPS (in order):
1. Validate the STG probe build itself: run FSK_SCORE_STG and compare all
   NON-probe rows/heads vs production (bit-identity expected). If clean, the
   nb128 zeros are real and we have a hard paradox (impossible per the
   attention-mass test) => the probe write path corrupts the probe rows only
   (g.o writes racing the epilogue?) - move the readout to a dedicated
   debug buffer (malloc'd, passed via an env-gated global pointer like
   fp4pv_dbg_ring but sized 128x128 floats).
2. With a TRUSTED readout, re-answer the single question: what do nb64's
   score cols 32-63 contain vs nb128's same logical cols (tile-0, row 45)?
   - If real-but-wrong: the B-operand's second half at nb64 comes from the
     wrong K pairs (16-pair subtile descriptor validity - fork (i): two
     QK issues per tile, autonomous; record if it explains the production
     nb64 lane's history).
   - If equal: the divergence moves to max/exp/sum for row 45 (re-instrument
     those with the trusted readout).
3. Fork (ii) (pair-granular QK) remains scope-on-paper-only per routing:
   TMEM (2 streams x (128 score + 128 out) + scales > 512), perf (one QK per
   pair = fewer issues), design implications - present, do not build.
4. Banner d2 re-verification stays queued behind the verdict.

State: probe hooks remain in-tree under FSK_SCORE_STG (build-gated, default
off; normal build re-verified bit-clean after this session). nb128
BIT-IDENTICAL + production floor verified. det gate closed. Dual blocked.

---

# Session 34 (2026-06-10): instrument-perturbation MECHANISM CORNERED -
# the probe modifies the probed block; reconciliation complete

FACTS (each verified):
(a) Attention-mass test: production h=0 row-100 O contains the upper-column
    mass (0.011 from torch-full vs 0.074 from torch-half).
(b) CLEAN-build fused nb128 == production at dO=0.000e+00 exactly,
    including row-100/h=0 => the clean kernel's scores are FULL.
(c) PROBE-build (FSK_SCORE_STG): trusted memory-path readout shows upper
    score cols ZERO at the probed block (nb128 cols 64-127; nb64 cols 32-63).
(d) PROBE-build excluding the probed block == production at 1.2e-4 floor.
RECONCILIATION: the probe CODE perturbs the probed block only - touching
scores_reg[32..63] in the probe (both printf and STG generations) causes the
upper LDTM chunks/QK results to vanish AT THAT BLOCK. (a)+(b) prove the
clean kernel is fine; (c)+(d) localize the corruption to the instrumented
block. The nb64 zeros from the same instrument inherit the same caveat -
NO nb64 conclusion can be drawn from any scores_reg-touching probe.

MECHANISM HYPOTHESIS (testable, F2FP perturbation class 3rd sighting): the
probe's extra liveness on scores_reg[32..63] pushes the probed block into a
ptxas spill/reorder that drops/defers the upper chunks. CONFIRM (1 build):
re-enable the probe WITH the epilogue store - the probed block's O should
then be WRONG vs production (perturbation visible in output).

NEXT INSTRUMENT (perturbation-proof by design): dump online_fixed_p_words
(the PAYLOAD words) from INSIDE the pack loop - those values are ALREADY
live there (no new register liveness), 4 STGs per qid, and they measure the
QUANT-CONSUMED view directly: payload e2m1 codes distinguish zero-vs-real
upper columns, which is exactly what the nb64 residual question needs.

ROUTING STATE: fork verdict pending the payload-word measurement. Fork (i)
autonomous / fork (ii) paper-only / banner-d2 queued, per standing routing.
References frozen (clean build re-verified bit-identical this session);
det gate closed; dual blocked.

## INSTRUMENTATION LAW (explicit, by directive - violations cost sessions 32-34)
NEVER probe registers whose liveness the F2FP/ptxas perturbation class can
touch: adding reads of large live register arrays (scores_reg etc.) into
printf varargs OR even plain STGs changes allocation/scheduling and can
corrupt the instrumented block while leaving the rest bit-clean. Read values
ONLY (a) where they are already live in existing dataflow (e.g., payload
words inside the pack loop), or (b) after they land in memory via the
kernel's own stores. Always bit-validate the instrumented build's
UN-probed regions before trusting any probe; treat the probed region's
output divergence as the perturbation check.

---

# Session 35 (2026-06-10): FORK VERDICT - nb64 upper columns REAL-BUT-WRONG
# (payload-word instrument); fix (i) routed; descriptor check is first action

## Perturbation mechanism FULLY pinned (inverts session 34's hypothesis)
Probe + epilogue-enabled: probed block O == production EXACTLY (0.000e+00)
=> the perturbation never corrupted the block - it corrupted THE PROBE'S
VIEW ONLY (ptxas hands the probe stale/zero register copies while the real
dataflow uses live ones). All register probes were lying; both kernels'
upper score columns were always real at consumption. The INSTRUMENTATION
LAW above stands, with this sharper mechanism note.

## The trusted payload-word readout (values already live in the pack loop)
Row 45 (causal window ends col 45 < 64 => tile-0 running max == full max =>
payloads MUST match if scores match):
  q0 (cols 0-31):  nb64 == nb128 BYTE-IDENTICAL => max identical, lower
                   half perfect.
  q1 (cols 32-45): CODES DIFFER (e.g. 65555655/00565655 vs 55555555/00555555)
                   => **nb64's tile-0 scores for columns 32-63 are REAL BUT
                   WRONG** - the QK's upper 32 output columns compute from
                   wrong B data. (Row-100 q0 differences are the legitimate
                   online-max regrouping, not a bug.)

## FORK (i) ROUTED - autonomous, with the planned validity check first
Mechanism candidate: the <32,96> (32-pair-row) K tile's B-walk pairing
differs from the <64,96> tile's (whose geometry is production-proven). The
desc-N=32 produced 64 D columns (real values both halves) => desc-N counts
pairs; the upper-half wrongness = pair-order/LBO mismatch at 32-row tiles -
ALSO the likely class behind the production nb64 lane's historical breakage
(same tile type; RECORD when confirmed).
FIX SHAPE: load K per 128-logical-row block as the production-proven <64,96>
tile (k_gl alt descriptor: fix the conditional to rows==32 -> <64,96>);
issue each 64-tile's QK against a 32-pair-row SUBTILE VIEW of the parent
(QK(2t): view at pair-row 0; QK(2t+1): view at pair-row 32... pairing order
TBD by the check). FIRST ACTION (the planned check): does
st_descriptor<view> of a row-offset subtile inherit the PARENT'S
layout/strides (then the fix works) or rebuild from the view's own 32-row
layout (then same problem - need descriptor-with-offset construction)?
Verify in a micro against known data before kernel surgery, per the
instrumentation law's spirit: measure first.

State: probe hooks build-gated off; clean build re-verified BIT-IDENTICAL
(nb128) + production floor. det gate closed pending fix (i). Dual blocked.
Banner d2 re-verification queued behind fix (i)'s landing.

## Session 35 addendum: THE DESCRIPTOR-MICRO DESIGN (mechanical resume spec)
File /tmp/desc_micro.cu (crib sfa_k192.cu harness: A=all-1.0 fp4, all scales
e4m3 1.0 via the cp path, M=128, single K=64 chunk is enough).
B DATA = per-LOGICAL-row marker: logical row n gets all elements = e2m1 value
cycle[n % 7], cycle = {0.5,1,1.5,2,3,4,6}. Then D[m][n] = 64 * b(n) decodes
WHICH logical row fed output column n (mod-7 identity).
FOUR MMA VARIANTS, same staged B data in a <64,96>-shaped smem region
(128 logical rows = K block):
 (1) REF: full <64,96> tile (production type) -> expect D cols n=0..127
     with D/64 = cycle[n%7] (calibrates the instrument + pair order).
 (2) VIEW-LOW: tile.subtile<32, 48... ACTUALLY <32, Dqk/2=96-col... K=64
     chunk => cols stay 32 bytes>: subtile<32,32>?? - K=64: b_st=<rows,32B>;
     REF tile = <64,32> (128 logical rows x 64 K-elems packed); views =
     subtile<32,32> at {0,0} and {32,0}. Issue with the VIEW as B -> which
     logical rows appear at D cols 0..63?
 (3) VIEW-HIGH: subtile at {32,0} -> logical rows 64-127 expected if
     strides inherit from parent.
 (4) STANDALONE <32,32> tile TMA/filled with logical rows 0-63 directly
     (the current broken nb64 shape) -> diff vs (2): if different, the
     32-row standalone descriptor is the bug; whichever matches ground
     truth is the valid construction for the kernel fix.
DECODE: print D[0][0..63]/64 per variant; map to cycle values; the permutation
(if any) IS the pair-order/stride answer. THEN the kernel fix: K per
128-block as <64,32*3-chunk> tiles + per-64-tile QK with whichever B
construction variant (2)-(4) proved correct.
Validation ladder after fix (unchanged): nonzero controls, det==0 + timing
hammering S=1024/2048/4096, nb128 floor match, production nb64 lane check
(record historical-breakage explanation if the 32-row descriptor confirms).

---

# Session 36 (2026-06-10): descriptor micro hangs even on the reference
# variant - patch bisect queued; SAFETY LAW AMENDED

STATUS: both descriptor-micro attempts hang (exit 124) on the REFERENCE
variant (full <64,K/2> tile, production-proven shape):
1. /tmp/desc_micro.cu (fresh rewrite): v1 hung; the user killed the original
   combined run after 15min - hangs are data.
2. /tmp/desc2.cu (PATCHED from the proven sfa_k192.cu): mode 0 ALSO hangs
   => MY PATCHES introduced it, not the variant geometry. BISECT ORDER:
   (a) the D readback restructure (revert to the proven out[tid]=v[0] form
   first); (b) K=192->64 / CHUNKS=1 staging-wait phases; (c) idesc N=64/32
   vs d_tt<128> mismatch; (d) subtile-view code compiled into all modes.
Per-variant guarded runner (timeout -k 2 25 per mode + pkill known-mine)
works - keep using it.

SAFETY LAW AMENDED: another user's training job is RESIDENT on this machine
(train.py HSA, "gpu2" run). NEVER blanket-kill compute-apps; match only
known-mine names (fsdbg, desc, sf_micro, sfa_, profile_fwd, nd_trap, _val).

GOAL STACK (unchanged): micro verdict -> K-per-128-block fix -> validation
ladder (nonzero controls, det==0 + timing hammering S=1024/2048/4096, nb128
floor match, production nb64 lane check + record historical explanation) ->
dual-stream gate. References frozen; default untouched; banner-d2 queued.

---

# Session 37 (2026-06-10): NB64 FIXED - det gate OPEN, ladder PASSED,
# production nb64 lane breakage EXPLAINED, dual-stream gate OPEN

## The geometry truth (descriptor micro, bisected + decoded)
- The micro hang was MY readback bug: tcgen05.ld is WARP-COLLECTIVE; issuing
  it under `if (threadIdx.x == 0)` wedges the warp. (Law: collective ops
  never inside thread-predicated blocks.)
- Micro decode (cycle markers): a THREAD-FILLED <64, K/2> fp4 tile behaves
  as 64 B-rows where each byte's two nibbles are two K-ELEMENTS of the same
  row; D col n = that row's sum. Row-offset subtile views (int2{32,0}) are
  BROKEN for MMA descriptors (all-zero D) - never use them.
- THE REAL KERNEL'S TMA-loaded <64, Dqk/2> K tile carries 128 LOGICAL K ROWS
  (nibble-packed pairs; the gl/TMA layout does the packing) and ONE
  N=64-idesc issue fills 128 score columns - proven empirically: nb128's
  single issue per 128-col tile is bit-identical to production with real
  payload codes through col 100 at m=0 (one iter). The earlier "pair
  semantics" flip-flops all reduce to this: TILE TYPES COUNT PACKED PAIRS
  ALONG K FOR THREAD-FILLED DATA, BUT THE TMA K LAYOUT PACKS S-PAIRS - the
  tile's meaning depends on the fill path. (Micro-vs-kernel N residual
  question noted; the kernel behavior is byte-validated, which is what
  matters.)

## The nb64 fix that landed (pair-granular QK)
At Nb=64: K loads as the 64-row tile type (k_tile_alt = <64, Dqk/2>, gl
extended + alias re-exported through globals_fp4pv_mxfp4_dv), coords in
PAIR units (one TMA per 128-K-row block); ONE QK issue per tile-PAIR at
even iterations i>0 (pair i/2), writing score0||score1 (128 cols, the two
adjacent 64-col slots) in one MMA - byte-equal to nb128's computation per
pair by construction; commits arm BOTH scores_arrived; refills on ODD
iterations (same-slot TMA-vs-issue clobber impossible); consumer/mask/pack/
recurrence stay 64-granular untouched. ZERO TMEM CHANGE. No lookahead
(QK latency exposed per pair) - perf pass deferred until after dual-stream.
Deadlock lesson: the pair issue must come AFTER both previous-pair LDTMs
(first attempt fired pre-wait at odd boundaries and clobbered unconsumed
scores).

## Ladder results (ALL PASSED)
1. Nonzero controls: payload-word instrument byte-equality (row 45 q0+q1,
   row 100 all words) nb64 == nb128 at S=1024.
2. det==0 WITH timing hammering: 14 interleaved runs (production + nb128
   between nb64 runs to perturb SM/TMEM residency) at S=1024/2048/4096:
   dO=dLSE=0.000e+00 at ALL sizes.
3. nb128 floor match: bit-identical to production (dO=dLSE=det=0) at all
   three S; production at canonical refs (2.441e-4 / 2.930e-3).
4. nb64-vs-nb128 envelope: S=1024 at FLOOR (dO 1.2e-4, dLSE 9.5e-7 - single
   pair == nb128 computation exactly); S=2048/4096 at 4-5e-3 dO / ~1.4e-3
   dLSE = the legitimate 64-granular running-max requantization schedule.

## PRODUCTION nb64 LANE: HISTORICAL BREAKAGE EXPLAINED (record for Robert)
config dualaccum_directrescale_localmax_split2wg_mb128_nb64_q152_p104_o48:
run-to-run det dO=4.395e-2 dLSE=2.211e-2 (NON-deterministic) and
vs-prod-nb128 dO=3.662e-2 dLSE=1.931e-2 - DIGIT-IDENTICAL to the fused
kernel's pre-fix signature. Mechanism (now proven): its k_tile <32, Dqk/2>
yields idesc N=32 -> each QK fills only 32 of 64 score columns; the upper
half reads stale TMEM (run-dependent -> the historical det breakage; heads
4-11 masked by the zero-scale-page accident). REPAIR: the analogous
pair-granular QK port into the production kernel's k_idx/slot machinery -
SCOPED, NOT LANDED (production frozen per directive; the lane is non-default
and known-broken, the fix recipe is this session's fused-kernel change).

## State
det gate OPEN. DUAL-STREAM GATE OPEN - next: mirrored WG1, stream-indexed
slots, shared K/V count-2 releases, IMMEDIATE per-WG stall sample at first
working dual, hammering set before any cycle claims, >=15% @4K (<=113k)
exit. nb64 solo cycles not yet measured (do at dual-stream baseline time).
Probe hooks remain build-gated off (FSK_PAYLOAD_DUMP/FSK_SCORE_STG/
FSK_KSC_TM0_ONLY); banner-d2 re-verification still queued. References
verified bit-clean after everything.

---

# SQUEEZE BACKLOG (Robert mandate 2026-06-10: crushing SOTA, ceiling claims
# are hypotheses to falsify; running list, measured-or-paper status each)

| # | Idea | Status |
|---|------|--------|
| 1 | Dual-stream (mirrored WG1, indep (b,h) streams) | BUILDING - gate open, TMEM shape per memo to confirm |
| 2 | Pair-QK lookahead at nb64 (currently zero lookahead, QK latency exposed per pair) | PAPER - measure solo cycle delta first |
| 3 | PC-sample the landed dual; rank bounds (Robert m2) | QUEUED behind dual number |
| 4 | Deeper streams (3+; TMEM is the wall) | PAPER - blocked on dual TMEM actuals |
| 5 | Spare/deferred-merge dataflow per stream | PAPER - re-open post-dual (protocol-free shape may re-admit it) |
| 6 | Persistent scheduling on the fused kernel | PAPER - old shape's traps (cold-start, seams) documented; re-test on new shape |
| 7 | qsum machinery vs exp chain | PAPER - Venue B closed at no-win on OLD shape; re-open only with new evidence |
| 8 | One-QK-serving-both-streams (option iii; M=128 stacked Q, Mb=64 streams, lane-stacked TMEM) | PAPER - TMEM fits (448), but M=64-per-stream MMA shape = Robert's call per standing routing |
| 9 | Production nb64 lane repair (pair-granular port) | SCOPED - recipe = session 37 fix; production frozen |
| 10 | NVFP4-vs-MX QK design question | PARKED for Robert (banner) |
| 11 | Banner d2 re-verification under box-unit lens | QUEUED |

---

# Session 38 (2026-06-10): VENUE A DUAL - design delta forced by the
# geometry truth; revised TMEM map (fits at exactly 512)

The frozen design's "scores 2 x 64 single slot per stream" is INVALID: a
64-col QK issue needs a 32-row B tile (idesc N=32) whose upper-half fill is
exactly the bug session 37 killed; the working nb64 QK is PAIR-GRANULAR
(128 cols per issue). Per-stream pair scores (2x128) + out (2x128) = 512
alone => scales cannot fit => REVISED MAP:

- scores: ONE SHARED 128-col pair buffer @0, multiplexed A/B. Pacing: a
  stream issues its pair-QK only after the other stream's LDTM of its own
  previous pair completed (one cross-WG arrive per pair - reintroduced hop,
  but on the SHORT op (QK ~3 chunk MMAs); the long per-tile chains
  (softmax/pack/PV/rescale) remain fully parallel across WGs).
- out: A @128 (128 cols), B @256 (128 cols).
- scales @384..511 (= exactly 128): q_scA 16 + q_scB 16 (per-chunk restaged,
  production-style - NOT the 48-col preloaded form) + k_sc 48 single-buffer
  (both streams' QKs consume the SAME page - same (b,h) k-pair; count-2
  release gates restage) + p_sc 16 (shared constant) + v_sc 32 (shared).
  TOTAL = 512 EXACT.
- WGs: WG0=stream A, WG1=stream B (mirrored, stream-indexed), WG2=producer
  (TMA K/V/Q/scales; K/V slots released by count-2 arrives from both
  streams). Streams own m-tile pair (2c, 2c+1) of the same (b,h).
- SMEM: Q 2x12KB + K 2x12KB(64-row-type pairs) + V shared 2x4KB + payload
  2 streams x 2 x 4KB + scale pages; ~70KB - fits.
- Grid: ceil(tiles_m/2) x H x B; odd tiles_m: stream B idles (iters=0).

---

# Session 38b (2026-06-10): DUAL KERNEL BUILT - lands solo-clean, deadlocks
# with both streams; bisect dossier for next session

BUILT: fwd_fused_dual_kernel.inc (kernel_fp4pv_fused_dual, dispatch
fuseddual_v0, Makefile dep added, include added to fp4_fa4_fwd_experiments.cu,
dual smem sized explicitly in the launcher - the production-derived
accounting under-allocates the dual layout; that was bug #1, illegal addr).
Design per session 38 map (shared 128-col score buffer + ldtm ping; shared
K/k_sc with count-2 k_drained; per-chunk q_sc restage 16 cols/stream;
p_sc/v_sc shared; V private LDG).

STATUS:
- FSK_DUAL_FORCE1 (streams=1, WG0 solo through the dual structure): RUNS,
  finite output => prologue, pair-QK lambda, consumer, epilogue all sound.
- streams=2: HANGS. Fixed already: the odd-iteration refill deadlock cycle
  (refill waits B's drain; B waits A's ldtm arrive later in the same
  iteration) - refills moved to even iterations (i/2+1 cadence).
- FSK_DUAL_NOPING (ldtm ping disabled, arrives kept): STILL HANGS => the
  deadlock is NOT (only) the ping: it lives in {B's plain k_ready wait,
  k_drained count-2, scores_arrived per stream, sc_staged, pv chain}.
- LAW (relearned): in-kernel printf is USELESS for deadlock tracing (buffers
  flush at kernel end). Use device-global progress markers + a SECOND-stream
  host poller, or bisect by construction.

NEXT-SESSION BISECT TABLE (in order):
(a) B-halt probe: streams=2 but B returns right after prologue AND
    k_drained re-inited count-1 (hook) - if A completes, B's loop is the
    deadlock side; if A hangs, the shared-prologue/init is.
(b) k_ready audit: B plain-waits phases of TMAs expected by A's leader.
    Verify the expect/arrive accounting allows a second observer mid-stream
    (suspicion: fp4pv_wait phase indices for B's pairs drift from A's expect
    cadence at p>=2 - count the flips on paper for S=1024 pair=3 (npairs
    7/8) carefully).
(c) scores_arrived: tensor commits from TWO streams' threads into per-stream
    sems - verify tcgen05.commit from B's leader (thread 128) lands on B's
    intended mbarrier and not a pipe-global commit point.
(d) If (b)/(c) inconclusive: device-global progress array (w, i, stage)
    written via STG (the trusted pattern) + read AFTER timeout-kill via a
    persistent allocation (torch tensor passed as o-aliased scratch...
    simplest: re-purpose g.lse rows when disable_lse_store - probe-only).
STATE: production + nb128 references RE-VERIFIED bit-clean after all dual
edits (dual is a separate kernel; shared sources touched: dispatch include,
Makefile, k_gl alt-descriptor (inert, verified)). nb64 single-stream det==0
ladder stands. Probe hooks: FSK_DUAL_FORCE1 / FSK_DUAL_NOPING / FSK_DUAL_TRACE
(printf - useless, scrub later).

---

# Session 39 (2026-06-11): DUAL LANDED CORRECT; latency-hiding hypothesis
# FALSIFIED in the 512-TMEM shape (stall sample); verdict + routing

## Deadlock kill chain (cuda-gdb attach method, per supervisor kit - worked)
#1 smem under-allocation (illegal addr) -> explicit dual sizing.
#2 odd-iteration refill cycle -> even-iteration refills.
#3 missing expect-ownership: the one K block only B consumes was never
   expect_bytes'd (A-only expects) -> tx phase never completed -> B's leader
   spun forever (warp map: A at end barrier, B lane-0 alone at TRYWAIT;
   mbarrier word pending-count nibble). Rule: EXACTLY ONE expect per block;
   B owns the final block beyond A's range.
#4 q_sc phantom slot: config storage has ONE slot; B's TMA into "slot 1"
   landed OOB in k_smem (signature: all B-tiles bad + A's later tiles
   partially corrupted). static_assert added; two private pages allocated.
   (Also: tail ldtm arrive paced - unpaced it is an even-overshoot parity
   alias per the old law; kept though #3 was the actual blocker.)
LAW (tooling): cuda-gdb attach to the GPU-RESIDENT pid (nvidia-smi pid !=
python pid - match /proc/cmdline); pkill -f patterns can self-kill the
invoking shell; in-kernel printf dead in deadlocks.

## Correctness: FULL PASS
det==0 hammered (8 runs x production-interleaved) AND BIT-IDENTICAL vs
fusedstream_nb64 solo (dO=dLSE=0.000e+00) at S=1024/2048/4096. References
untouched (nb128 bit-identical, production floor re-verified this session).

## THE NUMBER + THE STALL VERDICT (S=4096 H=12 B=1 fullgrid, ncu cycles)
fuseddual_v0:      508.0k   <- SLOWER than solo
fusedstream_nb64:  472.5k   (solo, same schedule)
[references: nb128 solo 458.3k; PRODUCTION DEFAULT 133.5k; exit bar 113k]
Per-warp stall sample (immediate, pre-tuning, per directive):
  stalled_barrier 7.76 / 17.4 cyc-per-issued-inst (45%, DOMINANT)
  long_scoreboard 3.60 (V LDG global latency)
  wait 1.47, short_scoreboard 1.03, math_pipe_throttle 0.03, membar 0.
VERDICT: the two streams TIME-SLICE, not cross-hide - the shared 128-col
score buffer's ldtm ping + zero-lookahead pair-QK serialize the streams
(barrier-dominated), and the 512-col TMEM wall is what forced that shape
(per-stream pair buffers + outs + scales do not fit). The >=15%@4K exit is
unreachable IN THIS SHAPE; the fused-family floor itself (solo 458-472k =
~80% exposed latency, session 18) stands 3.4x above production.

## Routing per the standing rules + squeeze backlog updates
- Venue A dual AS-SHAPED: CLOSED (measured, falsified). Backlog #1 -> MEASURED/CLOSED (508k).
- Backlog #2 (pair-QK lookahead): blocked by the same TMEM wall (needs a
  second 128-col pair buffer) - PAPER, fold into shape decisions.
- Backlog #8 (option iii, one-QK-serving-both-streams: stacked-Q M=128,
  Mb=64 streams, lane-stacked TMEM = 448 cols, NO shared-buffer
  serialization - the MMAs themselves serve both streams) is now the ONLY
  dual shape that fits TMEM without serializing: per standing routing THIS
  SHAPE CALL IS ROBERT'S. The measured dual gives the decision data: ping
  serialization costs >36k vs solo; option iii eliminates the ping class.
- The deeper truth candidate: the fused family's 80%-exposed-latency floor
  (458k solo) says single-WG-does-everything pays full serial latency per
  tile; production's 3-WG pipeline hides it (133.5k). The squeeze should
  weigh returning protocol WGs (producer/softmax split) ON TOP of the nb64
  pair-QK fix - i.e., port the SESSION-37 fix to the production nb64 lane
  (backlog #9) and measure THAT against 133.5k, before more venue-A shapes.

---

# DECISION MEMO FOR ROBERT: option iii (stacked-Q dual) - paper scope only
# (2026-06-11, per directive; NOT built)

SHAPE: one CTA serves TWO m-tiles (2c, 2c+1) of one (b,h) with Mb=64 ROWS
PER STREAM, STACKED into single M=128 MMAs: Q = [Q_a(64) ; Q_b(64)], one
128-row QK per K-pair and one 128-row PV per tile serve BOTH streams - there
is no shared-buffer ping because there is nothing to multiplex: the MMAs
themselves are the sharing. Per-WG consumers split by LANES (WG0 owns rows
0-63 = stream a, WG1 rows 64-127 = stream b) over the SAME score tile.
TMEM (lane-stacked, 64-row tensors two-per-column-range via two-arg
allocate): scores 128 (pair buffer, both streams' rows) + out 128 + q_sc 48
+ k_sc 48 + p_sc 16 + v_sc 32 = 400 <= 512 with 112 spare (room for score
double-buffering of ONE more pair = +128 -> 528 OVER; or +64 half = lookahead
options constrained but nonzero).
CYCLE MODEL (grounded in session-39 stall data): the dual's loss was 45%
barrier stalls from ping serialization + zero QK lookahead; option iii
removes the ping class entirely (one issuer, WG-local commits) and halves
MMA count vs two solo streams (one M=128 QK/PV instead of 2x M=128-on-64
work). Floor estimate: solo-nb64 critical path 472k is ~80% exposed latency;
iii's per-tile chain matches the solo's BUT amortizes every MMA + TMA over
2 streams -> projected 0.55-0.75x solo = 260-355k @4K. STILL 2-2.7x ABOVE
production 133.5k: iii CANNOT reach the 113k bar unless the consumer chains
(softmax/pack, the other 55% of stalls incl V-LDG long_scoreboard 21%) also
shrink ~2x. RISK: (a) the lane-split consumer needs warp-level (not
warpgroup) LDTM/pack paths - new code class, the 32x32b LDTM is warp-scoped
so feasible but every warpgroup::sync becomes 2-warp-group coordination;
(b) M=128 epilogue rescale spans both streams' rows - per-stream corr
diverges, so rescale_tt must go lane-predicated (exists: lane_needs_rescale
path); (c) the session-16 paired-ktile failure mode does not apply (no
cross-agent handoff) but was never tested at lane-split granularity.
RECOMMENDATION: do NOT build before the backlog-#9 production-lane port is
measured: #9 keeps the 3-WG latency-hiding that the 133.5k default proves
works, and its projected floor (production nb64 ~ production nb128 + pair
overhead) plausibly LANDS NEAR 133.5k where iii's optimistic case is 260k.
iii is the better *research* shape only if #9 measures poorly AND the
fused-family consumer chains can be halved.

---

# Session 40 (2026-06-11): backlog #9 port plan (production nb64 lane,
# pair-granular QK) - mechanical spec; execute next session

TARGET: fwd_streaming_kernel.inc, gated `if constexpr (C::Nb == 64)` so all
nb128 paths are byte-untouched (references frozen). Lane configs: the
4wg split2wg/rowsplit2wg dualaccum_directrescale_*_nb64 family (dispatch
lines ~690-697) + fp32pack_early_pready nb64 (line ~555).
NOTE: the kernel ALREADY has nb64-aware k coords under
STATIC_TRIPLE_SCORE_TMEM && Nb==64 (k_tile_limit, k_sc_depth=global_idx,
line ~2270) - read that variant FIRST; the lane we measured may or may not
set TRIPLE_SCORE; the port must cover the measured lane's flag combo.

THE CHANGE (mirrors session-37/39 proven fix):
1. K loads: k_tile type -> G::k_tile_alt (<64, Dqk/2>, 128 logical K rows),
   coord = PAIR index (global_idx/2), loads on even global_idx only
   (k_idx/k_phase cadence halved). expect_bytes sizeof updated.
2. QK issue (the issue_next_qk lambda, line ~2890): on even next_idx only,
   ONE issue with the 64-row tile (helper derives idesc N=64 -> 128 D cols)
   into score slots (next_idx, next_idx+1) - REQUIRES those two slots to be
   ADJACENT 64-col TMEM ranges: verify score_tensor_for_slot(0/1) addresses
   differ by exactly 64 (if not, re-base the nb64 score slot allocation to
   adjacent - allocation site near line 826 SCORE_TMEM_SLOTS).
   Commit scores_arrived[slot0] AND [slot1]. The pre-issue waits
   (score copy-done, spare-reuse, p_sc alias) must be taken for BOTH slots
   before the pair issue (the 128-col write overwrites both).
3. k_sc: depth = pair index; k_sc slot cadence halved to match.
4. SOFTMAX/consumer WGs: UNTOUCHED (per-tile consumption; the semaphores
   arm per slot exactly as today).
5. Odd tails: last odd score tile of a task at nb64 - the pair issue covers
   it (its pair partner may be masked-out cols); ensure the k pair coord
   clamps (k_tile_limit analog) and the second slot's commit still fires.
VALIDATION (in order): det==0 hammered on the lane; vs prod-nb128 envelope
(expect the legitimate 64-granular requant ~4-5e-3, NOT 3.6e-2); vs torch
refs at the production floor class; RECORD as the historical-breakage
repair; then ncu cycles vs 133.5k @4K; then the full 1K-16K sweep both
launch modes. Default untouched - the lane is opt-in.

STATUS: plan only this session (option-iii memo for Robert landed above;
budget spent on the dual verdict). The dual (fuseddual_v0) remains in-tree,
correct, measured at 508k - a falsified-but-clean experimental lane.

---

# Session 41 (2026-06-11): BACKLOG #9 LANDED - production nb64 lane REPAIRED
# (historical det-breakage fixed in production code); measured: not a perf path

## The port (fwd_streaming_kernel.inc, STATIC_NB64_PAIR_QK)
Gate: (Nb==64 && MXFP4_PV && SCORE_TMEM_SLOTS==2 && !ALIAS_SCALE && !TRIPLE
&& COMPACT_Q_SCALE && LOAD_STAGES==2) - nb128 paths byte-untouched
(VERIFIED: fused nb128 bit-identical, production at canonical refs after the
port build). Seven edit sites: flag def; k_tile_alt prefetch; loader pair-
gating (one 6KB <64,Dqk/2> pair tile per even global_idx spanning both 3KB
k slots, single-buffer k_phase cadence, k_sc depth=pair); issue_next_qk
(odd idx = no-op except task-final q_finished commit; even idx issues the
64-row tile -> idesc N=64 -> 128 score cols across BOTH adjacent slots
(score_tensor_for_slot(s) = base + s*Nb, adjacency by construction);
slot-1 copy-done pre-wait; commits scores_arrived[0]+[1]; pair k cadence);
task-prologue inline QK(0) same treatment. iters_per_task alias threaded
(nb64_pair_iters) - the lambda predates the per-task declaration.
Note: at nb64 iters_per_task = 2(m+1) is ALWAYS EVEN => pairs always full;
the odd-tail arm exists but never fires => persistent task seams keep
uniform phase advancement (no seam hazard).

## Validation (the lane's first healthy state EVER)
- det: dO=dLSE=0.000e+00 over 6 runs (was 4.4e-2 NON-deterministic).
- vs prod-nb128: dLSE=1.431e-06 FLOOR (was 1.9e-2); dO=2.34e-02.
- vs torch refs: dLSE=9.5e-07 (BETTER than nb128's own 1.4e-3 @4096!);
  dO=2.3-3.0e-2 - ATTRIBUTED: the designed 64-granular local-max pack
  envelope (cross-check: the healthy fused-nb64 measures 1.7e-2 vs refs,
  same class). RECORDED AS THE HISTORICAL-BREAKAGE REPAIR: root cause was
  the <32,Dqk/2> k_tile's idesc N=32 filling 32/64 score cols, upper half
  stale TMEM (run-dependent => the det breakage; session-37 diagnosis).

## THE NUMBER + SWEEP (ncu gpc__cycles_elapsed.max, fullgrid, H=12 B=1)
   S      nb64-repaired   prod-default   ratio
 1024        68.1k           32.7k       2.08x
 2048       122.9k           51.9k       2.37x
 4096       314.8k          132.9k       2.37x
 8192      1126.6k          422.6k       2.67x
16384      4077.6k         1510.5k       2.70x
VERDICT: the repair is CORRECT and shippable as a bug fix; as a PERF path
the lane is 2.1-2.7x behind the default - the "plausibly lands near 133.5k"
hypothesis is MEASURED FALSE. Structural costs: 2x tile iterations with
per-tile consumer protocol, single K pair buffer (LOAD_STAGES=2 collapses
to 1 pair slot - a LOAD_STAGES bump is the obvious partial squeeze), and
the lane's never-tuned 4wg split2wg shape. Default remains UNTOUCHED and
UNBEATEN at 132.9k.

## SQUEEZE BACKLOG updates
#9 -> LANDED/MEASURED (repair shipped-class; perf path falsified 2.4x).
#1 dual: CLOSED 508k. The board's remaining live perf ideas: option iii
(Robert's memo above), LOAD_STAGES bump on the repaired lane (paper, small),
persistent scheduling on fused (paper), spare/deferred-merge (paper).
The 133.5k default has now survived: phase-1/2 protocol work, venue B,
venue A solo+dual, and the nb64 repair - every measured challenger to date.

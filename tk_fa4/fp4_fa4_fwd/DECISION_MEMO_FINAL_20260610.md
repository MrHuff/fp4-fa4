# FP4 FA4 forward — consolidated decision memo (final, 2026-06-10)

For Robert. Fourteen sessions, ~40 measured variants. Everything below is ncu
`gpc__cycles_elapsed.max` at S=4096 H=12 B=1 fullgrid on cuda:3 unless stated.
Full forensics in `FWD_PROFILING_NOTES_20260609.md`; earlier interim memo
(`DECISION_MEMO_20260610.md`) is superseded by this one.

---

## (a) Shipped state and the measured ladder

**Shipped default** (unchanged, verified end-to-end at close):
`dualaccum_directrescale_q208_p96_o48` + dual corr slots — **133.5–133.7k cycles**,
bit-deterministic run-to-run (was ~5e-3 race noise at the start of the night),
persistent mode fixed and bit-identical to fullgrid, auto launch mode
(persistent <8K, fullgrid ≥8K) safe and empirically optimal. Wall-clock vs BF16
across S=1K–16K (clock-noisy; ratios from the freshest sweep):
0.73 / 0.73 / 0.94 / 0.96 / 1.29. FP4 **wins at every shape except 16K**.

**The full ladder of the night** (all bit-validated unless noted):

| Variant | Cycles | Verdict |
|---|---|---|
| start of night (racy build) | 129.0k | non-deterministic, persistent deadlocks |
| + dual corr slots (**shipped**) | 133.4k | correctness cost +3.4%; everything below compares to this |
| earlycorr / 4-way sums / o56 / config stack | ±1% | noise |
| o40 (narrow rescale) | 136.9k | rescale width matters |
| dual-QUANT ping-pong (4WG, bit-exact) | 144.3k | +8% |
| P-in-TMEM (bit-identical) | 136.0k | +2%; strategic value only |
| prefetch interleave, all placements | 142–194k | data-starved (see hypotheses) |
| QK two-ahead (any family) | 239–275k | copy-done lookahead serializes lane |
| Nb=64 diet | 317.5k | 2.38×: hop count doubles |
| spare family, free-running | 128.7k | **racy** — the only sub-default number, and it is unsafe |
| spare, count-paced depth-1 | 135.6k | correct |
| spare, depth-2 + dual pv phases | 135.9k | correct; pace depth was not the constraint |
| cta2 cluster config (as registered) | 163.2k | +22%, stale numerics (pre-dual-corr bitrot) |

**The ~40 variants compress into six falsified hypotheses:**

1. *"It's compute-bound."* No: −19% instructions → −2% cycles; 4-way sums ±0;
   2× QUANT warpgroups +8%; tensor pipe never exceeds ~7% of peak.
2. *"More pipeline slots/lookahead help."* No: pstage3/4 ±0; QK-two-ahead
   catastrophic (+80–100k) because the copy-done lookahead serializes the lane.
3. *"The QUANT load latency can be prefetched away."* No: under the production
   issue cadence QK(idx+1) drains the in-order tensor queue at ~QUANT(idx)'s
   END (probe hit-rates measured 0% / 7%) — there is nothing to prefetch.
4. *"Removing the P-through-SMEM hops pays."* No: those hops were off the
   critical cycle (P-in-TMEM bit-identical but +2%).
5. *"Taking the rescale out of the PV→PV gap pays."* Only unsafely: every
   COUNT-CORRECT pacing of the spare dataflow (depth-1, depth-2+dual-pv) costs
   more than the gap saves; the racy −3.5% was missing synchronization.
6. *"More CTAs/SM via smaller tiles."* No: TMEM admits 2 CTAs only at Nb=64,
   and Nb=64 alone is 2.38× slower (the period is per-tile hops; halving the
   tile doubles them).

**What the night established as fact:** the kernel is a latency-bound serial
loop, ~1100 cy/tile: ~350 cy of synchronization hops + rescale positioning and
~750 cy of serial segments (LDTM+mask ~200, max tree ~100, exp/pack ~600 for
128 rows, rescale ~300–400 inside the PV→PV gap, MMA latencies). Within this
dataflow (one pipeline per CTA, one softmax recurrence per tile), the shipped
default is the measured optimum. Side products now in tree, gated/registered:
dual-corr protocol (the determinism + persistent fix), P-in-TMEM path
(bit-identical; 16 KB SMEM freed; hardware-validated A-from-TMEM MMA), de-fused
cvt (ptxas bug workaround), dual pv-phase machinery, the protocol laws
(mbarrier parity breaks on even overshoot; every single-semaphore phase chain
needs count-enforced overshoot ≤1; phase state must be cumulative across task
seams), and a validation harness (bit-identity bar, cold-trap hammering,
mixed-task-size persistent seam tests, debug-ring + SMEM-map forensics).

---

## (b) Outbound items (both original "author questions" were answered locally)

1. **ptxas codegen bug — report-ready.** `cvt.rn.satfinite.e2m1x2.f32` ×4 +
   byte-mov lowers to a fused `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` chain;
   under register pressure ptxas feeds a stale register into a chain head
   (standalone repro: a chain head literally reading R1 with STACK:0 →
   satfinite(garbage)=0 nibble; host-validated; perturbation-sensitive).
   Repro harness: `/tmp/shape_micro.cu` (preserve into the repo if desired).
   Workaround shipped: `fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte_defused`.
   Worth filing with NVIDIA; nothing of ours blocks on it.
2. **Upstream gaps worth contributing:** TK lacks the A-from-TMEM wrapper for
   `tcgen05.mma.kind::mxf4nvf4.block_scale` (we proved it hardware-correct,
   bit-exact, dense lane-major layout, K64 chunks at +0/+8 cols with
   scale_factor_id 0/2); CUTLASS has no `_TS` atom for block-scaled kinds.
   Also: TK's `try_wait` wraps the POTENTIALLY-BLOCKING `mbarrier.try_wait`;
   a true `test_wait` probe belongs in the library.

---

## (c) Algorithm-shape options (paper-scoped, with arithmetic)

The remaining frontier is the dataflow shape itself. Constants used below:
TMEM = 512 cols/CTA (128 lanes × 32b); regfile 64K/CTA; SMEM ~228 KB/SM;
today's map: scores 2×128 + out 128 + scales 128 = 512 exactly; 168 regs/thread
≈ the whole file; SMEM 104 KB.

### Option 1 — `cta_group::2` paired MMA, Mb=256 per 2-CTA cluster

Mechanics: one MMA spans the CTA pair; each CTA's TMEM holds its 128-lane half
of the M=256 accumulator and scores. **Per-CTA TMEM unchanged** (scores 2×128,
out 128, scales 128 — same 512); registers unchanged; SMEM +12 KB (Q is 256
rows, 24 KB fp4, K/V shared via TMA multicast → per-CTA K/V bandwidth halves).
Effect on the serial cost: per-ROW exp/pack is unchanged (each CTA still
processes its 128 rows), but the per-TILE fixed costs — the ~350 cy hop budget,
QK issue, K/V staging, rescale handshakes — amortize over 2× the rows, and the
M-tile count halves, so total tiles halve at ~constant per-tile period.
**Projected ceiling: ~1.6–1.9× — the largest on paper.**
Risk: HIGH, and tonight added evidence. The registered cluster config
(Mb=128 × CLUSTER=2, which already shares loads and issues `cta_group::2`
MMAs) measures **163.2k (+22%) with broken numerics** — the cluster lane never
received the dual-corr-era correctness pass, and cluster-remote semaphore hops
(`tma::cluster::wait`, remote arrives) are MORE expensive than the local hops
that already bound the period. The paired-MMA Mb=256 shape reduces protocol
hops relative to that config, but every shared resource still signals across
the cluster. Realistic discount: the 1.6–1.9× paper ceiling shrinks by
whatever the remote-hop tax is; the +22% measurement says that tax is large
until the cluster protocol gets the same treatment the local one got (expect
several sessions of seam/cold-trap work — the failure class is known).

### Option 2 — two independent (b,h) tile streams in one CTA (lane-split 64/64)

Mechanics: stream A occupies TMEM lanes 0–63, stream B lanes 64–127, SHARING
columns (the allocator's two-arg lane-offset path exists). One QUANT WG: lanes
0–63 process A's rows, 64–127 B's rows — per-thread work identical to today,
no divergence (per-row recurrences are already per-lane registers). No softmax
recurrence between streams, so every hop/stall of A is hidden by B's compute
and vice versa. TMEM: scores 2×128 cols (shared columns, split lanes) + out
128 + q/k scales shared per... **caveat: A and B have different (b,h) so K/V
AND their scales differ** → per-stream p_sc/v_sc: p 2×16 + v 2×16(single-slot
each) = +32 over today → 512+32 **over budget** unless v_sc is single-slotted
per stream AND the P-in-TMEM map is not also used (they compose to exactly 512
without P-in-TMEM, or need p_sc single-slotting with it). Tight but feasible.
SMEM: +20 KB (second K/V stage set) → ~124 KB, fits. Registers: unchanged
per thread. Scheduling: pair streams with EQUAL m-tile (same m, different h —
plentiful at H≥2) so iteration counts match.
**Projected ceiling: ~1.4–1.6×** (the period approaches the compute-bound sum
of segments because stalls fill with sibling work; total per-tile work is
unchanged, so the gain is exactly the stall fraction — measured >60% of
samples are spins, but not all spins are fillable).
Risk: MEDIUM-HIGH. Entirely CTA-local (no cluster tax — its structural
advantage over option 1) and all protocol laws apply directly, but it is the
biggest mechanical build of the three: every semaphore family, slot map, and
the lane's issue cadence duplicates per stream. The seam/cold-trap harness is
ready for it.

### Option 3 — K-split across CTAs for long S (FlashDecoding-style)

Mechanics: split each (b,h,m) row-block's K-walk across G CTAs; each computes
a partial (O, m, l) with the EXISTING online recurrence (scale/amax math
untouched by construction); a trivial merge kernel combines partials
(log-sum-exp merge — exact, standard). Per-CTA TMEM/registers/SMEM identical
to today. Extra traffic at S=16K, G=4: partials = 4×(128×128×4B + 1KB) ≈
260 KB per row-block vs ~25 MB of K/V reads — negligible.
**Projected effect: only at long S — which is exactly where FP4 loses today**
(1.13–1.29× at 16K). At 16K the serial K-walk is 128 tiles; G=4 cuts the
per-CTA walk to 32 tiles + merge, and quadruples occupancy at small batch.
Expect the 16K ratio to move from ~1.2 to roughly the 8K ratio (~0.95) since
per-CTA work becomes 8K-shaped. No effect at ≤8K (don't enable there).
Risk: LOW-MEDIUM. Standard technique; host/dispatch + a merge kernel + a
G-way LSE-merge epilogue; no new intra-CTA protocol. The one watch-item is
determinism of the merge order (fixable by fixed-order merge).

### Option 4 — do nothing here: ship the default, move effort to bwd

The fwd kernel already beats BF16 at every shape except 16K (0.73–0.96×), is
bit-deterministic, and persistent-safe. Backward is ~2–2.5× of training FLOPs
and has not had any of this treatment; its expected headroom per engineering
hour is far higher than squeezing the last 1.4× out of a fwd kernel that
already wins. The 16K cliff is the only real fwd gap, and option 3 addresses
it surgically without touching the core dataflow.

---

## Ranking and recommendation

**1. Option 4 + Option 3 as the only fwd follow-up.** Ship today's default.
Redirect effort to bwd (or the fused-quant/norm kernels per the repo goals).
Schedule option 3 (K-split) as a contained, low-risk project targeting the one
shape where fwd loses; it reuses the existing recurrence and validation
harness and does not perturb the shipped protocol.

**2. Option 2** if a fwd structural rebuild is wanted anyway: CTA-local, laws
and harness apply directly, honest ceiling 1.4–1.6×, biggest build cost.

**3. Option 1** only after (or together with) a cluster-protocol correctness
pass: largest paper ceiling (1.6–1.9×) but tonight's +22%/broken-numerics
measurement of the existing cluster lane says the remote-hop tax and protocol
debt are real; treat the paper number as an upper bound that the cluster tax
will eat into until proven otherwise.

Not recommended: any further synchronization-level variants of the current
shape — the ladder above is the exhaustive answer to those.

---

*Validation bar used throughout and recommended for whatever comes next:*
bit-identical O/LSE where the math is unchanged (any drift = a bug), scale/amax
math byte-identical always, determinism = 0, cold-start traps 10/10, persistent
task-seam hammering with mixed task sizes including iters==1, S=1024 included
in every validation set, full 1K–16K sweep in both launch modes before any
default change, ncu cycles as ground truth (wall clock is clock-noisy on this
box; GPUs 1/2 remain wedged — use cuda:3).

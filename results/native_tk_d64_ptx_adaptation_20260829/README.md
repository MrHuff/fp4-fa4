# Native-TK D64 causal GQA backward: PTX-guided adaptation receipt

Date: 2026-08-29 UTC
Repository: `fp4_matmul_monolithic_tk_20260828`
Branch: `codex/monolithic-tk-fa4-train-20260828`
Receipt base commit: `1ede7dc715a58ea91165997b78d420439a3585b3`

## 2026-08-29 late-optimization addendum

**This addendum supersedes the historical executive result below.** The older
v384-v387 text and ledgers remain as a reconstruction receipt, but they are no
longer the current performance conclusion. The validated native lineage now
reaches v406, and the authenticated CuTe comparison has been separated into
its public fast policy and a native-EX2 diagnostic so that timing and numerical
contracts are not conflated.

Unless explicitly labeled diagnostic or inference, the results in this
addendum are verified on one NVIDIA GB200 at
B16/S4096/Hq32/Hkv8/D64 causal backward. Native timings use the full wrapper,
including three destination clears, with 10 warmups and 51 per-sample-rotated
measurements. The native public ABI consumes raw FP32 `l_aux` and `delta`.
The v414/v416 rows are explicitly different: they consume the production BSHD
tensor layout and prelifted `lstat`/`dstat` pages. Isolated kernel and
end-to-end transformer claims are labeled separately below.

### Verified native progression

The validated full-wrapper medians are:

| Candidate | Principal change | Full p50 (ms) | Verified comparison |
|---|---|---:|---:|
| v388 | split P and dS lifetimes | 5.795904 | paired parent for v394 |
| v394 | retain P in TMEM; use true TMEM-A for dV | 4.762112 | 1.217087x over paired v388 |
| v397 | publish each 64-column P half early and overlap its dV work with dS | 4.441504 | 1.071839x over paired v394; 1.304942x contextual over v388 |
| v400 | clean aggregate of native-EX2/clamp traversal, TMA raw statistics, asynchronous gradient publication, and v397 | 4.145152 | 1.071918x over paired v397; 1.398237x contextual over v388 |
| v403 | add owner-aligned score/dP/P/dS conversion to v400 | 4.098048 | paired v400 was 4.147456 ms: 1.012056x |
| v404 | issue/commit dQ before dK so reducers can drain dQ concurrently | **3.875968** | paired v400 was 4.145440 ms: **1.069524x** |
| v406 | compose v403 owner alignment with v404 dQ-first scheduling | **3.803328** | matched v403/v404 were 4.095392/3.875584 ms: **1.076792x/1.018998x** |
| v414 | adapt v406 to production BSHD and prelifted `+8/-16` statistics | **3.736960** | paired corrected-raw v406 was 3.800992 ms: **1.017135x** |
| v415 | vectorize adjacent raw-ABI dS publications with exact `st.shared.v2.b32` | **3.766592** | paired v406 was 3.805472 ms: **1.010322x** |
| v416 | port v415's exact vector publication to production v414 | **3.720256** | paired v414 was 3.738464 ms: **1.004894x** |

Rows come from several matched AB runs rather than one uninterrupted
multi-way run; only the explicitly paired ratios are direct comparisons.
Cross-generation ratios to v388 are engineering context. v403 was finite,
preserved the zero-dOut exact-zero invariant, and agreed with v400 at
B1/S4096 to relative-L2 0.000140/0.000962/0.000999 for dQ/dK/dV. v404 used
REG128 with no spills, preserved the static MMA, EX2, SHFL, and synchronization
counts, remained finite, and agreed with v400 to relative-L2
0.0000723/0.000230/0.000224 for dQ/dK/dV; zero dOut again produced exact zero
gradients. The v404 result is therefore a measured scheduling win, not a
reduction in attention math or a relaxed probability approximation.

v406 was measured in one matched 10-warmup/51-sample three-way run with v403
and v404. It retained REG128/STACK0, removed all static `SHFL` sites, remained
finite, and preserved the exact-zero result for zero dOut. Its 1.8998% gain
over v404 establishes that owner alignment and dQ-first scheduling compose;
the larger 7.6792% gain over v403 is dominated by the independently validated
dQ publication-overlap fix.

v415/v416 remove the final identified exact-path issue: each compute warp had
published a 64-column dS half with eight scalar shared-memory stores. Adjacent
packed E4M3 words remain contiguous because each lane's 32-byte owner span
stays within one 128-byte swizzle segment. Replacing each adjacent pair with
one `st.shared.v2.b32` reduced the production static instruction count from
3120 to 3088, scalar `STS` sites from 18 to 2, and introduced eight `STS.64`
sites. v416 remains REG128/STACK0 with no spills. At B16/S4096 its v414-relative
dQ/dK/dV changes (relative-L2 0.0000248/0.0002826/0.0002935) are comparable to
atomic repeat noise, all outputs are finite, and zero dOut remains exactly
zero.

Two bounded experiments sharpen what did and did not matter:

- v399 removed the ownership exchange on the v394 path and improved full
  latency from 4.764256 to 4.714176 ms, 1.010624x. Composing that conversion
  as v403 removed all 64 static `SHFL` sites from v400, but increased static
  TMEM-load sites from 10 to 14 and retained 32 `PRMT` sites; its net paired
  gain was 1.012056x. Ownership traffic was real, but eliminating it moved
  part of the cost into extra TMEM loads and scalar dS stores rather than
  removing the whole recurring phase.
- v402 deleted four source-level warp synchronizations from v397, but its SASS
  still contained exactly 13 `WARPSYNC.ALL` and 8 `BAR.SYNC` sites, identical
  to v397. Full latency changed from 4.455392 to 4.459776 ms
  (`0.999017x`, slightly slower). The compiler had already removed or folded
  those source synchronizations, so v402 is not a performance successor.

The subsequent bounded candidates also rule out several tempting shortcuts:

- v408 composed the prelifted-statistics ABI with v406, but regressed from
  3.802496 to 3.870368 ms in a paired run (`0.982464x`, or 1.785% more
  latency). Removing a small amount of scalar transform work did not improve
  the now-tighter recurring schedule, so raw-statistics v406 remains the
  measured successor.
- v405 attempted a full exact-FP32 P fragment, but compiled at REG128 with a
  96-byte stack frame and 24 spill registers. v407's 448-thread variant did
  not remove those spills and was rejected as unsafe because its final
  warpgroup was incomplete for the synchronous register-allocation protocol.
  v409's 16-column dP chunks reduced the frame to 64 bytes and 16 spill
  registers, but did not cross the no-spill gate. None of v405, v407, or v409
  was promoted to a GPU performance claim.
- v410 retains the full dS probability fragment as packed FP16. It builds at
  REG128/STACK0/LOCAL0/SHARED117984 and, in a rotated 10-warmup/51-sample
  B16/S4096 run, measured 3.797760 ms versus v406 at 3.807008 ms: only a
  1.002435x speedup. Its deliberate FP16-P approximation changed v406-relative
  dQ/dK/dV by relative-L2 0.004875/0.005273/0.000247 at B16/S4096, with cosine
  0.999988/0.999986/1.000000. Every result was finite and the zero-dOut,
  zero-delta case remained exactly zero. This is a measured approximate
  candidate, but the 0.2435% gain is too small to replace exact v406.
- v412 compacts the source declaration of v406's exact owner fragment and also
  builds at REG128/STACK0/LOCAL0/SHARED117984, with zero spill loads or stores
  and core operation counts identical to v406. It remained finite and within
  the atomic repeat noise of v406, but regressed to 3.895328 ms in that same
  run, 2.320% more latency than the 3.807008-ms v406 cell. The compiler had
  already eliminated the over-declared source half from the important machine
  path; source compactness was not the remaining speed lever.
- v411 tested a compact full exact-FP32 P representation, but still compiled
  at REG128/STACK64 with 68 bytes of spill traffic in each direction. It was
  rejected without GPU timing. Together with v412, this rules out the dead
  source half-stride itself as the cause of full-P register pressure.
- v413 transplanted cd57's authenticated degree-1/period-2 selective exponent
  policy into the v406 schedule. Although it reduced static native-EX2 sites
  from 64 to 32, it added 104 instructions while leaving MMA, TMEM,
  synchronization, and branch counts unchanged. In a rotated
  10-warmup/51-sample B16/S4096 run, v413 measured 3.893248 ms versus v406 at
  3.813856 ms: 2.081% more latency. Its v406-relative dQ/dK/dV differences
  were relative-L2 0.05387/0.05391/0.05467, cosine about 0.9988, and norm ratio
  about 0.976. All outputs remained finite and zero dOut produced exact zero
  gradients. The policy is beneficial in cd57, but this direct composition is
  a measured TK regression and was rejected.

### Production data-ABI adaptation: v414

v414 is a separate, non-invasive adaptation of v406. It retains the same
K128/Q128 owner-aligned probability path, two-stage gradient publication, and
dQ-before-dK tensor-issue schedule, while changing operand and output TMA
descriptors from contiguous BHSD to the production contiguous BSHD layout. It
also consumes the forward-produced statistic pages directly:
`lstat = 8-LSE*log2(e)` and `dstat = -16*sum(O*dO)`. The selected main kernel
matches v406's resource gate at REG128/STACK0/LOCAL0/SHARED117984 with no spill
loads or stores.

The production `+8` is a material semantic detail, not a naming change. The
first v414 build removed v406's post-EX2 multiply by 256 but accidentally kept
v406's `min(log2(P), 0)` clamp. Because the production statistic already makes
the exponent `log2(256*P)`, that clamp limited `Pscaled` to one. It selectively
collapsed the first causal blocks: at B1/S4096, block-zero dQ norm fell to
about 0.182x while blocks two through 31 were already near unit norm. The
correct production path removes the post-EX2 multiply and clamps the prelifted
exponent at `+8`, yielding `Pscaled = 256*P` without double lifting.

After that correction, v414 agreed with v406 under the explicit raw-reference
transforms `l_aux=(lstat-8)/(softmax_scale*log2(e))` and
`delta=dstat/-16`. At B16/S4096, dQ/dK/dV relative-L2 was
0.000179/0.000378/0.000345, all cosines rounded to at least 0.9999999, and all
norm ratios were within 0.000002 of one. B1/S128 and B1/S4096 also passed,
every output remained finite, and zero dOut produced exact zero gradients.
Against retained production v382 at B16/S4096, v414's norm ratios remained
near one and cosine was 0.997824/0.997834/0.999994 for dQ/dK/dV; the larger
dQ/dK difference is consistent with v414 retaining exact FP32 P for dS while
v382 reuses rounded E4M3 P.

In a paired, per-sample-rotated 10-warmup/51-sample B16/S4096 run with output
clears included, corrected-raw BHSD v406 measured 3.800992 ms and production
BSHD v414 measured 3.736960 ms, a 1.017135x speedup. This closed the isolated
layout/statistics ABI gap without losing the dQ-first gain. The compact
machine-readable receipt is `v414_production_bshd_receipt_20260829.json`.

The production runner now authenticates v414 and the selected v416 by exact
source identity and ABI metadata, then calls their clearing direct-BF16 output
entrypoint. The retained v382 route remains accepted through its original
13-argument accumulator/partial API. For v414/v416, one zero-length CUDA
sentinel preserves the shared-runtime identity audit while eliminating v382's
FP32 dQ accumulator and dK/dV partial buffers. At B16/S4096 this removes
exactly 1 GiB of persistent runner storage. The non-clearing v414/v416 `main`
entrypoint is deliberately not used.

Build the selected extension with
`make -C tk_fa4 native-d64-backward-v416`. The saturated runner still requires
the extension path, module name, SHA-256, and byte count explicitly, so a
different binary cannot silently replace the authenticated route.

### Saturated 1.2B end-to-end result

The selected v416 route was run on one GB200 with a 16-layer Llama-1.2B shape,
B16/S4096 (65,536 tokens/update), a fixed synthetic checkpoint/token stream,
torch-compiled cross entropy, three warmups, and 20 measured updates. This is
a source-stable saturation and short numerical-drift diagnostic, not a
long-training convergence result.

| Route | Step p50 (ms) | Total backward p50 (ms) | Decoder forward p50 (ms) | p50 tok/s | Useful MFU | Peak allocated/reserved GiB | Held-out loss, initial -> final |
|---|---:|---:|---:|---:|---:|---:|---:|
| Packed BF16 | 679.384 | 444.223 | 196.470 | 96,464 | 35.24% | 160.711 / 173.326 | 12.17247 -> 11.94001 |
| FP8 forward + retained v382 TK backward | 1883.245 | 1666.970 | 178.867 | 34,800 | 12.71% | 156.032 / 167.980 | 12.17158 -> 11.93875 |
| **FP8 forward + selected v416 TK backward** | **618.810** | **404.271** | **177.699** | **105,906** | **38.69%** | **155.032 / 166.980** | **12.17158 -> 11.93916** |

All 23 v416 records were finite. Relative to packed BF16, v416 is 1.09789x
faster per full step and 1.09882x faster over the total backward, while using
5.680 GiB less peak allocated memory. Relative to the retained v382 route it
is 3.04333x faster per step and 4.12339x faster over total backward. Its
20-update loss-trajectory MAE versus BF16 is 0.00606; the final held-out-loss
difference is -0.00085.

The causal attribution is unusually tight. A same-process production-ABI
B16/S4096 attention-backward A/B measured v414 at 3.740992 ms versus retained
v382 at 82.816643 ms, a 22.1376x kernel-boundary speedup. Multiplying the
79.076-ms per-layer reduction by 16 layers predicts about 1,265 ms, matching
the observed total-backward reduction from 1,666.970 to 402.411 ms. The old
end-to-end bubble was therefore the retained v382 attention implementation
and its FP32 accumulator/partial/finalization contract, not Python dispatch or
an unsaturated model fixture. The final v416 end-to-end receipt is
`v416_llama12b_saturated_receipt_20260829.json`.

### Verified statistics-ABI attribution

v401 is a deliberately **non-drop-in** experiment, not a replacement public
ABI. It accepts externally prelifted probability and dS row statistics and
excludes those transforms from the timed boundary, modeling a future
forward-fused producer. Against v400 in a matched run, full p50 changed from
4.157792 to 4.128384 ms: a 1.007123x speedup, or 0.7073% latency reduction.
The B16 relative-L2 differences were 0.000158/0.000223/0.000199 for dQ/dK/dV,
and every output remained finite. The standalone producer-transform cost was
not measured. This result bounds the prelift advantage in the backward kernel;
it does not establish that an unfused caller gets that saving for free, and it
does not explain the former 23% public-fast CuTe gap.

### Audited recurring critical path

The native deficit was not extra tensor-core math. Both v400 and authenticated
CuTe cd57 dynamically issue 16 FP8 MMA commands per query tile: two score, two
dP, four dV, four dK, and four dQ commands. v400's 40 versus cd57's 32 static
MMA sites are duplicated control-flow sites, not additional dynamic MMAs.

The material difference was ordering and fragment lifetime:

- v400 issued dK, then dQ, waited for dQ publication to drain, and only then
  advanced to dP. cd57 commits dQ first, performs dK while four reducer warps
  drain dQ, waits only at the alias boundary, and then advances to dP. v404's
  isolated 1.069524x gain from adopting dQ-before-dK directly validates this
  host-independent scheduling bubble.
- v400 formed exact FP32 P and dS in two sequential 64-column phases. Its
  original path had 64 `SHFL` sites, four small `STTM.x2` stores, and two
  half-ready events. cd57 keeps a full exact-FP32 P fragment for dS, uses no
  `SHFL`, uses two `STTM.x8` stores, and has one logical P commit. v403 removes
  the shuffle exchange, but retaining the half-fragment schedule explains why
  that change alone yields only about 1.2%.
- Even after owner alignment, v406/v414 emitted 16 scalar dS shared stores per
  compute warp/query iteration. v415/v416 vectorize the same exact bytes into
  eight 64-bit stores and reduce repeated swizzle-address formation. This is
  the final measured recurring-slope win; no probability approximation or
  full-P register lifetime is involved.

This is now a more specific diagnosis than “TK is slower”: the major verified
loss was a dQ publication bubble, followed by half-fragment probability/dS
serialization and conversion overhead. It was not a grid-size mismatch,
wrapper dispatch, spills in the selected kernels, or extra dynamic attention
MMAs.

### Fixed-grid scaling evidence

With total tokens fixed at 65,536, the grid fixed at 16,384 CTAs, all-zero
operands, and only the triangular causal loop length varied, the retained
full-wrapper fits are:

| Candidate | Fit in ms, where `x` is average causal iterations | R-squared | Claim level |
|---|---|---:|---|
| v394 | `~0.602 + ~0.249 * x` | 0.999995 | verified sweep; coefficients retained only at rounded precision |
| v397 | `0.637298 + 0.229145 * x` | 0.999994 | verified sweep |
| v400 | `0.484607 + 0.221492 * x` | 0.9999975 | verified sweep |
| v406 | `0.490561 + 0.200522 * x` | 0.9999984 | verified sweep |
| v415 | `0.489018 + 0.198183 * x` | 0.9999942 | verified exact vector-store sweep |
| cd57 native-EX2 policy | `0.645080 + 0.187955 * x` | 0.9999736 | diagnostic sweep |
| cd57 public period-2 policy | `0.698751 + 0.160499 * x` | 0.9999308 | diagnostic sweep |

The v394-to-v400 progression reduced the recurring slope, but v400 still
spent about 88% of B16/S4096 full latency in the fitted recurring component.
This makes the residual v400 cost overwhelmingly in-CTA causal work rather
than a long-sequence occupancy collapse. The measured v406 medians for
B512/S128, B256/S256, B128/S512, B64/S1024, B32/S2048, and B16/S4096 were
0.693504, 0.791680, 0.990176, 1.391776, 2.194336, and 3.799904 ms. Relative to
v400, v406 cuts the fitted recurring slope by 9.47% while adding only about
0.006 ms to the fitted intercept. This directly localizes its gain to repeated
causal-tile work rather than launch or fixed wrapper overhead.
v415 then lowers the exact recurring slope another 1.17%, from 0.200522 to
0.198183 ms per average causal iteration, while slightly reducing the fitted
intercept. That agreement between instruction audit, fixed-grid scaling, and
B16 paired timing attributes the gain to vectorized dS publication.

The cd57 fits are diagnostic because shapes above B16 required a local
route-gate surrogate even though the authenticated source and selected kernel
policy were otherwise retained. In that same sweep, B16/S4096 medians were
3.748544 ms under native EX2 and 3.350272 ms under the public period-2 policy.
The policy therefore reduces cd57's fitted recurring slope by 14.61%, from
0.187955 to 0.160499 ms per average causal iteration. Under the closer
native-EX2 comparison, v406's remaining fitted slope excess is only about
0.012567 ms per iteration; its lower fitted intercept offsets most of that
excess at B16/S4096. The larger gap to public period-2 cd57 is real under that
policy, but v413 shows that the approximation's gain is not automatically
portable to the otherwise unchanged TK schedule.

### Authenticated CuTe fairness audit

“Authenticated exact CuTe” in this receipt means the exact cd57 source
artifact, SHA-256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`;
it does **not** mean that every exponent evaluation in its public fast policy
uses native EX2. The public boundary intentionally uses a degree-1,
period-2 selective polynomial policy and externally prelifted statistics.
That is a valid performance/accuracy choice, but it is a different numerical
and producer ABI from clean native v400.

The comparison therefore has two claim levels:

| Comparison | Full p50 (ms) | Ratio | Claim level |
|---|---:|---:|---|
| v400 vs authenticated cd57 public fast policy | 4.147488 vs 3.364672 | 1.232657x | verified matched timing; both finite |
| cd57 native-EX2 policy vs its public period-2 policy | 3.766816 vs 3.367584 | 1.118551x | controlled diagnostic on the same represented fixture |
| v400 vs cd57 native-EX2 policy | 4.147488 vs 3.766816 | about 1.101x | cross-run inference from the two measurements |
| v404 vs cd57 native-EX2 policy | 3.875968 vs 3.766816 | about 1.029x | cross-run diagnostic context, not a matched AB result |
| v404 vs cd57 public fast policy | 3.875968 vs 3.367584 | about 1.151x | cross-run diagnostic context, not a matched AB result |
| **v406 vs cd57 native-EX2 policy** | **3.810560 vs 3.758176** | **1.013939x** | **same-process matched diagnostic; 1.394% residual** |
| **v416 vs cd57 native-EX2 policy** | **3.729120 vs 3.749088** | **0.994674x** | **same-process matched diagnostic; TK 0.535% faster** |
| v416 vs cd57 public period-2 policy | 3.729120 vs 3.356064 | 1.111159x | same-process timing under different exponent policies |

The native-EX2 CuTe diagnostic used a local in-memory verifier surrogate
because the public route gate rejects period 0; it did not edit repository
source. Holding v400 fixed, the measured exponent-policy ratio accounts for
about 56.1% of the original ratio excess. It also changes the gradients:
period-2 versus native-EX2 within CuTe measured relative-L2
0.06079/0.06097/0.06200, cosine 0.99862/0.99861/0.99855, and norm ratios near
1.0293 for dQ/dK/dV. These numbers do not declare either policy invalid; they
show why the 1.232657x headline and the approximately 1.10x native-EX2
comparison answer different questions.

The same-process v416/native-EX2 comparison is the closest numerical-policy
match in this receipt and puts native TK 0.535% ahead. The remaining 11.116%
gap to cd57's public fast result is therefore an exponent-approximation-policy
comparison, not evidence of an unresolved exact-schedule deficit. Native EX2
remains a diagnostic CuTe policy rather than its public ABI. v416 is now also
authenticated through the production runner and the saturated 1.2B fixture;
only long-training convergence remains outside this receipt's claim boundary.

## Historical executive result (superseded by the addendum above)

The matched Volt-prod v4 result closes the pending v387 performance gate. At
B16/S4096/Hq32/Hkv8/D64, exact CuTe cd57 measured 3.337696 ms p50 at its public
boundary, v384 measured 5.999264 ms with output clears included, v387 main
measured 6.054560 ms with caller pre-clear excluded, and v387 full measured
6.121408 ms with its three clears included. v387 full is therefore
1.834021986x CuTe latency and 1.020359869x v384 latency; even the narrower v387
main boundary is 1.813993859x CuTe latency. v384 remains the fastest verified
native-TK candidate under the matched public-with-clear comparison.

Copying the CuTe kernel's coarse K128/Q128 topology was not sufficient. v385
reduced the number of static TMA-load sites and modeled load transactions, but
expanded live register fragments, spilled to a 120-byte stack frame, and
retained a single-buffered, CTA-wide-synchronized schedule. It regressed to
12.914496 ms. v386 reused a 64-column post-processing fragment; that removed
the stack and all observed `LDL`/`STL` sites and recovered B16 main latency to
9.999680 ms. This is a real 1.29148x recovery relative to the paired v385 run,
but v386 is still 1.70039x slower than v384.

The exact CuTe source was authenticated at SHA-256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`.
Its fresh public distribution used 10 warmups and 51 rotated samples: p25
3.334368 ms, p50 3.337696 ms, p75 3.343392 ms, p95 3.370080 ms, CV 0.0139417,
minimum 3.327264 ms, and maximum 3.640992 ms. All four measured cells were
finite before and after timing.

v387 is the frozen attempt to close that gap with double-buffered
Q/dO/statistics/P-dS stages and phase-counted named barriers. Its final source,
extension, and CUBIN are authenticated; the kernel reports
REG128/STACK0/LOCAL0/SHARED101600, passes the supplied S128 and B1/S4096
donor-relative correctness checks repeatedly, and returns exact zero gradients
for zero dOut. The Volt-prod result now establishes its performance: the async
rewrite did not beat v384 and remains materially behind exact CuTe.

## Claim boundary and common contract

Unless a row says otherwise, the experiments are isolated causal attention
backward measurements on NVIDIA GB200 (SM100), locally or on Volt-prod, with:

- shape B1 or B16, S4096, Hq32, Hkv8, D64;
- contiguous BHSD inputs;
- E4M3 Q, K, V, and dO with the native operand value scale of 4.0;
- FP32 softmax statistics;
- `softmax_scale = 64**-0.5`; and
- additive dQ/dK/dV outputs.

This is not an end-to-end model-speed result. It excludes forward attention,
projection GEMMs, quantization producers, optimizer work, communication, and
the rest of a training step.

“Main” and “full” are deliberately narrow terms in this receipt:

- native v385/v386 **main** launches into caller-pre-zeroed BF16 output
  tensors;
- native v385/v386 **full** performs three `cudaMemsetAsync` output clears and
  then launches the same kernel; it still does not include backward preprocess
  or input quantization;
- historical donor/v384 measurements cover their FP32-accumulator main entry
  points;
- matched v4 **v384 public** includes three caller-owned output clears followed
  by `main_e4m3_bhsd`;
- matched v4 **v387 main** excludes caller pre-clear, while **v387 full**
  includes three internal `cudaMemsetAsync` operations; and
- CuTe **full** is the authenticated public
  `CompiledGqaBackward.run(reset=True)` boundary, including reset.

Most historical performance fixtures used all-zero tensors. The matched v4
run used the exact-cd57 represented-forward fixture, identical contiguous BHSD
represented-E4M3 operands, and identical contiguous FP32 statistics across its
four cells. GPU clocks were not pinned, and the complete historical series was
not one full ABBA experiment; cross-generation ratios remain engineering
evidence rather than a publication confidence interval.

## Timing ledger

All values are milliseconds. Triples are `min / median / max`. “Unavailable”
means no authenticated number was preserved; it does not mean zero or failed.

### Matched Volt-prod v4 result

Job `6p9jawlv3ntl` ran on one NVIDIA GB200 with 10 warmups and 51 samples per
cell, rotating cell order once per warmup and sample. All cells were finite
both before and after timing.

| Cell | Measured boundary | p50 | Relative latency |
|---|---|---:|---:|
| Exact CuTe cd57 | public `run(reset=True)` | **3.337696** | 1.000000x |
| v384 | public, including clears | **5.999264** | 1.797427x CuTe |
| v387 main | caller pre-clear excluded | **6.054560** | 1.813994x CuTe |
| v387 full | internal clears included | **6.121408** | 1.834022x CuTe; 1.020360x v384 |

The exact CuTe distribution was 3.327264 / **3.337696** / 3.640992, with
p25/p75/p95 of 3.334368/3.343392/3.370080 ms and CV 0.0139417. The recovered
51-sample vectors and all other cell quantiles are authenticated by the v4
JSON identified below.

### Historical ledger

| Implementation | B1 main | B1 full | B16 main | B16 full |
|---|---:|---:|---:|---:|
| Native donor | unavailable | unavailable | 7.679360 / **7.690592** / 7.711136 | unavailable |
| v384 two-WG | unavailable | not implemented/measured | 5.863104 / **5.880832** / 5.903200 | not implemented/measured |
| v385 K128/Q128 | 0.912032 / **0.920928** / 0.929504 | 0.924640 / **0.932064** / 0.938560 | 12.905856 / **12.914496** / 12.924960 | 12.970464 / **12.977120** / 12.987232 |
| v386 halfcols | correctness only; not timed | correctness only; not timed | 9.996256 / **9.999680** / 10.008448 | 10.057984 / **10.060352** / 10.070912 |
| v387 async pipeline | correctness only; not timed | correctness only; not timed | see matched v4 table | see matched v4 table |
| CuTe cd57 | unavailable | unavailable | 3.231840 single captured kernel; audit-only | **3.345824 median**, sample vector unavailable |

Timing protocol details:

- donor/v384 B16 used zero inputs, five warmups, and 31 CUDA-event samples.
  The historical event-harness output is the primary timing. An independent
  Nsight Systems capture containing 36 launches reported kernel medians of
  7.674976 ms and 5.864224 ms respectively; it includes warmup launches and is
  retained as corroboration rather than substituted into the event table.
- v385 B1 and its dedicated B16 timing used zero inputs, five warmups, and 31
  CUDA-event samples.
- v386 B16 was measured in the same process as v385 with zero inputs, seven
  warmups, and 31 samples per boundary. In that paired run v385 main was
  12.914368 ms and v386 main was 9.999680 ms, giving the 1.29148x recovery.
  v386 quartiles were 9.998912/10.000896 ms for main and
  10.059040/10.062464 ms for full.
- the durable CuTe fallback receipt preserves only the 3.345824 ms median and
  finite status; it does not preserve warmup count, repetitions, or the sample
  vector. The 3.231840 ms main value is exactly one kernel duration in an older
  Nsight Systems SQLite capture, not a median.
- the matched Volt-prod v4 result supersedes the pending v387 timing status but
  does not erase the older fallback/audit observations; only within-v4 ratios
  use matched boundaries and a shared rotated protocol.

## Correctness ledger

The B1 correctness tests use B1/S4096/Hq32/Hkv8/D64 and seeded random E4M3
inputs generated as `randn * 0.5`. v385 used seed 385; v386 used seed 386.
Candidate BF16 outputs were compared with the retained donor's FP32
accumulator outputs. These rows therefore test agreement with the donor, not
agreement with an independent BF16 FA4 reference.

| Candidate vs donor | Gradient | Finite | Cosine | Relative L2 | Max abs |
|---|---|---:|---:|---:|---:|
| v385 | dQ | yes | 0.9952292091 | 0.0978407125 | 10.3698120 |
| v385 | dK | yes | 0.9989421167 | 0.0459855875 | 1.7404985 |
| v385 | dV | yes | 0.9999950099 | 0.0031591414 | 1.4331055 |
| v386 | dQ | yes | 0.9957380414 | 0.0924391317 | 9.5411987 |
| v386 | dK | yes | 0.9989783221 | 0.0451945319 | 1.8722744 |
| v386 | dV | yes | 0.9999951204 | 0.0031239763 | 1.5548096 |

The v386-versus-v385 comparison on the same seed was much tighter:

| Gradient | Finite | Cosine | Relative L2 | Max abs | Exact-value fraction |
|---|---:|---:|---:|---:|---:|
| dQ | yes | 0.9999993346 | 0.0011536309 | 0.25 | 0.9801612 |
| dK | yes | 0.9999996298 | 0.0008604874 | 0.25 | 0.9664674 |
| dV | yes | 0.9999996584 | 0.0008264991 | 1.0 | 0.9666624 |

The post-handshake v387 donor-relative checks are:

| Fixture | Gradient | Finite | Cosine | Relative L2 | Max abs |
|---|---|---:|---:|---:|---:|
| S128 handshake | dQ | yes | 0.9989097078 | 0.0466857173 | 0.4711304 |
| S128 handshake | dK | yes | 0.9989772026 | 0.0452178992 | 0.3042588 |
| S128 handshake | dV | yes | 0.9999949000 | 0.0031937751 | 0.2697754 |
| B1/S4096 | dQ | yes | 0.9955745830 | 0.0942145844 | 10.2083740 |
| B1/S4096 | dK | yes | 0.9988616011 | 0.0477056633 | 1.7265600 |
| B1/S4096 | dV | yes | 0.9999949811 | 0.0031682604 | 1.6159700 |

The S128 repeat produced relative-L2 differences of exactly 0 for dQ,
0.00285198 for dK, and 0.00282223 for dV. The B1/S4096 repeat produced
relative-L2 differences of 0, 0.00095823, and 0.00093813 respectively; all
outputs remained finite. These small dK/dV repeat differences are consistent
with additive BF16 GQA publication order. A separate zero-dOut invariant
returned exactly zero for dQ, dK, and dV.

The donor is the comparison reference in this receipt. The v384 correctness
harness source is preserved, but its output was not; no v384 cosine/allclose
claim is reconstructed. The fresh CuTe fallback run was finite, but the
fallback receipt does not preserve operator-error metrics. None of these facts
supports numerical equivalence to BF16 training by itself.

### Quarantined v4 cross-implementation output metrics

The v4 JSON contains raw TK-versus-CuTe cosine, relative-L2, norm-ratio, and
maximum-error fields. They are **NON-SEMANTIC** and are quarantined from the
correctness ledger: the native and CuTe paths do not share the same prelift,
statistics-scaling, and output-scaling ABI. Those raw numbers therefore do not
measure an accuracy regression. The only cross-path numerical claim from v4 is
that every output was finite before and after timing. The donor-relative native
checks above remain the receipt's numerical evidence because they use the
native comparison contract.

## Topology lineage

| Implementation | CTA work and roles | Pipeline/ownership | Output contract |
|---|---|---|---|
| Native donor | K128 per CTA as two K64 consumers; Q64 work per consumer; 2 consumer WGs + 1 producer WG; 384 threads | double-buffered Q/dO and named-barrier/TMA pipeline; persistent per-consumer K/V; 512 TMEM columns | FP32 additive accumulators; caller zeroes |
| v384 | same K64x2 ownership, 2 consumer WGs, 256 threads; producer role folded into WG0 | retains staged pipeline but overlays dead TMEM regions; allocator limited to 256 TMEM columns to permit two-CTA residency | FP32 additive accumulators; caller zeroes |
| v385 | one K128/Q128 CTA; 8 compute warps, 4 reduction/publication warps, 1 MMA-issue warp, 1 loader warp, 1 stats warp, 1 spare warp; 512 threads | K/V loaded once; Q/dO once per query iteration; score/dP TMEM overlay; ordinary global stats loads; single shared stage and repeated CTA-wide barriers | BF16 additive outputs; main requires zero, full clears |
| v386 | v385's K128/Q128 roles and ownership | reuses one 16x64 score/dP fragment for the two column halves; otherwise retains v385's single-buffer/barrier schedule | BF16 additive outputs; main requires zero, full clears |
| v387 | v385/v386 K128/Q128 role split; 512 threads | two score TMEM stages and two shared stages for Q/dO/statistics/P-dS; phase-counted named barriers; zero steady-state CTA-wide barriers in exported metadata | BF16 additive outputs; main requires zero, full clears |
| CuTe cd57 | K128/Q128; 8 compute warps, 4 reducer warps, 1 MMA warp, 1 load warp, 2 spare warps; 512 threads | role-specialized named-mbarrier pipeline, staged Q/dO/statistics, direct TMA dK/dV, FP8 P in TMEM | authenticated public compiled-backward wrapper |

The source-level causal traffic model estimates about 35 bulk input TMA
commands per average v385/v386 K128 CTA: two persistent K/V commands plus two
commands for each of an average 16.5 causal Q tiles. The earlier K64/Q64
native schedule was modeled at roughly 136 input/statistics commands for
comparable work because it used smaller tiles and several small statistics
transactions. These are modeled source executions, not Nsight Compute counts.

## Launch resources

Resource values are exact static metadata from authenticated CUBIN/extension
images plus source/launch audit for dynamic shared memory and TMEM. They are
not achieved occupancy measurements.

| Implementation | Threads | Registers/thread | Stack B/thread | Static shared B | Dynamic shared B | TMEM columns | Residency declaration/intent |
|---|---:|---:|---:|---:|---:|---:|---|
| Native donor | 384 | 168 | 48 | 1,248 | 117,760 | 512 | `launch_bounds(...,1)`; one CTA/SM resource shape |
| v384 | 256 | 126 | 32 | 1,232 | 98,304 | 256 | `launch_bounds(...,2)`; designed for two CTAs/SM |
| v385 | 512 | 128 | 120 | 67,696 | 0 | 512 | `launch_bounds(...,1)` |
| v386 | 512 | 128 | 0 | 67,696 | 0 | 512 | `launch_bounds(...,1)` |
| v387 | 512 | 128 | 0 | 101,600 | 0 | 512 | `launch_bounds(...,1)` |
| CuTe cd57 | 512 | 128 | 0 | 1,024 | 76,800 | 512 | `.minnctapersm 1` |

For CuTe, `cuobjdump` reports the 1,024-byte static allocation; the 76,800-byte
dynamic allocation comes from the authenticated launch/source audit. v384's
two-CTA statement is a launch/resource design bound, not a measured achieved
occupancy percentage.

## Static ISA ledger

The native rows use one normalized parser: a SASS instruction site is a line
beginning with an encoded instruction address inside the selected kernel.
This corrects an earlier audit inconsistency: v385's historical value “6132
instructions” counted every disassembly body line, including encoding lines.
The normalized v385 count is 3,064 instruction sites. For reconciliation, the
raw body-line counts are donor 10,196; v384 5,396; v385 6,132; and v386 3,972.

| Native kernel | SASS instruction sites | `UTCQMMA` | `MUFU.EX2` | `UTMALDG` | `UTMAREDG` | `UTCBAR` | `LDL` | `STL` | `LDSM` | `STSM` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Donor E4M3 main | 5,096 | 28 | 64 | 28 | 9 | 9 | 10 | 3 | 0 | 8 |
| v384 main | 2,696 | 16 | 32 | 12 | 3 | 5 | 10 | 4 | 0 | 4 |
| v385 main | 3,064 | 24 | 64 | 4 | 3 | 7 | 61 | 61 | 0 | 32 |
| v386 main | 1,984 | 24 | 32 | 4 | 3 | 7 | 0 | 0 | 2 | 28 |
| v387 main | 2,880 | 40 | 32 | 12 | 3 | not separately tallied | 0 | 0 | 2 | 28 |

The frozen v387 SASS additionally contains 8 `BAR.SYNC`, 119 `SYNCS`,
14 `WARPSYNC`, and 160 `F2FP` instruction sites. These are static sites, not
dynamic execution counts.

The exact CuTe artifact is available as PTX rather than a normalized native
SASS slice. Its trusted raw static PTX-site counts are:

| CuTe cd57 PTX site | Count |
|---|---:|
| `tcgen05.mma` | 32 |
| `ex2.approx` | 32 |
| bulk tensor load | 6 |
| `cp.reduce.async.bulk.tensor` | 6 |
| `tcgen05.ld` | 10 |
| `setmaxnreg` | 5 |

Static source/PTX/SASS sites are not dynamic executed-instruction counts.
Branches, loop trip counts, predication, and compiler unrolling make raw site
counts incomparable to runtime work, and PTX sites are not directly
comparable with SASS sites.

## Why each adaptation helped or failed

### Donor to v384: a real native win

The donor is completion-correct and already asynchronous, but its three-WG,
384-thread block consumes all 512 TMEM columns and nearly a full 64K register
file (`168 * 384 = 64,512` registers). v384 removes the dedicated producer WG,
folds load/publication work into WG0, overlays dead score/dP/dQ TMEM regions,
and limits each CTA to 256 columns. Its 126-register, 256-thread launch can
admit the intended second CTA. The B16 main result improves by 1.307739x.

This is the strongest verified native result in the series. Its limitation is
that it still inherits the donor's K64 consumer decomposition and smaller
transaction structure instead of CuTe's K128/Q128 role pipeline.

### v384 to v385: fewer loads, slower kernel

v385 deliberately adopts the CuTe control's coarse K128/Q128 ownership. K/V
become persistent across the causal Q loop, and the static TMA-load sites fall
from 12 in v384 to 4. That change addresses real transaction overhead.

It also makes several unfavorable changes at once:

- 512 threads and a full 512-column TMEM allocation return to one CTA/SM;
- full-width score/probability and dP/dS register fragments remain live long
  enough to create a 120-byte stack frame and 61 static `LDL` plus 61 `STL`
  sites;
- Q/dO and the P/dS shared operand are single-buffered;
- the steady-state loop uses repeated `__syncthreads()` barriers; and
- one stats warp performs ordinary global loads rather than participating in
  a fully staged TMA/named-barrier pipeline.

The 12.914496 ms B16 result proves that reducing TMA-site count alone did not
pay for those costs. Without NCU counters, the receipt does not assign an
exact percentage to spills, barriers, occupancy, or exposed memory latency;
the explanation is a source/resource-consistent causal diagnosis, not a
counter-derived decomposition.

### v385 to v386: spill repair works, scheduling gap remains

v386 post-processes one 64-column half at a time and reuses the same 16x64
fragment. Relative to v385 it:

- removes the 120-byte stack frame;
- removes all static `LDL`/`STL` sites;
- halves static `MUFU.EX2` sites from 64 to 32; and
- reduces normalized SASS instruction sites from 3,064 to 1,984.

The paired B16 main timing improves from 12.914368 to 9.999680 ms, a 1.29148x
speedup. The B1 output remains very close to v385, and its donor-relative
metrics improve slightly for dQ/dK.

The result is still 1.70039x slower than v384 because v386 deliberately leaves
the v385 pipeline intact: Q/dO/statistics are not double-buffered, CTA-wide
barriers serialize steady-state stages, and load/compute/publication overlap
does not match CuTe. The fragment repair fixed a real compiler problem; it did
not fix the scheduling architecture.

### v387: frozen correctness/resource result and measured scheduling gap

v387 targets the remaining gap with private two-stage score TMEM, double
buffering for Q/dO/statistics/P-dS, phase-counted mbarriers, and role-specific
load/MMA/reduction loops. The source and binary were frozen after the final
reader-before/signal/writer-after handshakes and stats-warp convergence audit.
They report REG128/STACK0/LOCAL0/SHARED101600, pass the S128 and B1/S4096
checks above repeatedly, and satisfy the exact zero-dOut invariant.

The matched v4 run establishes the speed result: 6.054560 ms p50 for main and
6.121408 ms for full, versus 5.999264 ms for v384 public and 3.337696 ms for
exact CuTe public. The v387 async rewrite therefore remains slightly slower
than v384 and 1.834021986x CuTe latency at the full boundary.

The remaining gap is not a launch-grid mismatch. v387 and CuTe use the same
split-GQA grid of 16,384 CTAs and both have a one-CTA-per-SM resource shape.
The verified source/ISA comparison instead identifies three concrete schedule
differences:

- v387 aliases P and dS storage, which blocks overlap between their pipeline
  lifetimes;
- v387 immediately waits after each dQ TMA store rather than overlapping that
  publication with subsequent work; and
- native uses half-EX2 behavior rather than CuTe's period-2 ALU policy.

These are verified structural differences, not an NCU-derived attribution of
exact stall percentages. v388/v389 explorations, if continued, are
in-progress only and have no result in this receipt.

## Artifact identities

The repository was intentionally dirty during development. Source and binary
hashes below authenticate the exact artifacts used; inclusion here does not
mean the untracked native sources were committed at the receipt base commit.

### CuTe cd57 control

- precomposed source:
  `/tmp/fa4_cute_dsl_fallback_20260829/artifacts/fmha_bwd_d64_gqa_aug19_exact.py`,
  220,876 bytes, SHA-256
  `cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`;
- qutlass commit: `406e86fb2d7df436e94f825bcda8e59b1a7250a6`;
- CUTLASS commit: `b2ca083d2bb96c41d9b3c5a930637c641f6669bf`;
- copied `fmha_bwd.py`: SHA-256
  `953b86a13cc64789052e9d1f8090c9562ac579ee903d275d78cf426780c09eff`;
- generated backward CUBIN: 86,800 bytes, SHA-256
  `95d5e03ec1dfbfc1833f3d54e8c35d0cc44aa6a0345af9bb45ffd7b77ebd4abb`;
- generated backward PTX: 151,469 bytes, SHA-256
  `f1680a2f1583944d3937a85dd211fe4424ad92bffc7b6d48beda57fcd2668fcf`;
- authenticated switches: `TK_DIRECT_TMA_DKDV=True`,
  `TK_FP8_P_STORAGE=tmem`, `TK_DETACHED_FP8_P_TMEM=False`; and
- fallback environment: Python 3.12.3, Torch
  `2.9.0a0+145a3a7bda.nv25.10`, CUTLASS DSL 4.5.2, driver 580.126.09.

The generated files live under
`/tmp/fa4_cute_dsl_fallback_20260829/isa_dump_exact_cd57_20260829/` and have
the `CompiledGqaBackward` basename recorded in `summary.json`.

### Native donor

- `native_gqa_tk_bwd.cu`:
  `ee21f3f1cb64301ee81c7950feac5fb31d264ae75b05caf673afc773a5d6510e`;
- `native_gqa_tk_bwd.cuh`:
  `7530c3d3f811f62203c3cf92caf4b939d0e3f7bfc4021381ef8054b8ff4a5339`;
- `native_gqa_tk_bwd_pipelined.cuh`:
  `f2d9f789d2584133a10ccc9a9c615afb93a50b7e0ca472371223faa4db6c3f47`;
- `Makefile`:
  `ae7413f7deb1de4e437c42277d961c9b78ecf3e58eb52a7d893f6e3923eeac3a`;
- extension: 2,299,864 bytes, SHA-256
  `c2c8afb6c791ed6d4435593637e5dfa9afc960a1c913cd32b9b64482acd227c5`.

### v384

- `.cu`: `1b28ea505cb2917b4db97b0e0899266fa7bab2655690e048b4d0c69ddda63088`;
- `.cuh`: `fa31a7927f272dad69330aac41833430cc5ac848f35b1eec04c2b02ad4493467`;
- extension: 2,150,120 bytes, SHA-256
  `662512bfaa7627620586a1de8249fe45c2675e47a7621cec0506116b31c98737`;
- timing harnesses: `/tmp/test_v384_two_wg.py` and
  `/tmp/time_v384_two_wg_b16.py`; and
- Nsight Systems profile:
  `/tmp/v384_d64_e4m3_b16_isa_profile.nsys-rep`, SHA-256
  `0f7096cf03d8e0f59382c5fe25909cd872e6caa5446fb8eeefc772dbb8d153fd`.

### v385

- `.cu`: `74fde07b11c5b1954c6c12bea02ecd052c2feb63f162dc04be8e3ae01c3a25d0`;
- `.cuh`: `11ee92a96d883c15af41c8a23b552350b594000ffee152da45b38516ecca4d70`;
- `Makefile.v385`:
  `ef3bdc594647c887e2f86eaf9b165ca5561b6d71fda9c34594abd354949c5de7`;
- extension: 2,283,880 bytes, SHA-256
  `0b62572a70b2575a984630d0703c81a0490dda616fb8b2515f030131c5a7bc09`.

### v386

- `.cu`: `3d660517c21c6f17468bb60cb4f7a58a74ba274f50162a5dd981372e36b273c3`;
- `.cuh`: `0af425a83a3d08fc0268b056380fb6a3b50ab6851023b5e85cdea7868e00508e`;
- `Makefile.v386`:
  `229c1816d6a0fafb1c12c9f79dc4191e034a862b6112a68b84a1b27df8668d07`;
- extension: 2,480,032 bytes, SHA-256
  `cb55484a148c528f5a1e90140315d364c345210393365eff71f450398a0f33ff`;
- CUBIN: 1,255,080 bytes, SHA-256
  `c65921078b148868a9fe033cc8b8cb58e8abb8e141d8b2fe3bd776d88fa49001`;
- resource dump: `/tmp/v386_resources.txt`, SHA-256
  `11764772129992f1f27d913a50c5950ba56457e830aaf79ce102f61b12c872ee`.

### v387 frozen artifact

- `.cuh`: 22,489 bytes, SHA-256
  `49986b321f3ede18135bdda4055f10e247ca00cb9cc372ca5e49a20265a26662`;
- `.cu`: 6,849 bytes, SHA-256
  `9fcad6b67f6e2533ad38d96d94e42b57b50a545a14c33879262fe865ba3eb362`;
- `Makefile.v387`: 1,023 bytes, SHA-256
  `b394ed92d54e7f7698a572876d19329b960523f42bfe33fcaff56012ba267734`;
- extension: 2,743,440 bytes, SHA-256
  `386e694d45176d2ad7e9d3083e97830d46c3b8d30889f8c15063f7dc5988f50b`;
- CUBIN: 1,522,216 bytes, SHA-256
  `c4cfc352827c1d88641ce6ab56a72091642b7690925e7bd242e785e9eb04a8d0`;
- resource dump: 1,013 bytes, SHA-256
  `f68fe0a46e35a26d9ef519ecf7ce9ae482e420248f04f292cd4cf910948e8047`;
- selected-kernel SASS: 373,109 bytes, SHA-256
  `babfe03cec514d7d9f0c7d0a4f8646f1c0db632c984182c27d2209b624e55956`;
- full disassembly: 9,000,369 bytes, SHA-256
  `c211c920a3ca3f4c5a48e70f4413ad2fe978298dc12f4c1115adbe060983f835`;
- static resources: REG128, STACK0, LOCAL0, SHARED101600; and
- matched v4 full p50: 6.1214079857 ms; main p50: 6.0545601845 ms.

### Canonical code, timing assets, and Volt audit

- canonical source-branch HEAD at this audit update:
  `abd3f33104ac885434f1d6136ab5100361de51ee`;
- frozen native lineage commit:
  `a0e7b4c9bde0fa89f5dbd6ecabd6d3abc11f9b60`;
- gitless/minimal-asset timing support commit:
  `0bc33c0da9932fc23f05eef39f4438e38930395f`;
- timing harness at that commit: 27,698 bytes, SHA-256
  `ea9ebae07bed1c8720c683524f95772f27de3e045d4f27b5cdb2eed8f3e5b404`;
- historical Volt manifest at that commit: 10,015 bytes, SHA-256
  `463a0e1b22add56ac7f78ab2010fdeb472c567379205db5972f0c304148ca1d2`;
- v4 minimal code asset commit:
  `fd797a2883db3f1c2cce52ac2fe2f5447b13f852`;
- v4 provenance-record commit:
  `3aa17d7a5cebac4415e2fbb565f854c13ccf019e`;
- v4 manifest SHA-256:
  `149db760f3308c75eaf9d85f0041953b9f04e7a30fdf2defc9ec3cc73384d4b3`;
- v4 `VOLT_ASSET_PROVENANCE.json` SHA-256:
  `51d9ff8e657789f5fc42a50408c066a65cb16715a2cea7867f8ce8512b491ee1`;
- recovered v4 timing JSON: 24,353 bytes, SHA-256
  `04f2fad7afe47a76c35b926755b9cb6df34bd091cf0ae24eff3a55b8996d3605`;
- Volt-prod job `q0lae9sorjfd` failed before build or timing because the mounted
  root asset had its `.git` directory stripped;
- `tgox0eziegy4` failed on an invalid `nvidia-smi` inventory query;
- `mjqpcdkzhyfd` succeeded for native-only timing, but exact CuTe was
  unavailable because `tk_fa4/interface.py` was absent from the minimal asset;
  and
- v4 job `6p9jawlv3ntl` succeeded with the SHA256-authenticated asset and exact
  CuTe cd57 available. This is the matched result claimed above.

## Profiling caveats

Nsight Compute hardware-counter collection was unavailable on this node:
`ERR_NVGPUCTRPERM` was returned and `RmProfilingAdminOnly` was enabled. No
driver or module setting was changed. Therefore this receipt makes no claim
about achieved occupancy, warp-stall percentages, tensor-core utilization,
TMA bandwidth, cache hit rates, or dynamic instruction counts.

All occupancy statements are static resource/launch bounds. All ISA counts are
static sites. The transaction estimates are source models. The timing numbers
are the runtime performance evidence; donor-relative checks and finite-output
invariants are the runtime numerical evidence. Quarantined v4 cross-path output
metrics are not correctness evidence.

The machine-readable ledger is `summary.json` in this directory.

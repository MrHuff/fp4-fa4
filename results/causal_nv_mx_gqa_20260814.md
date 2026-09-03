# Causal NV/MX GQA Forward, GB200 and B300

Date: 2026-08-14

## Scope

The retained kernel uses NVFP4 QK, MXFP4 PV, and two adjacent query tiles per
logical job. Both D64 and D128 skip the fully masked stage-0 tail. Causal D128
additionally pipelines exact score loads, quarter maxima, and the represented
denominator through P construction. At S4096 and above it also refills Q0's
retired register halves with Q3 while Q0 is still being packed. GB200 defers
the final scalar denominator conversion until P publication; B300 finalizes it
in place. The policy preserves the existing TMEM allocation and barrier
protocol on both architectures.

Two GQA model templates are covered:

| Template | Shape |
|---|---|
| Llama 3.2 1B-style | B1, S4096, Hq32, Hkv8, D64 |
| Llama 3 8B-style | B1, S4096, Hq32, Hkv8, D128 |

The D128 input uses folded K64 Q/K scales (`--nv-qk-fold-k64-scales both`).
D64 has one K64 chunk and uses the ordinary Q/K scale path.

## Matched A/B

CUDA event medians use 100 ms warmup, 500 ms repetition, and seed 20260814.
The baseline and candidate run in the same process on the same tensors.

| Template | Baseline (ms) | Tail skip (ms) | Improvement | Output delta |
|---|---:|---:|---:|---:|
| D64 | 0.098592 | 0.096288 | 2.34% | exact |
| D128 | 0.094240 | 0.092768 | 1.56% | exact |

For both shapes, the causal leakage prefix and LSE are bitwise equal after
changing the latter half of V.

## Causal D128 Score Pipeline

The original D128 path loaded and processed most score quarters in serial.
The retained schedule uses values that are already known from the CTA and
moves independent work into otherwise exposed latency windows:

1. The scheduler omits the fully masked stage-0 tail tile.
2. Only the diagonal tile needs a lane mask. Earlier tiles are known to be
   fully valid from `key_tile < query_tile`.
3. At S4096 and above, Q0's low register half starts an x16 Q3 load after P
   words 0-1 are packed. Its high half starts the second x16 load after words
   2-3 are packed. S2048 retains the cheaper monolithic x32 load.
4. Q0's transform computes Q1's exact maximum, Q1 computes Q2's maximum, and
   Q2 computes Q3's maximum. Each following quarter therefore begins with its
   scale dependency already resolved.
5. Each pair of packed denominator words is decoded beside its producer. Two
   independent DP4A reductions finish the halves before one final scalar add.

This is register lookahead, not another TMEM score slot. It adds no TMEM
columns, CTA barriers, shared memory, register spills, or tensor-core work.
P still contains exact causal zeros before PV; the CTA-coordinate observation
removes redundant mask decisions but does not defer masking until PV.

The B200 and B300 causal `fast` policies select this schedule automatically
for D128, including the S4096 threshold for progressive Q3 refill. D64 keeps
its original score schedule because applying the D128 interleave to D64
regressed latency.

## Pipeline A/B Across Shapes

These measurements were taken while the GPU was loaded, so the paired ratios
are meaningful and the absolute times should not be mixed with the earlier
idle-device table. Each candidate and baseline ran in one process on identical
inputs with 100 ms warmup and 1500-2000 ms repetition.

| S | Hq/Hkv | D | Tail-skip baseline (ms) | Score pipeline (ms) | Improvement | Output delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 128 | 0.063488 | 0.059680 | 6.00% | exact |
| 4096 | 32/8 | 128 | 0.157696 | 0.149504 | 5.20% | exact |
| 8192 | 32/8 | 128 | 0.495968 | 0.467136 | 5.81% | exact |
| 4096 | 64/8 | 128 | 0.271968 | 0.254528 | 6.41% | exact |

Reversing candidate/baseline order also reproduced the result. Every row had
zero output delta, and the causal leakage prefix and LSE remained bitwise
equal.

## Progressive Q3 Register Refill

The first score pipeline waited until all four Q0 P words were packed, then
issued one x32 Q3 load into Q0's retired registers. The refined schedule
reuses each 16-register half as soon as its two words retire. This starts half
of Q3 earlier without keeping another carrier live or allocating storage.

| S | Hq/Hkv | Monolithic x32 (ms) | Progressive x16 (ms) | Change | Output delta |
|---:|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 0.059424 | 0.059968 | 0.92% slower | exact |
| 4096 | 32/8 | 0.149504 | 0.147488 | 1.35% faster | exact |
| 8192 | 32/8 | 0.467520 | 0.459792 | 1.65% faster | exact |
| 4096 | 64/8 | 0.254528 | 0.251488 | 1.19% faster | exact |

The fixed extra load issue does not amortize at S2048, so the production
policy compiles the old x32 path there. A final gated S2048 A/B measured
0.059744 ms for both builds. All rows retained bitwise-equal causal prefixes
and LSE. Ptxas reports 128 registers, one barrier, 400 static shared-memory
bytes, and no spills for the progressive kernel, unchanged from x32.

## BF16 Comparison

The matched HAO/CuTe DSL BF16 provider was timed by the same benchmark driver.

| Template | NV/MX (ms) | BF16 (ms) | NV/MX speedup | Cosine vs BF16 | Relative L2 |
|---|---:|---:|---:|---:|---:|
| D64 | 0.095840 | 0.120832 | 1.261x | 0.950856 | 0.317503 |
| D128 | 0.092768 | 0.143168 | 1.543x | 0.950383 | 0.316980 |

Small timing differences between the paired A/B and BF16 runs are normal
run-to-run variation; use the paired rows for optimization attribution.

With the currently loaded GPU, the final policy measured 0.147488 ms against
0.210976 ms for the same HAO/CuTe BF16 provider at B1/S4096/Hq32/Hkv8/D128,
or 1.430x. Accuracy is unchanged: cosine 0.950383 and relative L2 0.316979.

## Diagnostic Ceilings

The causal production path was compared with progressively removed P work at
B1/S4096/Hq32/Hkv8. These are diagnostic kernels, not valid attention modes.

| D | Production (ms) | Score-pack ceiling (ms) | Fixed-P floor (ms) |
|---:|---:|---:|---:|
| 64 | 0.163952 | 0.126016 | 0.106816 |
| 128 | 0.156256 | 0.119360 | 0.098592 |

For D128, replacing only the represented denominator measured 0.137216 ms,
while retaining row-max and packing measured 0.120832 ms. The denominator was
therefore about 11.9% of production latency, affine/scale/pack work about
10.5%, and max scheduling only about 1.5 microseconds. This motivated hiding
score loads and exact reductions rather than adding another mask shortcut.

## Profile Read

Nsight Compute shows that synchronization is not the remaining bottleneck.
The final D128 pipeline changes the profile as follows:

| Metric | Tail-skip baseline | Score pipeline | Progressive refill |
|---|---:|---:|---:|
| Kernel duration (ns) | 154080 | 148864 | 147456 |
| Executed instructions | 38653899 | 37880569 | 37936065 |
| Tensor instructions | 154928 | 154928 | 154928 |
| Issue utilization | 36.85% | 37.52% | 37.84% |
| Tensor-core active | 26.28% | 26.67% | 26.88% |
| Wait stall | 0.70 | 0.57 | 0.58 |
| Long-scoreboard stall | 4.01 | 4.07 | 4.04 |
| Barrier stall | 0.04 | 0.04 | 0.04 |

The base score schedule removes about 2.0% of executed instructions. Splitting
Q3 adds only 0.15% back, while improving issue utilization and trimming the
long-scoreboard ratio enough to reduce duration another 0.95% in NCU. Tensor
work, barriers, eligible warps (0.63), shared memory, and TMEM are unchanged.
The remaining limiter is still the score-load-to-P-pack dependency chain;
DRAM utilization is low. Further work must hide or shorten that dependency
rather than add synchronization.

## Rejected Pipeline Variants

Split-half Q3 lookahead, keeping a Q3 carrier live from the beginning of Q0,
and chaining the second denominator DP4A into the first all lost to the final
schedule. Applying the D128 max interleave to D64 also regressed latency. Those
paths were removed; the build now accepts only the original fallback or the
validated full-Q3/two-independent-DP4A schedule.

A four-step x8 progressive refill was also tested. It exposed each Q3 slice
earlier, but four TMEM load issues cost more than the extra overlap saved:
x8 measured 0.147776-0.148480 ms versus 0.147456-0.147520 ms for two x16 loads
in reversed A/B order. It was exact and leakage-clean, but was removed.

## Deferred Denominator Finalization

The represented MXFP4 denominator originally converted each quarter's exact
integer E2M1 sum to FP32 and applied its E8M0 scale before publishing the
corresponding P page. That scalar work does not affect the packed P payload or
the scale page consumed by PV. The retained path therefore carries one 32-bit
word per quarter containing the exact integer sum and stored E8M0 byte,
publishes both P K64 pages first, and reconstructs the same FP32 denominator
while the tensor-core issuer consumes P.

The reconstructed quarter values and their accumulation order are
bit-identical to the previous fast kernel. The new path also adds no TMEM or
shared-memory allocation, barrier, register spill, or stack frame. Ptxas
still reports 128 registers, one barrier, and 400 bytes of static shared
memory.

| S | Hq/Hkv | Previous fast (ms) | Deferred finalize (ms) | Improvement | Output delta |
|---:|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 0.059712 | 0.057664 | 3.43% | exact |
| 4096 | 32/8 | 0.147488 | 0.145408 | 1.41% | exact |
| 8192 | 32/8 | 0.466944 | 0.454656 | 2.63% | exact |
| 4096 | 64/8 | 0.251360 | 0.247808 | 1.41% | exact |

The B200 causal D128 `fast` policy now selects this handoff automatically.

Several adjacent scheduling changes were tested against this new baseline and
removed. Delaying the upper Q2 score load until Q0 packing and moving Q3 under
Q1 was exact but regressed 0.145408 ms to 0.147616 ms. Publishing P scales
directly from the denominator carrier was also exact but measured
0.145984--0.146528 ms because it extended the carrier dependency into scale
publication; the existing shared-memory byte traffic is already hidden.

The saturated S4096/H64 shape retained 12 K/V stages: 11 stages measured
0.255008 ms and 13 measured 0.248896 ms versus 0.248224 ms for 12. Head-major
task order measured 0.338224 ms, and a 147-CTA grid measured 0.256512 ms,
versus about 0.248 ms for head-interleaved order on all 152 SMs. At
S2048/H32, an exactly balanced 128-CTA grid tied the 152-CTA default at
0.057728 ms. No shape-specific scheduling override was retained.

## B300 Transfer and Tuning

The causal score pipeline transfers to SM103, but deferred denominator
finalization does not. The discovery run used causal B1/S4096/Hq32/Hkv8/D128
with three 1000-iteration windows per variant. Every binary used 128
registers, one barrier, no spills, and the same 512 TMEM columns.

| B300 variant | Median (ms) | Change vs legacy |
|---|---:|---:|
| Legacy KV6 | 0.092954 | reference |
| Tail skip only | 0.092736 | 0.24% faster |
| Score pipeline, finalize in place, KV6 | 0.087072 | 6.33% lower latency |
| Score pipeline, deferred mode 2, KV6 | 0.089674 | 3.00% slower than in place |
| Score pipeline, finalize in place, KV12 | **0.086213** | 7.25% lower than legacy |

The mode-2 and in-place outputs are bit-identical. B300 simply has a
different best instruction schedule: moving the scalar reconstruction under
P consumption adds pressure to the path that was just exposed, while its
faster scalar tail can finish the conversion cheaply before publication.
Mode 1 also lost at 0.088220 ms. Consequently, deferred finalization remains
GB200-only; B300 retains the rest of the causal ownership pipeline.

Task order 1 measured 0.096823--0.099524 ms, versus 0.087072--0.089674 ms for
order 0. A 128-CTA grid measured 0.093539 ms at S4096, also slower than all
148 SMs. The follow-up depth sweep selected KV12. Quarter-3 native `exp2`
measured 0.086479 ms versus 0.086124 ms for the all-ALU route. A 144-CTA grid
reached 0.085980 ms, only 0.17% below the 148-CTA result, so it was not
promoted without another independent confirmation. S16384 similarly put KV8
only 0.018% ahead of KV12; the production policy keeps the simpler KV12
default.

Matched HAO/CuTe DSL BF16 comparisons use 100 ms warmup, 1000 ms repetition,
the same seeded tensors, folded K64 Q/K scales, and causal GQA. The custom
timing in this table comes from the same `do_bench` process as its BF16 row.

| S | Hq/Hkv | Selected schedule | NV/MX (ms) | HAO BF16 (ms) | Speedup | Cosine | Relative L2 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2048 | 32/8 | KV12, grid 128 | 0.037824 | 0.050112 | 1.325x | 0.951009 | 0.313470 |
| 4096 | 32/8 | KV12, grid 144 | 0.091136 | 0.123808 | 1.359x | 0.950178 | 0.317117 |
| 8192 | 32/8 | KV12, grid 148 | 0.279328 | 0.420768 | 1.506x | 0.949441 | 0.319596 |
| 16384 | 32/8 | KV8, grid 148 | 0.980480 | 1.573920 | 1.605x | 0.948713 | 0.321439 |
| 4096 | 64/8 | KV12, grid 148 | 0.154592 | 0.222240 | 1.438x | 0.950101 | 0.317557 |
| 8192 | 64/8 | KV12, grid 148 | 0.517088 | 0.818208 | 1.582x | 0.950271 | 0.316604 |

All outputs are finite. For every selected binary, changing the latter half
of V leaves the earlier output prefix and LSE bitwise unchanged. The B200
default was rebuilt after the architecture split and reproduced 0.145536 ms
at S4096/Hq32/Hkv8, deferred mode 2, with the same leakage guarantee.

The B300 evidence comes from Volt jobs `e61vydxupba8` (discovery),
`p78qmpnm51pi` (depth sweep), and `xz398n4qedei` (cross-shape/BF16
validation). The first and third completed all kernel runs but ended in their
final JSON aggregators; the downloaded per-run artifacts are complete. The
checked-in job specifications correct both reporting-only parser errors.

## Rejected Diagonal Experiment

A second experiment directly published zeros for the six fully masked 32x32
blocks in the surviving stage-1 diagonal P tile. It was not equivalent to the
current quantized-denominator path: D128 slowed from 0.092704 to 0.108096 ms
and odd query-tile output changed materially. The experiment was removed.

## Rejected Causal Approximation Shortcut

The existing mask writes `-inf` into masked score lanes. Mode-23's affine
score approximation already maps that sentinel to a non-positive value, and
the E2M1 packer emits zero. An exact shortcut was tested inside the MXFP4 P
transform: a fully masked 32-value block wrote zero payload and zero scale
without running max, approximation, pack, or denominator decode. The normal
P publication and barrier sequence was preserved.

Applying the shortcut to every quarter added 112 static SASS instructions and
16 branches, slowing D128 materially. Restricting it to Q3 reduced the control
cost but still lost on both model templates under a matched 100/500 ms A/B:

| Template | Tail skip (ms) | Q3 zero shortcut (ms) | Regression | Output delta |
|---|---:|---:|---:|---:|
| D64 | 0.163840 | 0.170336 | 3.96% | exact |
| D128 | 0.157280 | 0.165440 | 5.19% | exact |

These absolute times were measured while the GPU was loaded and should not be
compared with the earlier idle-device table; the paired ratios are the useful
result. The shortcut cannot shorten Q3 publication because one reader warp
still owns the valid diagonal prefix and executes the full transform. The
additional branch also disrupts the compiler's schedule around the P chain.
The experiment was removed.

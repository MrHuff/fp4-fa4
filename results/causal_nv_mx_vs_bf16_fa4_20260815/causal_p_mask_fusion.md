# Causal P-mask fusion experiment

Date: 2026-08-15

This experiment tested whether the diagonal causal mask could be moved from
FP32 score registers into the MXFP4 approximation/packing path. Measurements
use GB200, `B1/S4096/Hq32/Hkv8/D128`, NVFP4 QK, MXFP4 PV, the retained causal
policy, a 100 ms warmup, and a 500 ms timing window.

| Variant | Time (ms) | Cosine vs BF16 | Relative L2 | Leakage-safe |
|---|---:|---:|---:|---:|
| Retained `-inf` score mask | 0.145408 | 0.950341 | 0.316311 | Yes |
| No-mask timing ceiling | 0.135776 | 0.700160 | 0.716904 | No |
| Fully masked warp-quarter skip | 0.170528 | 0.950341 | 0.316311 | Yes |
| Packed mask, all quarters | 0.158016 | 0.928756 | 0.386095 | Yes |
| Packed mask, Q3 only | 0.147808 | 0.950309 | 0.316470 | Yes |
| Q3 mask precomputed once | 0.147488 | 0.950309 | 0.316470 | Yes |

The no-mask ceiling is 6.6% below production, so diagonal masking has a real
cost. The obvious fusions do not capture it:

- Skipping fully masked warp-quarter blocks does not shorten CTA latency. The
  bottom reader warp still processes all four quarters and gates P
  publication. The extra control flow added 12 SASS branches and regressed
  latency by 17.3%.
- Applying the triangle after E2M1 packing restores causal correctness, but
  each packed word must be masked before both represented-denominator decode
  and TMEM publication. This inserts a new dependency into the critical
  score-to-P chain.
- Q3-only packing preserves production accuracy. Precomputing all four masks
  also restores the baseline branch count (289), but it remains 1.4% slower.
  The remaining four dependent word masks, not branch overhead or occupancy,
  account for the loss.

All functional variants retained 128 registers, one barrier, zero stack, and
zero spills. No packed-mask implementation was promoted. The retained `-inf`
path is better scheduled: mask materialization happens before the
approximation, and the existing E2M1 conversion naturally emits zeros.
Capturing the 6.6% ceiling requires a different diagonal row/warp ownership
scheme or a hardware pack operation that accepts a validity mask; moving the
same work later in the current ownership model is counterproductive.

## Conservative pair-boundary mask

A second experiment keeps Q0 exact, but optionally gives both scores in an
adjacent pair the validity of the pair's second score. This rounds the legal
causal prefix down by one score on affected even rows. It can drop a legal
self score, but it never admits a future score. Keeping Q0 exact is essential:
pair-rounding Q0 removes token 0's only legal score and creates non-finite LSE
rows.

The quarter bits are controlled by
`HAO_CAUSAL_PAIR_QUARTER_MASK`. The best quality/speed trade-off is mask 12,
which pair-rounds Q2 and Q3 while leaving Q0 and Q1 exact.

| Pair-rounded quarters | Mask | Time (ms) | Cosine vs BF16 | Relative L2 | Finite / leakage-safe |
|---|---:|---:|---:|---:|---|
| None, exact default | 0 | 0.145408 | 0.950341 | 0.316311 | Yes / Yes |
| Q1 | 2 | 0.145888 | 0.949823 | 0.318089 | Yes / Yes |
| Q2 | 4 | 0.145696 | 0.950145 | 0.316985 | Yes / Yes |
| Q1 + Q2 | 6 | 0.145408 | 0.949627 | 0.318760 | Yes / Yes |
| Q1 + Q3 | 10 | 0.145344 | 0.949622 | 0.318743 | Yes / Yes |
| **Q2 + Q3** | **12** | **0.144416** | **0.949943** | **0.317641** | **Yes / Yes** |
| Q1 + Q2 + Q3 | 14 | 0.144096 | 0.949426 | 0.319412 | Yes / Yes |
| Q0 + Q1 + Q2 + Q3 | 15 | 0.143904 | 0.910706 | 0.420819 | No / Yes |

Mask 14 is about 0.2% faster than mask 12, but its extra Q1 rounding costs
more accuracy. Mask 12 is therefore retained as an explicit opt-in candidate;
the exact mask remains the default.

The two reader stages were also separated. Pair-rounding stage 0 alone
regressed to 0.147456 ms. Stage 1 alone preserved more accuracy, but repeated
1000 ms timing windows measured 0.145408, 0.145408, and 0.145440 ms, which is
baseline speed. The repeatable gain requires Q2/Q3 pair-rounding in both
stages, so no stage selector was retained.

### Repeated A/B

| Seed | Exact (ms) | Q2+Q3 pair (ms) | Exact cosine | Pair cosine | Exact rel. L2 | Pair rel. L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260815 | 0.145408 | 0.144416 | 0.950341 | 0.949943 | 0.316311 | 0.317641 |
| 20260816 | 0.145536 | 0.144512 | 0.951104 | 0.950676 | 0.313806 | 0.315229 |
| 20260817 | 0.145440 | 0.143936 | 0.949759 | 0.949431 | 0.318355 | 0.319467 |

Every run had finite output, bitwise-identical leakage prefixes, and
bitwise-identical LSE in the leakage test.

### Cross-shape check

| Shape | Exact (ms) | Q2+Q3 pair (ms) | Gain | HAO BF16 (ms) | FP4 speedup | Exact / pair cosine | Exact / pair rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2048, Hq32/Hkv8, D128 | 0.057696 | 0.057632 | 0.1% | 0.086944 | 1.509x | 0.951413 / 0.950994 | 0.313747 / 0.315185 |
| S4096, Hq32/Hkv8, D128 | 0.145408 | 0.144416 | 0.7% | 0.211968 | 1.468x | 0.950341 / 0.949943 | 0.316311 / 0.317641 |
| S4096, Hq64/Hkv8, D128 | 0.247808 | 0.245344 | 1.0% | 0.342048 | 1.394x | 0.949989 / 0.949564 | 0.317118 / 0.318527 |
| S8192, Hq32/Hkv8, D128 | 0.454752 | 0.451616 | 0.7% | 0.628768 | 1.392x | 0.950187 / 0.949849 | 0.316471 / 0.317607 |

The candidate preserves 128 registers, one barrier, and zero spills. Relative
to the exact binary, SASS drops 64 `ISETP` instructions and six `FSEL`
instructions while retaining the same 256 `BRA` and 13 `BSSY` instructions.
This explains why the gain is small but consistent: it removes integer mask
work without changing occupancy or synchronization.

## Further exact-mask attempts

- Removing Q3's range gate produced identical timing and effectively identical
  SASS; the compiler had already scheduled the uniform condition well.
- Splitting fully invalid warps from partial warps added divergent control flow
  and regressed to 0.156672--0.160832 ms.
- Interleaving four Q3 mask fragments through Q2's transform/max path regressed
  to 0.347616 ms and changed output (cosine 0.827631). Four reconvergence regions
  replaced one compact mask region and disturbed the Q3 register lifetime. The
  implementation was removed completely.
- SM100a does not accept a paired `min.f32x2` instruction, so a two-score causal
  ramp cannot replace the scalar comparisons on this path.

The remaining 5--6% gap to the no-mask ceiling is therefore not ordinary
branch cleanup. Recovering it exactly needs a different score ownership/layout
that can represent a triangular valid prefix without scalar FP32 score
materialization on the softmax reader's critical path.

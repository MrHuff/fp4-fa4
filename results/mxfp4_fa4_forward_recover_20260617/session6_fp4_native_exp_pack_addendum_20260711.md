# Addendum: FP4-Native Exp2 And Direct E2M1 Packing

Execute this only after completing the baseline mixed FA4 ALU-emulation cadence and packing matrix in `session6_stage2_ex2_alu_pack_20260711.md`. Do not abandon or invalidate that comparison.

## Motivation

The destination is nonnegative E2M1, whose finite magnitudes are:

```text
code:   0    1    2    3    4    5    6    7
value:  0   0.5   1   1.5   2    3    4    6
```

RNE decision thresholds in linear and log2 space are:

```text
linear: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
log2:  -2.0, -0.4150374993, 0.3219280949, 0.8073549221,
        1.3219280949, 1.8073549221, 2.3219280949
```

Therefore a full FP32 `exp2` followed by generic E2M1 conversion is not intrinsically required to produce the payload nibble. The row-sum recurrence is the reason an exponent value is still needed on the exact-semantics route.

Prior work rejected seven-comparison ladders. Do not repeat that implementation. The candidates below reuse FA4 range reduction and require at most one or two fractional comparisons for a given integer exponent region.

## Candidate F1: FA4 Cubic Plus Direct Nibble

Extend the paired FA4 ALU helper so each input returns:

- reconstructed approximate FP32 `2^z` for the existing row-sum accumulation;
- the positive E2M1 nibble rounded from that same approximate value.

The FA4 range reduction already provides:

- `n = floor(z)` through the magic-add representation;
- `p ~= 2^(z - n)` in `[1, 2)` from the cubic.

Derive the nibble from `n` and `p` without a generic `cvt.e2m1`:

```text
n < -2: code 0
n = -2: code 0 at p == 1.0 (RNE tie), otherwise code 1
n = -1: code 1 + (p >= 1.5)
n =  0: code 2 + (p > 1.25) + (p >= 1.75)
n =  1: code 4 + (p > 1.25) + (p >= 1.75)
n =  2: code 6 + (p > 1.25)
n >  2: code 7
```

The strict/non-strict comparisons encode ties-to-even. Verify this against hardware `cvt.rn.satfinite.e2m1x2.f32`, including every midpoint and adjacent FP32 values.

For a mixed native/emulated group:

- emulated pairs supply their byte directly;
- native pairs may still use one `cvt.e2m1x2` each;
- combine four bytes into the existing 32-bit payload order.

This should remove E2M1 F2FP instructions in proportion to the emulation ratio while preserving the existing row-sum concept. Measure whether integer/predicate ALU cost is lower than F2FP pressure.

## Candidate F2: Direct Log-Threshold Nibble Plus ALU Row Sum

For the payload code, classify the original log2 input `z` directly. After `n=floor(z)` and `f=z-n`, only these fractional thresholds are needed:

```text
log2(1.25) = 0.3219280949
log2(1.50) = 0.5849625007
log2(1.75) = 0.8073549221
```

Use the same exponent-region mapping as F1 with at most two comparisons. This payload classification targets the mathematically exact E2M1 code without evaluating exp2 for packing.

Continue using the degree-3 FA4 cubic for the FP32 row-sum value. Compare payload bytes against native Stage2 and F1. If degree 3 wins, also test degree 2 using FA4's existing `POLY_EX2[2]` coefficients; it saves one packed FMA and may be sufficient when the payload code no longer depends on polynomial accuracy.

## Candidate F3: Fully FP4-Native Quantized Denominator

This is an explicitly approximate/math-changing speed-of-light candidate:

1. Produce E2M1 codes directly from `z` using F2 classification.
2. Do not compute cubic or native exp2.
3. Accumulate the dequantized E2M1 values, adjusted by the block P scale, for the denominator/row sum.

Reuse the prior convergent nibble/LUT or DP4A infrastructure where helpful, but ensure this route actually removes EX2 and F2FP; the earlier qsum experiment retained production exp2+cvt and therefore did not test this speed path.

Label F3 clearly as changed denominator semantics. Report output/LSE degradation separately. It must remain default-off even if fast unless numerical acceptance is explicitly justified.

## Required Micro-Validation

Before full-kernel timing, build a focused helper probe over:

- `z` from `-127` through `log2(6)`;
- all seven E2M1 midpoints;
- neighboring FP32 values around each midpoint;
- representative causal `-inf` values.

For F1/F2 report:

- nibble mismatch rate versus native `ex2.approx + cvt.e2m1`;
- mismatch locations and whether they are only threshold-adjacent;
- exp max absolute/relative error for F1/F2 row-sum values;
- SASS instruction counts and pipe classes per pair.

## Kernel Matrix

Test F1 and F2 first with the best cadence from the existing mixed-emulation task. Test at most one neighboring cadence if the direct-pack ALU work changes the optimum. Test F3 only after F1/F2 results are recorded.

For any candidate, measure:

- Stage2 sparse scale-to-exp/pack interval;
- ptxas and spills;
- repeated h4/s2048, h8/s1024, h8/s4096;
- h16/s4096 for a winner;
- candidate versus Stage2 and BF16 numerical error;
- NCU pipe redistribution for a repeatable winner.

Also test combination with retained `pchainc` only after an FP4-native candidate wins independently.

## Acceptance

- F1/F2 may be retained only if deterministic, spill-free, and numerically within a defensible envelope.
- F3 may be retained only as an explicit speed/accuracy experiment, never global default in this task.
- Remove rejected selectors and behavior.
- Add results to the current EX2/pack report or a clearly linked addendum section, then append the ledger and restore default/off.

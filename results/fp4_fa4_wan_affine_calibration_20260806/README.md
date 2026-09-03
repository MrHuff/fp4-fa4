# Wan Affine E2M1 Boundary Calibration

This directory tests whether the affine constants in the direct FP4
probability path should be calibrated per model or layer. The operation under
test is

\[
\widehat q(x)=Q_{\mathrm{E2M1}}(\max(0,Ax+B)),
\]

where the target is the E2M1 code of the scaled value `2^x`. The experiment
does not replace softmax. It changes the boundaries at which the already
normalized log-domain probability is rounded to an E2M1 payload.

The teacher-forced sweep runs one FP4 attention layer at a time and leaves all
other self-attention layers in BF16. Seven compile-time alternatives around
the `A=1.60, B=0.95` base are compared on an identical BF16 trajectory. Every
candidate has the same instruction count, 128-register allocation, one
barrier, and 400 bytes of static shared memory, so the selected boundaries do
not alter kernel scheduling or latency.

## Wan2.1-14B

Twenty-four of 36 non-guard layers improve relative L2 by more than `1e-4` in
isolation. Selecting every local winner overfits. The retained regularized
route uses `(1.575, 1.05)` in layers
`1,3,6,8-12,15-17,22-27,30-31,35` and the base pair elsewhere. Across four
same-seed runs per prompt, it changes mean cosine/relative L2 from
`0.932272/0.365616` to `0.932798/0.362731` on the calibration prompt and from
`0.914132/0.405767` to `0.917118/0.398699` on the held-out prompt. Guard
layers 33, 34, 38, and 39 retain the sampled QK-range path.

## Wan2.1-1.3B

The 14B layer map does not transfer. A fresh H12 sweep also produces many
isolated winners, but broad 20-, 13-, and 8-layer routes regress on the
two-prompt end-to-end average. The retained route changes only layer 0 to
`(1.625, 0.95)` and layer 11 to `(1.575, 1.05)`. On the calibration four-step
run it changes cosine/relative L2 from `0.969336/0.251446` to
`0.970305/0.249594`; on the held-out run it changes
`0.965941/0.265302` to `0.967855/0.256256`. The two-prompt mean therefore
moves from `0.967638/0.258374` to `0.969080/0.252925`. Guard layers 27--29
remain unchanged.

Run `python build_summary.py` to regenerate the 14B repeat summary and LaTeX
table. Run `python ../fp4_fa4_wan_20260805/build_downstream_rows.py` to
regenerate the final 1.3B/14B downstream rows and
`wan_calibrated_downstream_summary.json`. The raw unrestricted and regularized
routes remain in this directory as failed controls; they show why local
teacher-forced wins require end-to-end validation.

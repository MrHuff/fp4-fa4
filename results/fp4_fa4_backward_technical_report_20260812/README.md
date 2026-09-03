# Hardware-aware FP4 FlashAttention-4 backward report

This is the standalone backward companion to
../fp4_fa4_technical_report_20260728. It deliberately keeps the forward
report read-only while reusing its section cadence and visual language.

The report distinguishes:

- the learned Q/K projection, \(Q=XW_Q^\mathsf{T}\) and
  \(K=XW_K^\mathsf{T}\);
- the attention score contraction, \(S=\alpha QK^\mathsf{T}\);
- the retained adaptive FP4+FP8 backward;
- the faster but less accurate mixed-dP route;
- the experimental pure-MXFP4 precision ladder.

Its primary implementation boundary is ../../tk_fa4/lowp_fa4_bwd/.
Measurements and claims are backed by the dated Markdown and JSON artifacts
listed in Appendix C.

Build the PDF with:

    make

The report is intentionally separate for now. A later umbrella paper can
share notation, hardware background, and an end-to-end forward/backward
evaluation, while retaining independent forward and backward method sections.

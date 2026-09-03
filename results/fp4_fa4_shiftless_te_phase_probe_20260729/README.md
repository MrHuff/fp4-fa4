# Transformer Engine P-scale phase probe

This focused seed-0 B1/S4096/H24 probe tests the canonical Transformer
Engine NVFP4 mantissa phase (`2688 / 2048 = 1.3125`) and nearby phases on
the fast shiftless P path.

The best sampled phase was `1.28125`, with relative L2 `0.27992132` versus
`0.27994469` at identity. The canonical `1.3125` phase reached
`0.27996153`, which is worse than identity. The tiny local gain did not
justify changing the P default.

See the broader
[`../fp4_fa4_nvfp4_global_scale_calibration_20260729/README.md`](../fp4_fa4_nvfp4_global_scale_calibration_20260729/README.md)
for the retained Q/K/V implementation and cross-shape validation.

# Reciprocal-448 P-scale phase probe

This seed-0 B1/S4096/H24 negative control tests the safe E4M3 mantissa
phase corresponding to a literal reciprocal-448 adjustment:
`512 / 448 = 8 / 7`.

Identity produced relative L2 `0.27994469`; `8 / 7` produced
`0.27999306`. The reciprocal phase therefore does not improve the fast
shiftless P path.

See the broader
[`../fp4_fa4_nvfp4_global_scale_calibration_20260729/README.md`](../fp4_fa4_nvfp4_global_scale_calibration_20260729/README.md)
for the retained Q/K/V implementation and cross-shape validation.

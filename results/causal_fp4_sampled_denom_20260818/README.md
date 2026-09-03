# Causal NV/MX sampled-denominator sweep

## Scope

This sweep keeps the causal `fast` NVFP4-QK/MXFP4-PV score-to-P transform
fixed and changes only the denominator estimator. Measurements use a GB200,
`B1/S4096/Hq32/Hkv8`, seed 0, 100 ms warmup, and a 500 ms timing window.
D64 uses interleaved K/V quarters; D128 uses MSE-selected folded-K64 Q/K
scales. Accuracy is against Torch BF16 attention on the same inputs.

`qN` samples N represented E2M1 values from complete packed words. `s8`
samples four two-value pairs spread across all four packed words. `a8` and
`a4` sample the pre-pack affine approximation and are included only to
reproduce the older four-pair style ceiling. The `+max` variants add the
block's guaranteed represented maximum exactly and estimate only residual
mass.

## Results

| D | Denominator | Time (ms) | Change vs full | Cosine | Relative L2 |
|---:|---|---:|---:|---:|---:|
| 64 | full represented 32 | 0.148128 | - | 0.956750 | 0.293523 |
| 64 | q16 | 0.160288 | +8.2% | 0.854443 | 0.599059 |
| 64 | q8 | **0.144384** | -2.5% | 0.725182 | 0.973374 |
| 64 | q16 + max | 0.180608 | +21.9% | 0.935715 | 0.354988 |
| 64 | q8 + max | 0.150144 | +1.4% | 0.925780 | 0.378492 |
| 64 | s8 | 0.145408 | -1.8% | 0.715080 | 1.041346 |
| 64 | s8 + max | 0.166912 | +12.7% | 0.927066 | 0.374902 |
| 64 | a8 | 0.137568 | -7.1% | 0.017040 | 43.652245 |
| 128 | full represented 32 | 0.145440 | - | 0.950080 | 0.316973 |
| 128 | q24 | 0.190464 | +31.0% | 0.878402 | 0.500037 |
| 128 | q16 | 0.161728 | +11.2% | 0.854071 | 0.546846 |
| 128 | q8 | 0.145408 | -0.0% | 0.776189 | 0.689038 |
| 128 | q16 + max | 0.178912 | +23.0% | 0.918851 | 0.394880 |
| 128 | q8 + max | 0.151584 | +4.2% | 0.872312 | 0.489956 |
| 128 | s8 | 0.146432 | +0.7% | 0.766122 | 0.780722 |
| 128 | s8 + max | 0.163840 | +12.7% | 0.895560 | 0.451391 |
| 128 | a8 | 0.141312 | -2.8% | 0.056957 | 1.892432 |
| 128 | a4 | **0.135456** | -6.9% | 0.005745 | 52.266602 |

All outputs were finite. The D64 q8 per-row rescaling oracle restores cosine
to 0.959789, confirming that its numerator remains sound and that sampling
introduces row-dependent normalization noise. A global scalar, per-head
scalar, query-position fit, and per-query-tile/head scalar recover only
0.725182, 0.821433, 0.707351, and 0.854383 cosine respectively; there is no
cheap shared calibration for that noise.

The only quantized sampled variant with a D64 timing gain, q8, was also run on
seeds 0--3. D64 cosine averaged 0.673075 with a 0.475633--0.770481 range and
mean relative L2 of 1.199877. D128 cosine averaged 0.722898 with a
0.592848--0.793163 range and mean relative L2 of 0.819838. Sampling is not
stable across random inputs.

## Interpretation

The full represented denominator is already interleaved with P packing and,
at D128, supports deferred finalization and progressive Q3 score reuse.
Sampling removes a few packed decode operations but breaks that useful
schedule. Consequently, q8 saves only 2.5% at D64 and nothing at D128. Its
variance is far too high. Adding the known maximum reduces variance but adds
more work than the exact four-word sum.

No sampled denominator is promoted. The default remains `fast` with all four
packed words, or all 32 represented P values. The diagnostic override knobs
remain available for reproducibility.

# Compact dQ publication for SM100 causal GQA backward

Date: 2026-08-14

Device: GPU 0

Geometry unless noted: B=1, Hq=32, Hkv=8, causal, warm-cache CuTe
event timing. D64 represents the Llama-3.2-1B attention geometry and D128 the
Llama-3-8B geometry.

## Result

The retained optimization shortens the final dQ path in two steps:

1. `--compact-dq-acc` converts each FP32 TMEM fragment to BF16 on the reducer
   warps and TMA-reduces through a BF16 workspace accumulator.
2. `--direct-compact-dq` applies the softmax scale before that conversion and
   TMA-reduces directly into the caller's zero-initialized BF16 dQ tensor. The
   fused dK/dV reduction remains, but the later dQ conversion sweep is removed.

Writing `alpha = 1 / sqrt(D)` and `G_i` for one CTA's FP32 dQ fragment, the
publication paths are

```text
baseline:  A_fp32 = sum_i G_i;                 dQ = BF16(alpha A_fp32)
compact:   A_bf16 = reduce_i BF16(G_i);        dQ = BF16(alpha FP32(A_bf16))
direct:    dQ_bf16 = reduce_i BF16(alpha G_i)
```

The direct path therefore changes the rounding order and requires a zeroed dQ
destination, but removes both the FP32 dQ workspace and the separate dQ read,
scale, convert, and write pass.

## Llama attention geometries

The aggressive FP8 policies are degree-1 selective packed-ALU EX2 with period
3 for D64 and period 2 for D128. Exact BF16 is the tuned split-GQA control.

| S | D64 BF16 (us) | D64 FP8 direct (us) | Speedup | D128 BF16 (us) | D128 FP8 direct (us) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 78.911 | 68.713 | 1.148x | 108.757 | 77.652 | 1.401x |
| 2048 | 165.998 | 151.890 | 1.093x | 245.752 | 164.394 | 1.495x |
| 4096 | 440.934 | 409.832 | 1.076x | 651.366 | 419.391 | 1.553x |
| 8192 | 1351.064 | 1296.512 | 1.042x | 2026.470 | 1258.906 | 1.610x |

Both S4096 controls and candidates were rerun after correcting exact workspace
sizing. The preceding D64/D128 direct measurements were 408.612/419.731 us,
showing that the retained result is stable within normal run variance. D64 is
now close to its tuned BF16 compute ceiling, while D128 exposes enough
low-precision tensor work and enough removed dQ traffic to produce a much
larger gain.

Against a compact-BF16 publication control at S4096, the direct FP8 path is
1.068x faster for D64 (408.612 versus 436.254 us) and 1.427x faster for D128
(419.731 versus 598.916 us). Direct publication is not automatically useful
for BF16: D128 BF16 direct measured 606.033 us and regressed from the compact
BF16 control.

## Accuracy rungs

Small-shape checks use S512, Hq=4, Hkv=1 against the FP32 PyTorch reference.

| Policy | dQ rel-L2 / cosine | dK rel-L2 / cosine | dV rel-L2 / cosine |
| --- | ---: | ---: | ---: |
| D64 degree-1 / period-3 direct | 0.03459 / 0.999530 | 0.02774 / 0.999808 | 0.21790 / 0.981643 |
| D128 degree-1 / period-2 direct | 0.04383 / 0.999426 | 0.03652 / 0.999775 | 0.14941 / 0.989805 |
| D128 degree-2 / period-2 direct | 0.01107 / 0.999939 | 0.01085 / 0.999943 | 0.14284 / 0.989830 |

The D128 degree-2 policy is the near-lossless EX2 rung and measures 427.384 us
at S4096, still 1.525x faster than exact BF16. The degree-1 policies remain
convergence-gated; the dominant small-check error is still the existing FP8
operand/P path rather than compact dQ publication.

## Profile attribution

Nsight Compute profiles at D128/S4096 compare the compact path before and after
direct publication. The main kernel itself is slightly longer after absorbing
the scale, but the complete event becomes about 18 us shorter because the dQ
conversion work disappears.

| Metric | Compact workspace | Direct publication |
| --- | ---: | ---: |
| Main-kernel duration | 363.14 us | 365.18 us |
| Issue slots busy | 39.02% | 42.28% |
| SM busy | 39.15% | 42.28% |
| Executed instructions | 96,645,249 | 105,367,397 |
| Warp cycles / issued instruction | 7.54 | 6.96 |
| Registers / thread | 128 | 128 |
| Spill requests | 0 | 0 |
| Achieved occupancy | 21.24% | 21.24% |

Profile artifacts:

- `/tmp/d128_fp8_compact_dq_d1p2_gpu0_20260814.ncu-rep`
- `/tmp/d128_fp8_direct_compact_dq_d1p2_gpu0_20260814.ncu-rep`

## Forward plus backward context

The read-only forward benchmark reports 1.261x at D64/S4096 and 1.543x at
D128/S4096 for the projection-native NVFP4/MXFP4 forward route. Summing the
separately measured forward and backward attention components gives an
indicative 1.111x D64 attention-core gain and 1.551x D128 attention-core gain.
These sums are not full transformer-block or model measurements: projection,
RoPE, MLP, optimizer, communication, and launch integration still need to be
measured in one graph.

## Rejected experiments

- Publishing the final dQ tile before dK/dV did not move D64 wall time and
  slightly regressed S8192.
- Replacing two 128x32 dQ TMA publications with one 128x64 publication was
  correct but regressed S4096 by about 5.4%.
- More dQ barrier stages are not indicated: the retained kernel remains at 128
  registers with zero spills, and the useful gain came from deleting traffic.

## Dispatch and integration contract

The new paths remain opt-in until a model convergence gate is complete:

```bash
# D64 aggressive, S >= 1024
--dtype fp8 --compact-dq-acc --direct-compact-dq \
  --exp2-alu-degree 1 --exp2-alu-period 3

# D128 safer rung
--dtype fp8 --compact-dq-acc --direct-compact-dq \
  --exp2-alu-degree 2 --exp2-alu-period 2

# D128 speed rung
--dtype fp8 --compact-dq-acc --direct-compact-dq \
  --exp2-alu-degree 1 --exp2-alu-period 2
```

The direct route requires dQ to be zero before launch because the kernel uses
TMA reduction into the destination. The reference harness enforces this. A
production dispatcher must preserve the same contract or fuse the clear into
an upstream allocation/epilogue.

The next ceiling experiment is not another standalone attention barrier edit.
It is a single GQA model-boundary benchmark that combines projection-native
Q/K/V quantization plus RoPE, the retained causal forward path, this backward
path, and projection backward. That will expose the real 1.2B and 8B end-to-end
gain and whether the dQ destination clear can be eliminated by ownership in the
projection-backward reduction.

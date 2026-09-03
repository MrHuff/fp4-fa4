# Projection-N topology and causal full-chain evaluation

Date: 2026-08-16

This pass checkpoints the retained causal GQA backward, tests the proposed
projection-N multicast/query-owner topology, and reconnects the newer causal
NVFP4-QK/MXFP4-PV forward to the complete D128 GQA activation-gradient chain.
The forward implementation was consumed through read-only compiled artifacts;
no source under `tk_fa4/fp4_fa4_fwd` was modified.

The retained checkpoint is commit `c805474` (`Optimize causal GQA
low-precision backward`) on `tk-fa4-sm100-rewrite`, pushed to `origin` before
the topology experiment.

## Result

For the Llama-3-8B-like attention geometry B1/S4096/Hq32/Hkv8/D128/K4096,
the regular materialized-gradient route remains the production choice.  The
new causal forward reduces the full low-precision layer component sum to
**800.480 us**.  This is:

- **2.027x** faster than the currently implemented BF16 materialized chain
  (1622.880 us), and
- **1.544x** faster than a favorable BF16 lower bound (1236.032 us) that
  replaces the 553.280 us materialized inverse-RoPE/QKV-dgrad path with a
  separately measured 166.432 us pre-materialized BF16 GEMM.

The second number is the defensible performance claim.  It deliberately gives
BF16 credit for a fused inverse-RoPE/materialization implementation that does
not exist in the current harness.

## Projection-N multicast admission

The prototype groups independent two-CTA tensor-core pairs into a 2/4/8/16
CTA cluster. CTA 0 multicasts the shared M tile by even/odd parity, while each
pair loads a private N256 weight tile and retains pair-local TMEM ownership.

The first unrestricted run exposed a stage-reuse race: CTA 0 could overwrite
an A stage after pair 0 released it while another pair was still consuming
the same multicast stage.  The repaired version uses pair-counted cluster
barriers for the tile and scale rings.  All four cluster sizes then matched
the retained projection bit-for-bit in 20 consecutive full-size launches.

Projection shape: M4096/K6144/N4096, NVFP4, one GB200 GPU.

| Topology | Median (us) | Speed vs retained | 20-run maximum abs error |
|---|---:|---:|---:|
| Retained two-CTA | 52.256 | 1.000x | -- |
| Cluster 2 | 50.496 | **1.035x** | 0 |
| Cluster 4 | 69.184 | 0.755x | 0 |
| Cluster 8 | 67.360 | 0.776x | 0 |
| Cluster 16 | 67.104 | 0.779x | 0 |

Thus the multicast/query-owner mechanics are admissible, but the actual
multi-pair projection is 22--24.5% slower at the target K6144 shape.  It is
not integrated into the production backward.

There is also no hidden win from perfect projection overlap under the current
owner schedule.  The owner backward is 472.224 us versus 369.504 us for the
regular route, while its entire prepacked projection is only 52.000 us.  Even
if that projection were hidden perfectly, the owner event lower bound remains
roughly 16--21 us slower than the 512.512 us regular materialized event.  A future
owner topology must shorten owner-side reduction/quantization as well as
overlap projection MMA.

## Causal forward replacement

Identical projection-produced Q/K/V operands were used for the old and new
read-only artifacts.  Timings below are from a full-clock GB200 paired run.

| Forward | Median (us) | Output cosine vs BF16 | Relative L2 | LSE relative L2 |
|---|---:|---:|---:|---:|
| Old NVFP4-QK/FP8-PV exact-scale | 176.960 | 0.999476 | 0.032378 | 0.0000166 |
| New NVFP4-QK/MXFP4-PV aggressive | **92.192** | 0.991048 | 0.135022 | 0.0007120 |
| CuTe BF16 causal GQA | 184.160 | -- | -- | -- |

The new forward is 1.919x faster than the old low-precision artifact and
1.993x faster than CuTe BF16.  It is not a free replacement numerically.  When
each attention output is passed through the same materialized NVFP4 output
projection, cosine to the true-BF16 projected output is 0.982297 for the new
route and 0.990382 for the old route.  The new route should remain the
aggressive mode until convergence tests validate that trade.

## Full activation-gradient chain

All entries are same-process medians from the new causal full-boundary run,
except the explicitly marked BF16 GEMM-only lower-bound probe.

| Boundary | Low precision (us) | BF16 (us) | Speedup |
|---|---:|---:|---:|
| QKV projection (BF16 omits RoPE) | 122.912 | 167.136 | 1.360x |
| Causal attention forward | 92.416 | 184.160 | 1.993x |
| Output projection, including lowp pack | 72.640 | 112.096 | 1.543x |
| dO projection | 71.040 | 110.496 | 1.555x |
| Attention backward with clear | 369.504 | 495.712 | 1.342x |
| QKV dgrad projection | 89.440 | 553.280 materialized / 166.432 GEMM-only | -- |
| dO + backward + QKV dgrad event | 512.512 | -- | -- |

The low-precision component sum is

```text
122.912 + 92.416 + 72.640 + 512.512 = 800.480 us.
```

Substituting the old exact-scale forward gives 885.024 us, so the newer
causal forward improves the already-low-precision full chain by 1.106x.  If a
future attention epilogue directly publishes the output-projection NVFP4
operand, the measured prepacked-output ceiling is 769.696 us.  Against the
favorable 1236.032 us BF16 lower bound, that ceiling is 1.606x.

This component sum includes QKV projection, attention forward, output
projection, output-projection dgrad, attention backward, inverse RoPE, and QKV
projection dgrad.  It does not include MLP, collectives, optimizer work, or
projection weight gradients, so it is an attention-layer result rather than a
whole-model MFU claim.

## Gradient quality against true BF16

| Quantity | Cosine | Relative L2 |
|---|---:|---:|
| Forward output | 0.991418 | 0.132791 |
| dQ | 0.996052 | 0.088845 |
| dK | 0.995562 | 0.094180 |
| dV | 0.999293 | 0.037794 |
| Materialized NVFP4 QKV dgrad projection | 0.990335 | 0.139202 |
| Row-scaled FP8 QKV dgrad projection | 0.998615 | 0.052664 |

All measured values are finite.  The attention gradients remain strong; the
largest end-to-end accuracy pressure is now the aggressive forward plus the
NVFP4 projection boundaries.  Short checkpointed convergence runs should
therefore compare at least the old exact-scale forward, the new aggressive
forward, and the row-scaled FP8 QKV-dgrad option.

## Reproduction

```bash
make -C tk_fa4/lowp_fa4_bwd \
  -f Makefile.projection_n_multicast -j2

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. \
python3 tk_fa4/lowp_fa4_bwd/profile_projection_n_multicast.py \
  --rows 4096 --reduction 6144 --output-width 4096 \
  --clusters 2 4 8 16 --stress-runs 20 --warmups 5 --samples 11

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
python3 tk_fa4/lowp_fa4_bwd/profile_causal_forward_pair.py \
  --warmups 5 --samples 21

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:flash-attention \
python3 tk_fa4/lowp_fa4_bwd/profile_gqa_d128_chain.py \
  --sequence 4096 --q-heads 32 --kv-heads 8 --hidden 4096 \
  --forward-extension \
    /tmp/_C_tk_gb200_causal_s4096_h32_d128.cpython-312-aarch64-linux-gnu.so \
  --forward-module _C_tk_gb200_causal_s4096_h32_d128 \
  --true-bf16-reference --owner-only --direct-workspace-stats \
  --exp2-period 0 --reuse-quantized-p --owner-fuse-kv \
  --diagnose-dv --diagnose-projection-formats --full-layer-boundaries \
  --warmups 3 --samples 7
```

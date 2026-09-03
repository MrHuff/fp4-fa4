# Exact-denominator pipeline probes

Date: 2026-07-30

Branch: `tk-fa4-sm100-rewrite`

Starting commit: `48166de`

GPU: NVIDIA GB200

## Scope

This study targets the NVFP4 QK + NVFP4 PV `fast-corrected` forward
kernel at B=1, S=4096, H=24, D=128. The kernel computes the exact
denominator from the represented E2M1 P payload in a correction
warpgroup while the two softmax warpgroups continue the main pipeline.

The benchmark command used:

```bash
PYTHONPATH=/workspace/codebases/flash-attention-fp4:$PYTHONPATH \
python3 hao_direct_fp4pv_benchmark.py \
  --extension <candidate.so> \
  --extension-module <candidate> \
  --qk-format nvfp4 \
  --pv-format nvfp4 \
  --tk-only \
  --summary-only \
  --global-anchor-kv \
  --global-anchor-samples 32 \
  --warmup-ms 100 \
  --rep-ms 500
```

## Anchored baseline

| Metric | Result |
| --- | ---: |
| Wall time | 0.112640 ms |
| Cosine vs BF16 | 0.963246 |
| Relative L2 vs BF16 | 0.293019 |
| RMSE vs BF16 | 0.007519 |
| Registers | 128 |
| Barriers | 1 |
| Spill loads/stores | 0 / 0 |

The clean post-experiment rebuild reproduced every value above.

## Profile attribution

An exact NCU replay measured 190.720 us. Replay time is not directly
comparable to the wall-time benchmark above.

| Counter | Exact path |
| --- | ---: |
| Tensor-pipe utilization | 19.36% |
| Issue active | 53.60% |
| Active warps | 3.71 |
| Eligible warps | 0.67 |
| Barrier stall | 0.42% |
| Long-scoreboard stall | 58.03% |
| Wait stall | 10.42% |
| Dynamic instructions | 56.292 M |

The dominant problem is the serial data-latency chain, not a large
full-CTA barrier stall. The correction path must wait for represented P,
load payload and scales from TMEM, reconstruct the exact row sum, and
publish it before final normalization can complete.

Static SASS attribution relative to the approximate path found that the
exact path adds 48 `IDP.4A`, 46 `PRMT`, and 16 TMEM wait instructions.
The approximate transform itself contains 128 `FFMA2`, 64 `MUFU.EX2`,
128 E2M1 conversion instructions, and 144 `FMNMX3` instructions relative
to score-pack.

## Experiments

All timing deltas use 0.112640 ms as the baseline.

| Candidate | Time (ms) | Delta | Numerics | Decision |
| --- | ---: | ---: | --- | --- |
| Baseline exact correction WG | 0.112640 | 0.00% | Reference | Keep |
| Sum Q0/Q1 under next snapshot | 0.112640 | 0.00% | Identical | No gain |
| Sum all stage 0 under next snapshot | 0.112640 | 0.00% | Identical | No gain |
| Mirror P scales in SMEM | 0.113568 | +0.82% | Identical | Reject |
| Two independent DP4A sums | 0.113120 | +0.43% | Identical | Reject |
| Read Q0/Q1 at first-P publication | 0.122880 | +9.09% | Incorrect | Reject |
| Producer computes all denominator work | 0.114240 | +1.42% | Identical | Reject |
| Producer Q0/Q1, correction Q2/Q3 | 0.116736 | +3.64% | Identical | Reject |
| Producer Q2/Q3, correction Q0/Q1 | 0.113216 | +0.51% | Identical | Reject |
| Softmax/correction registers 176/112 | 0.115008 | +2.10% | Identical | Reject |
| Softmax/correction registers 192/80 | 0.114688 | +1.82% | Identical | Reject |
| Scale encoder mode 2 | 0.112640 | 0.00% | Identical | No gain |

The early-half candidate is invalid because first-P publication does not
make that half an independently stable snapshot for the correction
consumer throughout subsequent PV and tail ownership.

## Ceilings

| Diagnostic | Wall time | Valid output |
| --- | ---: | --- |
| Exact score-pack ceiling | 0.100640 ms | No |
| Exact full path | 0.112640 ms | Yes |

The score-pack diagnostic removes the P transform but retains the exact
correction structure. Its output is intentionally invalid because it
does not publish usable NVFP4 scales. It establishes two useful bounds:

1. P transform and packing account for about 12.0 us at this shape.
2. Even eliminating that work entirely leaves about 100.6 us of
   structural QK/PV, publication, readback, and output work.

Therefore a valid sub-0.100 ms kernel cannot come from another local
exp2 or packing micro-optimization alone. It must also shorten the
score/P publication-to-PV path or remove correction readback from the
serial normalization dependency.

## Conclusions

The current 184/96 softmax/correction register split is a local optimum.
Both redistribution directions lose despite zero spills.

Moving denominator work into the P producer also loses. It saves
correction work but places that work directly on the P publication
critical path, delaying PV by more than the consumer-side saving.

SMEM scale mirroring does not remove the required lifetime ordering and
adds shared-memory traffic, so it is slower than loading the compact
scale word with the TMEM snapshot.

The next high-value experiment must change representation or ownership,
not instruction ordering: retain a denominator contribution as part of
P production without delaying first-P publication, then combine it
outside the producer's PV-critical window. Without a new handoff that
satisfies both conditions, the measured baseline should remain selected.

All rejected implementation scaffolding was removed after measurement.

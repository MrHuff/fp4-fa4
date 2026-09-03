# Llama 1.2B D64 backward accuracy recovery

## Scope

This experiment isolates the S4096, Hq32/Hkv8, D64 low-precision attention
backward from the known causal-forward approximation error.  Q, K, V, and dO
are held at the same decoded E4M3 values for the CuTe backward and the BF16
reference.  A second run replaces the deployed forward output/LSE statistics
with exact statistics from those decoded operands.

## Bugs and retained changes

1. The full-model gradient stitch decoded the fixed x4 E4M3 publication scale
   for dV but not dQ/dK.  The common x4 decode is now folded into the existing
   gradient-scale multiply.
2. The old S4096 E4M3 dS lift of 256 saturates model-distributed gradients.
   The retained lift is 16.
3. Degree-2 packed-ALU exp2 at period 2 preserves the throughput-oriented
   schedule while matching the native-exp2 accuracy result.
4. The deployed mode-23 shiftless forward reconstructs mean probability mass
   of about 1.572.  A route-specific 0.632 backward correction is folded into
   the existing gradient handoff.  It is disabled for forward policies that
   do not advertise mode-23, shiftless softmax, and quantized denominator.

## Backward-only accuracy

The exact-statistics comparison removes forward output/LSE error.

| Configuration | dQ cosine | dK cosine | dV cosine | dQ norm | dK norm | dV norm |
|---|---:|---:|---:|---:|---:|---:|
| Old: degree 1, period 2, dS lift 256 | 0.8569 | 0.8460 | 0.9986 | 0.6517 | 0.6341 | 0.9718 |
| Retained: degree 2, period 2, dS lift 16 | 0.9970 | 0.9968 | 0.9996 | 0.9909 | 0.9895 | 0.9997 |

The isolated median attention-backward times were 489.4 microseconds for the
old configuration and 484.4 microseconds for the retained configuration.
These separate runs establish no measurable performance regression; the small
difference should be treated as clock noise.

## Deployed-forward statistics

Across two independent seeds, reconstructed probability mass was 1.5717 and
1.5725.  The inverse-mean corrections were 0.63625 and 0.63595, while the
least-squares gradient gains clustered between 0.629 and 0.634.  The retained
common correction is 0.632.

| Gradient | Cosine before/after correction | Norm before | Norm after | Relative L2 after |
|---|---:|---:|---:|---:|
| dQ | 0.9880 | 1.5658 | 0.9962 | 0.1544 |
| dK | 0.9889 | 1.5605 | 0.9929 | 0.1485 |
| dV | 0.9969 | 1.5846 | 1.0082 | 0.0801 |

The scalar changes magnitude but not direction.  Rowwise probability-mass
variation accounts for most of the remaining difference.

## Full-model effect

The six-round alternating 16-layer run kept low-precision step time near 78
milliseconds.  At matched round 5, low-precision loss improved from 7.5312 in
the prior configuration to 5.3125.  BF16 reached 0.5586, so the short-run
convergence gap is reduced but not resolved.  The remaining dominant source
is the forward attention probability/denominator approximation.

Artifacts:

- `llama12b_backward_isolation_baseline_timed_s4096_20260817.json`
- `llama12b_backward_isolation_degree2_p2_ds16_s4096_20260817.json`
- `llama12b_backward_isolation_retained_mass_s4096_20260817.json`
- `llama12b_backward_isolation_retained_mass_seed20260818_s4096.json`
- `llama12b_e2e_interleaved_backward_accuracy_fix_s4096_20260817.json`

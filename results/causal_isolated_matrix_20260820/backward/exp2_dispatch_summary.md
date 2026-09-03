# Causal D64 backward selective-EX2 dispatch (2026-08-20)

## Outcome

The retained causal D64 low-precision backward now resolves EX2 policy by an
explicit measured-shape table. The verified shapes use degree 1 / period 2;
unmeasured shapes retain native degree 2 / period 0. An explicit integer period
always overrides automatic dispatch, so `period=0` remains the native control.

Automatic degree-1/period-2 shapes:

- S4096, Hq/Hkv=16/4
- S4096, Hq/Hkv=32/8
- S4096, Hq/Hkv=64/16
- S8192, Hq/Hkv=32/8

D128 behavior is unchanged.

## Isolated backward evidence

All timings include every required destination/workspace clear. The BF16 and
low-precision routes consume the same represented E4M3 Q/K/V/dO state.

| S | Hq/Hkv | policy | BF16 (us) | lowp (us) | speedup | cosine | norm ratio |
|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 4096 | 32/8 | native d2/p0 | 347.808 | 345.824 | 1.006x | 0.999669 | 0.999493 |
| 4096 | 32/8 | selective d1/p2 | 354.208 | 324.736 | 1.091x | 0.998417 | 0.969500 |
| 8192 | 32/8 | native d2/p0 | 881.184 | 982.240 | 0.897x | 0.999675 | 0.998628 |
| 8192 | 32/8 | selective d1/p2 | 876.704 | 867.936 | 1.010x | 0.998497 | 0.970841 |
| 4096 | 16/4 | native d2/p0 | 237.696 | 234.496 | 1.014x | 0.999677 | 0.999875 |
| 4096 | 16/4 | selective d1/p2 | 242.848 | 226.240 | 1.073x | 0.998493 | 0.972684 |
| 4096 | 64/16 | native d2/p0 | 579.616 | 585.536 | 0.990x | 0.999676 | 0.999062 |
| 4096 | 64/16 | selective d1/p2 | 577.568 | 532.800 | 1.084x | 0.998532 | 0.971080 |

The high-sample S8192 selective result used 13 warmups and 101 samples. Three
accuracy seeds produced aggregate cosine 0.998494--0.998512. Period 3 is a
rejected control on the current direct-compact-dQ/direct-TMA schedule: at
S8192 it measured 1285.536 us against 884.672 us BF16 (0.688x).

Primary artifacts:

- `exp2_guard_s4096.json`
- `exp2_validation_s8192.json`
- `exp2_headcount_guard_s4096.json`
- `exp2_screen_s8192.json` (includes the rejected period-3 route)
- `exp2_dispatch_plan_v2.json` (final measured-shape dispatch manifest)

## Current-contract training gates

The S4096 16-layer 1.2B routes used the current causal forward artifacts,
represented per-block NVFP4 Q/K operands, MX split-V backward publication, and
the backward extension with SHA256
`aeed2603d40290b815218cc77142ddacda0c734384429f26c0d4a6a200fbe884`.

The 24-update smoke gate completed with all steps finite. Final validation loss
was 8.471453 BF16, 8.469826 MX-PV, and 8.470898 FP8-PV.

The automatic-policy gate then completed 177/177 updates, crossing the obsolete
period-2 failure point preserved in historical rollouts. All three current
routes remained finite. Final validation loss was 7.522984 BF16, 7.508274 MX-PV,
and 7.512901 FP8-PV. Median step speedup was 1.2682x for MX-PV and 1.2677x for
FP8-PV in that run. This validates the current degree-1/period-2 path; it does
not rehabilitate the obsolete degree-2/period-2 contract.

That training artifact records policy version v1. Its effective S4096 32/8
settings are identical to final v2; v2 only narrows dispatch to an explicit
measured-shape table and adds the subsequently measured 16/4 and 64/16 S4096
entries.

Training artifacts:

- `../training/c4_unique_24_d4q01_split_v_exp2_d1_p2.json`
- `../training/c4_unique_177_d4q01_split_v_auto_exp2.json`

## Limits and remaining work

- The automatic table deliberately does not extrapolate to unmeasured sequence
  and head-count combinations; those remain native or can be forced explicitly.
- The selective gradients have a repeatable approximately 0.97 norm ratio versus
  the isolated BF16 control. Current 24- and 177-update loss gates are clean, but
  the requested 2K rerun remains necessary for the final convergence claim.
- S8192 Hq/Hkv=16/4 and 64/16, sequences between 4096 and 8192, sequences above
  8192, and D128 selective EX2 are unresolved.

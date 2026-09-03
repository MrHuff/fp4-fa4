# TK BF16 Backward 2CTA dK/dV Candidate Result

Date: 2026-07-09

Scope: BF16 backward only. No forward or MXFP4 code was changed.

Shape: `B=1, S=2048, H=2, Dqk=192, Dv=128`, causal, BSHD.

## Change

Added a new opt-in TK route:

`b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_internal`

and a preallocated-output graph-friendly route:

`b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_out_internal`

The default finite fallback `b300_mha_bwd_hot_cute16_candidate_internal` was not changed. The new route uses the existing `seq2048_exact` 2-CTA dK/dV kernel and keeps the current clustered dQ route.

Changed paths:

- `tk_fa4/b300_bwd_cute16_candidate.cuh`
- `tk_fa4/tk_fa4.cu`
- `results/tk_bf16_bwd_2cta_dkdv_candidate_s2048h2_20260709.md`

## Validation

Reference: CuTe DSL BF16 backward from local `flash-attention/flash_attn/cute/interface.py`.

All compared outputs were finite:

| Path | dq | dk | dv |
| --- | --- | --- | --- |
| CuTe DSL | finite | finite | finite |
| TK fallback | finite | finite | finite |
| TK new 2CTA dK/dV route | finite | finite | finite |

Errors vs CuTe DSL:

| Path | Tensor | rel L2 | max abs |
| --- | --- | ---: | ---: |
| TK fallback | dq | 1.031396 | 3.337106 |
| TK fallback | dk | 0.002855 | 0.006218 |
| TK fallback | dv | 0.001707 | 0.014886 |
| TK new 2CTA | dq | 1.031411 | 3.337106 |
| TK new 2CTA | dk | 0.002855 | 0.006218 |
| TK new 2CTA | dv | 0.001707 | 0.014886 |

The new route preserves the same finite/correctness behavior as the fallback for this smoke.

## Timing

Short event timing, 2 warmup / 8 measured:

| Path | Median | Min |
| --- | ---: | ---: |
| CuTe DSL BF16 backward, event timing | 0.2032 ms | 0.1945 ms |
| TK fallback, event timing | 0.4235 ms | 0.4220 ms |
| TK new 2CTA, event timing | 0.4672 ms | 0.4639 ms |
| TK fallback, CUDA graph replay | 0.4183 ms | 0.4160 ms |
| TK new 2CTA, CUDA graph replay | 0.4614 ms | 0.4597 ms |

Supervisor smoke, same shape/script family:

| Path | Time / component | Notes |
| --- | ---: | --- |
| CuTe DSL BF16 backward | about 0.174 ms | same S=2048/H=2 smoke |
| TK new 2CTA dK/dV route | total about 0.47-0.49 ms | finite, not a win |
| TK new 2CTA dK/dV route | dK/dV about 455-465 us | after warmup |
| TK new 2CTA dK/dV route | dQ about 249-256 us | existing clustered dQ path |
| TK fallback dK/dV | about 410 us | still faster than new 2CTA dK/dV |

Supervisor errors vs CuTe:

| Path | Tensor | rel L2 |
| --- | --- | ---: |
| TK new 2CTA | dq | about 1.03 |
| TK new 2CTA | dk | about 0.00286 |
| TK new 2CTA | dv | about 0.00168 |

Steady split timing:

| Path | preprocess | dK/dV | dQ stream total | dQ kernel | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| TK fallback | 11.55-14.34 us | 420.16-420.86 us | 254.08-257.50 us | 240.70-242.02 us | 441.89-445.98 us |
| TK new 2CTA | 9.28-12.32 us | 461.09-462.82 us | 254.59-255.78 us | 241.76-242.37 us | 481.70-485.28 us |

Conclusion: this 2CTA ownership scaffold is slower because its dK/dV kernel is slower than the current fallback dK/dV critical path. The dQ path is unchanged and still overlaps.

## Resource Use

Build: `timeout 900s make -C tk_fa4 -B _C$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")`

Relevant ptxas/cuobjdump resource data:

| Kernel | Registers | Stack | Shared | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| current fallback dK/dV `main_kernel_causal_fullseq_dkdv_only<config<...,1>, float>` | 255 | 552 B | 85016 B | 1596 B | 3988 B |
| new 2CTA dK/dV `main_kernel_causal_seq2048_exact_dkdv_only<seq2048_exact_config<...>>` | 255 | 544 B | 85016 B | 908 B | 1552 B |
| current clustered dQ | 168 | 16 B | 190532 B | 0 B | 0 B |
| preprocess | 47 | 0 B | 0 B | 0 B | 0 B |

The new dK/dV route reduces spill traffic but still hits 255 registers and remains slower. More CTAs and 2CTA launch structure alone do not reproduce CuTe's 0.12 ms path.
The supervisor smoke agrees: static spills improved vs the old fallback dK/dV kernel, but `REG:255` remains and the route is slower than fallback.

## Next Patch

Do not promote the new 2CTA route to default. Keep it as an opt-in scaffold for comparison.

The next implementation should either fix the candidate2 non-finite fast path or stop reusing per-warp register-accumulator dK/dV kernels. The seq2048 exact route is not enough.

If using candidate2 as the basis, focus on its causal diagonal / patch ownership:

- candidate2 is the closest local route to CuTe structurally: 16-warp roles, 2CTA launch, and TMEM-style accumulators.
- it is fast, but current `dk`/`dv` are not viable because `dk` becomes non-finite.
- do not promote candidate2 until finite `dq/dk/dv` and graph timing beat fallback.

If starting a fresh dK/dV-only kernel behind the same `_2cta_dkdv` route, use:

- 512 threads per CTA and CuTe-like warp roles: reduce 0-3, compute 4-11, MMA 12, load 13, relay 14, empty 15.
- `tile_m=128`, `tile_n=128`, `cluster_shape=(2,1)`.
- accumulator storage moved out of per-thread register tiles. The immediate target is dK/dV accumulation in TMEM/tcgen05 if TK exposes the needed primitives; otherwise use a staged chunked accumulator prototype that never holds `dk0`, `dk1`, `dk2`, and `dv` full tiles live in the same warp at once.
- first acceptance gate: dK/dV kernel resource use below 255 registers with no spills, then graph time below the current 0.418 ms fallback while preserving finite `dq/dk/dv`.

This run shows that ownership/CTA count is not the main blocker. Register/TMEM scheduling is the blocker.

## Follow-Up Measurement After Candidate2 Probes

After restoring all failed candidate2 local probes, I reran a compact
one-process BF16 comparison with preallocated TK outputs:

| Path | Event median | Correctness |
| --- | ---: | --- |
| CuTe BF16 | 0.190272 ms | finite |
| TK fallback | 0.432736 ms | finite |
| TK 2CTA experimental | 0.470592 ms | finite, slower than fallback |
| candidate2 restored | 0.226240 ms | invalid: `dk` non-finite |

Errors vs CuTe in that run:

| Path | dq rel L2 | dk rel L2 | dv rel L2 |
| --- | ---: | ---: | ---: |
| TK fallback | 1.02687 | 0.0028605 | 0.0016762 |
| TK 2CTA experimental | 1.02691 | 0.0028605 | 0.0016762 |
| candidate2 restored | 1.12615 | n/a, `dk` non-finite | 0.822619 |

Fresh split timing after warmup:

| Path | preprocess | dK/dV | dQ stream total | dQ kernel | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| TK fallback | 16.99-23.74 us | 415.42-423.01 us | 265.15-278.18 us | 248.19-260.67 us | 452.99-476.19 us |
| TK 2CTA experimental | 16.06-17.66 us | 463.07-464.13 us | 264.42-266.02 us | 246.59-249.98 us | 495.68-498.08 us |

The 2CTA route remains opt-in only. It is finite but still loses the critical
dK/dV path by about `40-50 us` against fallback.

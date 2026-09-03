# Bounded-FMA NVFP4 scale encoder

This experiment tests whether the finite E4M3 scale guard can be folded into
the scale approximation instead of issuing explicit floating-point min/max
instructions.

## Construction

For the valid normal E4M3 log2-scale interval

```text
lo = -6
hi = 8.75
range = 14.75
```

the experimental encoder computes

```text
t = sat((x - lo) / range)
x_bounded = lo + range * t
```

where `sat` maps to PTX `fma.rn.sat.f32`. The magic-bias carrier is formed
directly from `t`:

```text
carrier = fma(t, 118, MAGIC + 8)
```

This is a bounded degree-one fit. It is exactly the identity inside the
target interval before floating-point rounding and constant outside it.
Compile-time scale-encode mode 3 selects the experiment.

## Generated code

Both S4096/H24 builds use 128 registers, one barrier, 400 bytes of static
shared memory, and zero spills.

| SASS instruction | Existing mode 2 | Bounded-FMA mode 3 | Delta |
|---|---:|---:|---:|
| `FMNMX` | 224 | 208 | -16 |
| `FFMA` | 166 | 174 | +8 |

Eight scale events each replace two `FMNMX` instructions with one
`FFMA.SAT`, reducing the static floating-point instruction count by eight.

## Timing

Measurements use one GB200, B1/S4096/H24/D128, NVFP4 QK and NVFP4 PV.
Order-reversed runs use 200 ms warmup and a 3000 ms timing window.

| Policy schedule | Existing mode 2 | Mode 3 | Result |
|---|---:|---:|---|
| `fast`, candidate first | 0.108800 ms | 0.108576 ms | within timing noise |
| `fast`, baseline first | 0.108544 ms | 0.108576 ms | within timing noise |
| `accurate`, candidate first | 0.126560 ms | 0.126976 ms | mode 3 slower |
| `accurate`, baseline first | 0.126976 ms | 0.127296 ms | mode 3 slower |
| `fast`, S256/H16 | 0.010240 ms | 0.010240 ms | tied |

The likely mechanism is dependency latency: mode 3 removes instructions but
places two dependent FFMA-family operations on the scale-publication path.
That does not improve `fast` and is measurably worse for `accurate`.

## Numerics and model smoke

Across S4096 random seeds 0 through 3, mode 3 versus mode 2 has cosine above
0.9999997 and relative L2 between 0.00048 and 0.00057. The small difference
comes from moving a rounding boundary into the normalized affine map.

The S256/H16 model smoke gate remains finite:

| Adapter | Inputs | Non-finite outputs | Logit cosine vs BF16 | Relative L2 |
|---|---:|---:|---:|---:|
| ViT-B/16 CIFAR-10 | 4 images | 0 | 0.994282 | 0.109723 |
| BERT-base WikiText-2 | 2 blocks | 0 | 0.964284 | 0.278992 |

## Decision

Keep mode 3 as an explicit experiment, but do not promote it into the named
`fast`, `balanced`, `accurate`, or `exact` policies. It proves that the
finite bound can be represented by a bounded affine FMA and reduces static
instructions, but it does not improve wall time and regresses `accurate`.

An ordinary polynomial without a saturating input cannot provide a global
finite guarantee for unbounded shiftless scores. Any follow-up should target
the E4M3 byte construction or move the saturation to a non-critical owner,
not add a higher-degree polynomial to this serial path.

## Direct-byte follow-up

Scale-encode mode 4 moves E4M3 code generation onto conversion and integer
pipes:

```text
code = cvt.rni.sat.s32(fma(x, 8, 56))
code = min(max(code, 8), 126)
encoded_log2 = (code - 56) / 8
```

Although this has more static instructions than mode 2, it removes 16
floating `FMNMX` operations and substitutes integer `IMNMX`, `F2I`, and
`I2F` work. This relieves the floating-point issue path in the four-native
pair schedules.

| Policy | Mode 2 | Direct mode 4 | Change |
|---|---:|---:|---:|
| `fast`, seeds 0--3 | 0.10854--0.10883 ms | **0.10448--0.10499 ms** | 3.5--3.8% faster |
| `balanced`, order reversed | 0.11059--0.11085 ms | **0.10883--0.10925 ms** | 1.4--1.6% faster |
| `accurate`, order reversed | **0.12698 ms** | 0.13024 ms | 2.6% slower |

For `fast`, mode 4 versus mode 2 has relative L2 near `1.6e-4` and cosine
at numerical unity. Its four-image ViT and two-block BERT smoke results
exactly reproduce the guarded `fast` metrics above and contain no non-finite
outputs.

Scale-encode mode 5 also tested payload classification against the unrounded
log-scale, making byte production independent of the payload DAG. It
removed eight `FADD` and two `FMUL` instructions but tied baseline latency
and changed the payload output by about 0.070 relative L2, so it was
rejected.

The named `fast` and `balanced` policies now use direct mode 4. `accurate`
and `exact` retain mode 2 because conversion-pipe work is latency-critical
in their schedules. A post-promotion replay of the named `balanced` binary
reports policy ID 2, scale mode 4, finite guards enabled, 0.108672 ms,
cosine 0.968126, and relative L2 0.254818 against BF16.

The finite smoke gate is a legality check, not a model-quality result. The
subsequent matched 1,000-image/200-block replay in
`../fp4_fa4_direct_scale_downstream_20260729/README.md` finds meaningful
downstream degradation for `fast` despite zero non-finite rows.

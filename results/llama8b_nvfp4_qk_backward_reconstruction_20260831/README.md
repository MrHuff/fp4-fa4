# 8B NVFP4-QK backward reconstruction

Date: 2026-08-31 UTC

## Scope

This note records the authenticated step-5,000 natural-C4 replay used to
isolate the 8B NV-projection + FP8-PV training failure and the SM100
implementation rungs built from that diagnosis.  The represented-E4 publisher
is reachable only through the default-false
`exact_d128_represented_qk_backward` diagnostic selector.  v508 now has a
second, default-false B1-only Python runtime selector,
`exact_d128_native_score_backward`; it is not the production default.  The
gc-training tree has a hard-exit diagnostic renderer for authenticating the B1
v508 capsule, but no production training renderer selects v508.  v509 is a
separate fail-closed B1-only Python adapter and artifact.  It integrates the
mixed E4M3-by-E5M2 descriptor into the native-score kernel, but its E5M2 dO
producer is still standalone and neither v508 nor v509 is connected to a
production dispatcher.

The model loss on the fixed replay batch was `6.875484466552734`.  Captured
checkpoint, model, optimizer, job, and dataset artifacts were read only.

## Root cause

The native forward NVFP4 Q/K payload, row-K16 scale pages, global scales, and
layout decode are sound.  Decoding those bytes reproduces saved forward LSE
with RMS error `0.02898` at layer 10 and `0.02030` at layer 22.

The failure occurs when block-scaled NVFP4 Q/K are quantized a second time to
one plain E4M3 number per element for v501 backward.  The resulting LSE RMS
errors are `148.664` and `120.027--120.553`.  Changing the power-of-two lift
does not recover the mantissa bits, and E5M2 is worse for Q/K (`408.655` and
`313.575` LSE RMS error) because it trades away another mantissa bit.

This is not a missing learned-projection tensor scale.  The D128 exact runtime
already requires two-dimensional projection-weight scaling, separate global
scales, and local row-K16 Q/K scales.  The error is downstream of those scales.

The default-off represented-E4 publisher itself is mechanically sound: an
exact-source production-extension build passed B1/B2 checked-versus-unchecked,
represented-oracle, ownership, and sentinel gates.  Its receipt SHA256 is
`8bd0abf394379b18cd6a5524bc0fcb6bd0b832082cca1741b62bc760971a18c4`.
That validates the diagnostic implementation, not its training numerics; the
large LSE error above is why it remains fail-closed.

## v508 native-score hybrid

The v508 kernel uses the exact native NVFP4 Q/K payload and forward scales for
score/probability reconstruction, while retaining represented E4M3 Q/K/V/dO
for the gradient GEMMs.  It is restricted to
`B1/S4096/Hq32/Hkv8/D128` and rejects every other shape.  The kernel remains a
separate experimental artifact with no C++ production-dispatcher entry.  A
strict Python adapter can now select that authenticated artifact explicitly;
it reuses the exact caller-owned forward workspace and always enters through
the wrapper that clears additive dQ/dK/dV outputs.

Against chunked FP32 equations for that exact hybrid ABI:

| Layer | dQ rel-L2 / cosine | dK rel-L2 / cosine | dV rel-L2 / cosine |
|---:|---:|---:|---:|
| 10 | 6.759% / 0.997736 | 2.898% / 0.999595 | 1.522% / 0.999935 |
| 22 | 7.741% / 0.997055 | 2.626% / 0.999660 | 3.991% / 0.999612 |

The corresponding captured v501 relative-L2 errors were `74.197/31.881/0.460`
at layer 10 and `38.478/35.813/0.795` at layer 22 for dQ/dK/dV.  v508 improves
relative-L2 by `19.9x--1363.8x`.  dK and dV repeat bitwise; additive multi-owner
dQ is numerically repeatable within `3.3e-5` relative-L2.  The explicit 5% dV
gate is provisional and format-aware: v508 rounds reconstructed P to E4M3,
whereas the reference keeps P in FP32.

The compiled kernel used 128 registers, zero spills, and 193,840 bytes of
shared memory.  These are correctness results, not a throughput or convergence
claim.

The first real-adapter gate loaded the authenticated `5c92ecd...` artifact and
bound the immutable layer-10 step-5,000 workspace through the new adapter.
Two launches were finite and nontrivial.  dK/dV repeated bitwise; dQ repeated
within `5.71e-6` relative-L2 with cosine `0.99999976`.  This verifies the
Python-to-TK ABI.  A complementary exact-zero-dO gate poisoned all three
outputs before each of two launches; both launches returned exactly zero dQ,
dK, and dV, verifying that the runtime cannot bypass the clearing entrypoint.
These are still not an end-to-end optimizer trajectory.

The subsequent full-model replay loaded all 32 layers of the same step-5,000
checkpoint, reproduced the exact fixed-batch SHA256
`db5ffb7b4fc10076ba7c12b4041f0a67118f8884bfab5206d5dd044cf97cd53f`
and loss `6.875484466552734`, and completed exactly 32 authenticated v508
bind/run calls without a hang.  All 96 returned dQ/dK/dV tensors and all 128
Q/K/V/O projection parameter gradients were finite.  Peak PyTorch allocation
and reservation were `45,853,504,000` and `47,074,770,944` bytes.  The replay
performed no optimizer step and saved no checkpoint.

That execution gate exposed a second numerical failure rather than clearing a
training route.  Fixed-x4 E4M3 dO was `97.084%` zero on average across the 32
layers and exactly all-zero in eight layers.  With native forward score
reconstruction, dQ/dK were therefore exactly zero in 19 layers and dV in eight;
46 of the 128 projection gradients were exactly zero.  The represented-v501
control had only layers 30--31 at zero, but that apparent signal is not a
benefit: it came from the already-demonstrated Q/K score mismatch and produced
very large gradients even when the published dO tensor was almost entirely
zero.  v508 removes that false score-gradient source and makes the independent
dO range failure visible.

The full replay receipt SHA256 is
`6f135846c5e5838991d615f3322b89a431810baf2179e2edc971dbad544a99c9`.
This clears full-model binding, ABI, finiteness, and no-hang gates only.  It
does not clear a numerical or optimizer-trajectory gate.  v508 remains a
failed E4M3-dO rung; the separate v509 experiment below tests wider-range dO.

Sources:

- `tk_fa4/native_gqa_tk_bwd/v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_experimental_bshd.cu`
- `tk_fa4/native_gqa_tk_bwd/v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_experimental_bshd.cuh`
- `tk_fa4/native_gqa_tk_bwd/Makefile.v508`
- `tk_fa4/lowp_fa4_bwd/native_tk_d128_nvfp4_score_backward.py`

The create-only replay driver is portable, but its two historical layer
captures and exact historical extension are not distributed in this tree.
Supply authenticated copies explicitly; the driver checks their recorded
SHA256 values before importing the extension or touching the GPU:

```bash
python validate_v508_native_score_hybrid_step5000_v3.py \
  --capture-root /absolute/path/to/step-5000-captures \
  --v508-binary /absolute/path/to/exact-v508-extension.so \
  /absolute/path/to/new-receipt.json
```

`--repo-root` defaults to the inferred release checkout. The output must not
already exist. Absence of the captures or exact binary keeps this historical
replay blocked; rebuilding a new binary is new evidence, not the same replay.

Authenticated receipt SHA256:
`fea37a7beeab57d439bc420b77a867187350f4c345feb48b848ae130293a4b92`.

## E5M2 dO hardware rungs

E5M2 is independently useful for dO range.  Across all 32 captured layers it
reduces mean published zero fraction from `43.235%` to `2.931%` and relative-L2
error from `25.697%` to `5.763%`; cosine rises from `0.883765` to `0.998225`.
It does not repair Q/K score reconstruction.

Two standalone SM100 gates establish the missing implementation pieces:

1. Mixed E4M3-by-E5M2 tensor-core descriptors pass dP and dV, transpose and
   non-transpose, overwrite and accumulation cases.  All relative-L2 errors
   are below `8.6e-8`; the test includes 363 E5M2 values outside E4M3 range.
2. A fused BF16-dO-x4 producer emits genuine E5M2 bytes and computes dstat from
   the exact bytes published.  It matches all `16,777,216` bytes of the
   authenticated layer-12 replay, with dstat relative-L2 `1.19e-9` and p50
   `23.008 us` for the standalone full-capture kernel.

Their canonical receipt SHA256 values are respectively
`8fe3753b19f31842c3f032d2ee2d15681b7b02ed9569859ec735d2c6fc8cb145`
and
`53bd2385132d7b15c8ac90add8347d6335ff9600b1de07e4e6fe2478cf2e3b9d`.
Neither gate is integrated into v501/v508 or any training route.  v509 imports
the mixed-MMA implementation; its full-model replay invokes the producer as a
separate diagnostic kernel rather than through the production projection
epilogue.

## v509 mixed-E5M2 native-score rung

v509 preserves native NVFP4 Q/K and forward scales for score reconstruction,
represented E4M3 Q/K/V for gradient products, and changes only represented dO
to E5M2 with x4 encode and x0.25 decode.  dP and dV use the independently
verified E4M3-A by E5M2-B descriptor, while dQ and dK retain the v508 E4M3
path.  The authenticated artifact SHA256 is
`d06744a0073c7360d8db3e1314805706b83e3652b679c3712a98c5a27c99f54b`
(`5,971,528` bytes), built from fp4_matmul commit
`93e935d5aa39495ad5513ae84b5608b26db22d1e`.

The represented-equation gate validates the exact quantized v509 ABI, not a
BF16 backward or a training trajectory.  On natural step-5,000 C4 layer 12,
dQ/dK/dV relative-L2 errors were `0.048359`, `0.030575`, and `0.017404`, with
cosines `0.998836`, `0.999534`, and `0.999915`.  E5M2 dO was `0.9368%` zero and
producer dstat matched `-4*sum(O*raw_E5M2_dO)` at `1.19e-9` relative-L2.  The
exact-zero-dO sentinel gate returned finite exact-zero dQ/dK/dV.  dK and dV
repeated bitwise; additive dQ repeated within `4.86e-5` relative-L2.  The
canonical represented-oracle-v2 receipt SHA256 is
`d8467de9fd338320c32912015aecd525abb3f64432f316da7026f96e05f4ecb8`.

The subsequent read-only full-model replay used gc-training commit
`dfffe71e892ebdb11632a2ce59bac78388d75e2f`, the same fixed-batch SHA256
`db5ffb7b4fc10076ba7c12b4041f0a67118f8884bfab5206d5dd044cf97cd53f`,
and the unchanged forward loss `6.875484466552734`.  All 32 authenticated v509
calls completed: all 96 dQ/dK/dV tensors were finite and nontrivial, no layer
had exact-zero dQ, dK, or dV, and all 128 Q/K/V/O projection gradients were
finite and nontrivial.  This removes the local extinction observed with v508,
which had 19/19/8 exact-zero dQ/dK/dV layers and 46 exact-zero projection
gradients on the identical checkpoint and batch.

Mean E5M2-dO zero fraction was `14.4447%` across the full replay (minimum
`0.0508%`, maximum `23.8374%`), with no all-zero layer.  All 32 producer-dstat,
E5-payload pointer, and dstat-pointer gates passed.  Peak PyTorch allocation
and reservation were `45,922,185,728` and `47,162,851,328` bytes.  The source
HEADs, checkpoint manifest, DCP metadata, and replay driver stayed unchanged.
The replay performed no optimizer step or parameter update, saved no
checkpoint, and mutated no job.

The full replay summary SHA256 is
`4e65ddc8a4da8d2c30504f1660c93eafadd81c61715ab19502a7956aead40516`;
the result-note SHA256 is
`9c104bc688616f75876f7919d7be57e7ff74367617168b6eb4eee14585248615`.
This clears the represented-equation, live binding, finiteness, and local
gradient-nonextinction gates for one backward only.  It is not a throughput,
optimizer-trajectory, checkpoint-resume, or convergence result.  The replay
diagnostically forced a BF16 dO materialization, emitted the existing E4M3 dO
without consuming it, then invoked a separate caller-owned E5M2 producer and
allocation before v509.

Sources:

- `tk_fa4/native_gqa_tk_bwd/v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cu`
- `tk_fa4/native_gqa_tk_bwd/v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cuh`
- `tk_fa4/native_gqa_tk_bwd/Makefile.v509`
- `tk_fa4/lowp_fa4_bwd/native_tk_d128_nvfp4_score_e5m2_dout_backward.py`

## RHT audit

The current paired RHT path in `mfu_maxing` applies only to activation and dY
column carriers for projection weight gradients.  It leaves projected Q/K,
forward output, and dX byte-identical, so it cannot directly fix this
post-QKV/RoPE representation boundary.  An attention-specific paired Q/K RHT
would be a new ABI and remains untested.

## Remaining work

- Fuse E5M2 dO publication and represented-byte dstat into the projection
  epilogue, eliminating diagnostic BF16 storage, the unused E4M3 publication,
  the separate producer launch, and its external allocation.
- Authenticate that integrated runtime in a bounded renderer, then run a
  matched optimizer-update and checkpoint save/fresh-resume canary before any
  long training launch.  v508/E4M3-dO remains ineligible for training.
- Measure isolated and whole-step performance only after the production
  publication path exists; the current correctness replay has no throughput
  result.
- Extend the native-score path to B2 without reviving the known direct-D128
  CuTe 2-CTA hang.
- Run a short matched optimizer trajectory and then real-data convergence;
  one nonextinct backward is not convergence proof.

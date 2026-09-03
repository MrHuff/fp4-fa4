# TK low-precision backward restart

This directory is a clean restart from the optimized TK BF16 V382 backward.
The launcher and kernel were copied directly from the production V382 files
before any low-precision edits.  The production files remain untouched, and
the previous FP4/FP8 backward experiment tree has been removed.

`backward_bf16_control` is the invariant: it uses the copied V382 template
selection and must match the production route in output and steady-state time.
The new paths are modes inside that same copied schedule:

- `backward_fp8_native`: prepacked E4M3 Q/K and register-converted E4M3 dS;
  dK and dQ use native F8F6F4 tensor-core MMAs.
- `backward_fp4_native`: prepacked aligned E2M1 Q/K and E4M3 dS; dK and dQ
  use mixed E4M3 x E2M1 F8F6F4 tensor-core MMAs.
- Both modes retain the V382 BF16 score/softmax and dV path.  This first pass
  targets the dK = dS^T x Q and dQ = dS x K contractions only.

Q/K quantization and layout conversion happen before the timed backward call;
the kernel reads the supplied low-precision tensors directly.  The required dS
conversion is fused into the sliced FP32 dS producer.  Each 32-column quarter is
converted directly to its consumer-ready E4M3 register layout before the next
quarter is produced, so the kernel never materializes the complete BF16 dS
tile.  The fixed 4096 scale is folded into the FP32 exponent immediately before
the native saturating E4M3 conversion, avoiding a scale multiply, the
intermediate BF16 round/re-expand cycle, and a preprocessing kernel.

## Formats and layouts

| Mode | Q input | K input | dS | Default scales |
| --- | --- | --- | --- | --- |
| FP8 | contiguous E4M3 `[B,H,D,S]` | contiguous E4M3 `[B,S,H,D]` | E4M3 in shared memory | Q=256, K=256, dS=4096 |
| FP4 | aligned E2M1 byte container `[B,H,D,S]` | aligned E2M1 byte container `[B,S,H,D]` | E4M3 in shared memory | Q=16, K=16, dS=4096 |

The FP4 byte container follows the aligned-U4 layout expected by F8F6F4: each
16-value group occupies eight packed E2M1 bytes followed by an eight-byte
alignment gap.  `quantize_fp4_bhds_unpacked` and
`quantize_fp4_bshd_unpacked` create these two layouts.

The BF16 control supports the original V382 sequence/head matrix.  The initial
low-precision specializations currently support causal B=1, S=8192, H=8/16,
Dqk=192, Dv=128.

## D64/D128 causal GQA specialization

`tune_d64_gqa_cute.py` covers the ratio-4 causal-GQA geometries used by
1.2B-class D64 and 8B-class D128 Llama models.  It applies a project-owned patch
to a clean local SM100 CuTe backward at load time, so the nested CUTLASS checkout
stays read-only.  The retained topology gives each `(query head, K tile)` pair
its own CTA instead of serializing four query heads, writes BF16 dK/dV partials,
and merges them in FP32.

The merge is fused into the dQ conversion launch whenever that wins.  The
conversion grid has enough CTAs to cover every dK/dV vector, which overlaps the
two bandwidth-bound epilogues without adding mainloop barriers or TMEM.  D64
uses 128 reduction threads and a 16-row fused tile from S1024 upward; S512 keeps
the separate merge.  D128 uses 256 threads and a 32-row BF16 fused tile.

At causal B=1, Hq=32, Hkv=8, S4096 on GPU 0:

| Shape/mode | Retained (us) | Serialized GQA (us) | Speedup | MHA ceiling (us) |
| --- | ---: | ---: | ---: | ---: |
| D64 BF16 | 440.730 | 521.544 | 1.183x | 433.730 |
| D128 BF16 | 651.635 | 783.258 | 1.202x | 629.760 |
| D128 FP8 input / BF16 gradient | 496.655 | 569.508 | 1.147x | 486.116 |

The D128 FP8 route can afford Q2/dO3 staging because its operands consume half
the shared memory of BF16.  It uses one dK/dV stage and a 64-row fused epilogue
from S1024 upward (32 rows at S512).  At S4096 it is 23.78% faster than the
optimized D128 BF16 GQA route and only 2.17% above the FP8 MHA ceiling.  D64
does not cross over: its best FP8 result is 463.155 us versus 440.730 us BF16,
so BF16 remains the D64 backward default.

At the D128 S4096 geometry, pairing this backward with the measured causal
native-GQA NVFP4-QK/MXFP4-PV forward gives a 655.407-us attention-core F+B
aggregate versus 863.347 us for native BF16: 1.317x faster, or 24.09% less
time.  This component sum excludes projections, RoPE, packing, and optimizer
work; it is not a full-model E2E claim.  The diagnostic forward output cosine
is 0.949414 and naive backward E4M3 dV cosine is 0.963943, so adaptive
projection-native scaling remains mandatory before convergence evaluation.

The split path is fixed-length only.  BF16 validation at S256 has dQ/dK/dV
cosine above 0.999997 for both dimensions.  The timing control's naive unscaled
FP8 cast gives D128 dQ/dK cosine 0.999865/0.999838 but dV cosine 0.963943;
production use therefore requires the projection-native adaptive scaling path,
not this diagnostic input generator.

Reproduce either retained geometry from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  tk_fa4/lowp_fa4_bwd/tune_d64_gqa_cute.py \
  --sequence 4096 --query-heads 32 --kv-heads 8 \
  --dtype bf16 --warmup 10 --iterations 100

CUDA_VISIBLE_DEVICES=0 python \
  tk_fa4/lowp_fa4_bwd/tune_d64_gqa_cute.py \
  --head-dim 128 --sequence 4096 --query-heads 32 --kv-heads 8 \
  --dtype fp8 --warmup 10 --iterations 100
```

Use `--no-split-gqa-heads` for the serialized control.  The full shape sweep
and profile attribution are recorded in `results/tk_fa4_d64_gqa_20260814.md`
and `results/tk_fa4_d128_gqa_20260814.md`.

## Current GB200 results

Medians use nine warmups and 41 rotated-order samples on an otherwise idle
GPU.  Prepacking is outside the timed region.

| Shape | Production V382 | Copied control | FP8 native | FP4 native |
| --- | ---: | ---: | ---: | ---: |
| S8192 H8 | 0.518624 ms | 0.517920 ms | 0.513760 ms (-0.80%) | 0.516192 ms (-0.33%) |
| S8192 H16 | 0.914400 ms | 0.913504 ms | 0.909664 ms (-0.42%) | 0.914752 ms (+0.14%) |

Full-gradient accuracy relative to production V382:

| Shape/mode | dQ relative L2 / cosine | dK relative L2 / cosine | dV |
| --- | --- | --- | --- |
| H8 FP8 | 0.038973 / 0.999247 | 0.039245 / 0.999233 | exact |
| H8 FP4 | 0.121102 / 0.992640 | 0.120964 / 0.992659 | exact |
| H16 FP8 | 0.037377 / 0.999302 | 0.037605 / 0.999293 | exact |
| H16 FP4 | 0.120795 / 0.992682 | 0.121158 / 0.992642 | exact |

The hot FP8 and FP4 variants compile with 128 registers and no spill
loads/stores.  Following the forward FP4 kernel's direct-publication pattern,
the converted dS register fragment now fans out directly to the complete dK
tile, local dQ half, and peer-exchange dQ half.  This removes the old shared
store/reload/split round trip.  An NCU H8 FP4 comparison measured 31.3% fewer
shared-load wavefronts, 12.3% fewer total shared wavefronts, and 4.5% fewer
executed instructions; profiled kernel duration fell from 893.2 us to 855.7 us.
The existing publication fence immediately before the cluster transfer covers
these stores, so the earlier duplicate warp synchronization and fence were also
removed.

The dS producer now follows the same quantize-at-production rule.  The four
FP32 quarters are converted and retained incrementally, then published directly
to dK, local dQ, and peer dQ.  The ownership-preserving pack uses two shuffles
and a halfword rotation per output word; direct FP32 conversion removes 32
static BF16-pack instructions from the hot path.  Both low-precision kernels
remain at 128 registers, 16 barriers, and zero stack or spills.

Against the preceding direct-publication/staged-conversion H8 FP4 NCU report,
the retained kernel reduces profiled duration from 848.86 us to 826.46 us,
executed instructions from 121.78 M to 115.41 M (-5.23%), total sampled stalls
from 27,784 to 26,904 (-3.17%), and long-scoreboard samples from 10,531 to
10,160 (-3.52%).  The remaining profile is no longer dominated by dS
conversion: long-scoreboard exposure is largest, while 4,752 of 7,644 barrier
samples map to the overlapping score-half TMEM load.  That load must complete
and fence before the cross-warp handoff permits the dP producer to overwrite
the aliased columns, so its wait cannot safely be deferred past the handoff.
The next target is therefore earlier score production or useful work before
the load; more dS conversion work is unlikely to move the current S8192 H8/H16
result materially.

## Matched causal D64/D128 matrix

`benchmark_causal_backward_matrix.py` is the reproducible isolated comparison
for the retained causal GQA backward.  It creates one E4M3-represented
Q/K/V/dO state, decodes the same bytes for the CuTe BF16 control, and reports
dQ/dK/dV accuracy plus rotated-order timing.  The primary timing includes all
required output/workspace clears; clear-only and kernel-after-clear samples are
also retained.  Shapes run sequentially, each result is checkpointed to JSON,
and host/device free-memory guards run before each specialization.

D64 selects the coherent direct-TMA policy used by the 1.2B training route.
D128 selects the corrected coordinate-preserving shared E4M3 probability path
with two dO stages.  The two policies are recorded independently in every
result; D128 must not be described as having the D64 direct-TMA reducer.

For the controlled D64 epilogue A/B, request both `retained_lowp` and
`retained_lowp_no_direct_tma`.  The control reloads the same D64 split-GQA
topology with `direct_tma_dkdv=False` and passes that same flag to
`CompiledGqaBackward`; represented operands, projection-published statistics,
native EX2, dO staging, dS lift, probability storage/reuse, and the
clear-inclusive timing boundary remain unchanged.  This control is D64-only:
D128 already uses `direct_tma_dkdv=False`, and the harness rejects that pairing
before importing CUDA.

Resolve the direct-TMA A/B without touching CUDA:

```bash
python3 tk_fa4/lowp_fa4_bwd/benchmark_causal_backward_matrix.py \
  --dry-run --sequences 4096,8192 --head-dims 64 --head-pairs 32/8 \
  --routes retained_lowp,retained_lowp_no_direct_tma \
  --seeds 20260820 --output /tmp/causal_backward_direct_tma_plan.json
```

Resolve a full core matrix without touching CUDA:

```bash
python3 tk_fa4/lowp_fa4_bwd/benchmark_causal_backward_matrix.py \
  --dry-run --sequences 512,1024,2048,4096,8192 \
  --head-dims 64,128 --head-pairs 32/8 \
  --seeds 20260820,20260821,20260822 \
  --output /tmp/causal_backward_matrix_plan.json
```

Run it with exactly one otherwise-idle GPU exposed:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 -B \
  tk_fa4/lowp_fa4_bwd/benchmark_causal_backward_matrix.py \
  --sequences 512,1024,2048,4096,8192 \
  --head-dims 64,128 --head-pairs 32/8 \
  --seeds 20260820,20260821,20260822 \
  --warmups 9 --samples 41 \
  --output /tmp/causal_backward_matrix.json
```

The route registry intentionally marks `mx_exact_replay` unavailable.  The
clean retained line does not contain a callable adapter from the forward
MXFP4 payload and published E8M0 probability scales into this backward.  The
exploratory MX probes are not silently substituted.  The JSON manifest records
the required adapter inputs and timing boundary so that route can be added
without changing the comparison schema once its source is reconstructed.

## Saturated B16 training-compatible forward factorial

`benchmark_b16_forward_factorial.py` isolates the training-compatible forward
boundary that the 1.2B saturated trainer otherwise hides inside decoder and
backward timing. The projection specializations publish the Q/K/V operands
retained for backward, but the harness executes no backward kernel. It runs
exactly four B16/S4096/Hq32/Hkv8/D64 cases from one BF16 activation and packed
learned-QKV weight draw:

- E4M3 QKV projection with exact E4M3 FP8-PV;
- E4M3 QKV projection with MXFP4-PV;
- native NVFP4 QKV projection with exact E4M3 FP8-PV; and
- native NVFP4 QKV projection with MXFP4-PV.

Every case has a private preallocated publication workspace, attention output,
and LSE. One untimed first call authenticates the allocating legacy projection
against the compact out-parameter ABI; measured projection calls use the
unchecked shape-bound symbol. CUDA-event provider order rotates across the
four cases for operand preparation, projection/publication, attention,
prepared projection-plus-attention, and the full combined boundary.

After timing, the harness computes an untimed BF16 packed-QKV projection,
applies the same pairwise RoPE, and runs causal Torch SDPA with GQA. Every case
is compared to that BF16 output and the process exits nonzero unless both the
configured cosine and relative-L2 thresholds pass. Pairwise MX-versus-FP8 and
NVFP4-versus-E4M3 output/LSE metrics remain diagnostic.

The allocation distinction is intentional. Projection/publication, attention,
and their prepared combined boundary use caller-owned output APIs. The current
E4M3 and NVFP4 operand-preparation helpers return new payload and scale tensors,
so operand preparation and the full combined scope are functional allocating
APIs. No allocator trace is collected, so the harness does not claim that
caller-owned stages are free of hidden transient CUDA allocations. The JSON
schema records this limitation.

All three selected binaries must be regular non-symlink files. Their paths and
SHA-256 digests are captured before import, then each loaded module's
`__file__` and digest are reauthenticated before any benchmark allocation.

Resolve and inspect the complete command without importing Torch or touching
CUDA:

```bash
python3 tk_fa4/lowp_fa4_bwd/benchmark_b16_forward_factorial.py \
  --mx-extension /path/to/b16_mxfp4_pv.so \
  --fp8-extension /path/to/b16_exact_fp8_pv.so \
  --projection-extension /path/to/_C_b300_lowp_bwd.so \
  --minimum-bf16-output-cosine 0.95 \
  --maximum-bf16-output-relative-l2 0.35 \
  --output /tmp/b16_forward_factorial.json \
  --dry-run
```

Run on one selected otherwise-idle GB200 with the Python environment used to
build the three extensions:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /path/to/venv/bin/python -B \
  tk_fa4/lowp_fa4_bwd/benchmark_b16_forward_factorial.py \
  --python /path/to/venv/bin/python \
  --mx-extension /path/to/b16_mxfp4_pv.so \
  --fp8-extension /path/to/b16_exact_fp8_pv.so \
  --projection-extension /path/to/_C_b300_lowp_bwd.so \
  --warmups 4 --samples 40 \
  --minimum-bf16-output-cosine 0.95 \
  --maximum-bf16-output-relative-l2 0.35 \
  --output /tmp/b16_forward_factorial.json
```

## Fused attention RMSNorm to native NVFP4

The experimental B16/S4096/H2048/D64 route now fuses attention RMSNorm with
native-NVFP4 activation preparation and uses a two-stage CUDA RMSNorm
backward.  Enable it only together with native caller-owned NVFP4 QKV output
and 2D learned-weight scaling:

```bash
--qkv-projection-format nvfp4 \
--experimental-native-nvfp4-projection-out \
--experimental-fused-attention-rmsnorm-nvfp4
```

At this exact shape, two deterministic component runs measured a 5.22x p50
forward speedup over eager RMSNorm plus the same exact native packer and a
6.32x p50 backward speedup over the closed-form eager derivative.  In the
saturated 16-layer Llama-1.2B harness, fused MXFP4-PV improved pooled p50 step
time from 599.423 to 539.578 ms (1.1109x) and reduced peak allocated HBM by
20.001 GiB.  Fused MXFP4-PV decoder forward was 0.615 ms faster than fused
FP8-PV at p50; their whole-step throughput was tied within 0.073%.

The implementation, exact format boundary, numerical caveats, component JSON,
and six raw saturated results are recorded in
`results/llama12b_fused_rmsnorm_nvfp4_saturated_20260826/README.md` and
`results/fused_rmsnorm_nvfp4_component_20260826.json`.  Use
`benchmark_fused_rmsnorm_nvfp4.py` for the fixed-shape authenticated component
gate.

## Build and reproduce

From this directory:

```bash
make -j1
```

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 -B \
  tk_fa4/lowp_fa4_bwd/benchmark_control.py \
  --seqlen 8192 --heads 8 --warmup 9 --iterations 41
```

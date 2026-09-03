# Native ThunderKittens D128 causal-GQA backward provenance audit

Date: 2026-08-29

## Verdict

No validated, optimized **native ThunderKittens (TK)** SM100 implementation of
the Llama-8B backward geometry `B=1, S=4096, Hq=32, Hkv=8, D=128, causal`
was recovered.

The remembered approximately `1.3x` D128 causal-GQA backward result is real,
but it belongs to a **CuTe DSL low-precision backward compared with a CuTe DSL
exact-BF16 control**. It is not evidence that native TK beat CuTe DSL.

A genuine native TK SM100 D128/GQA backward implementation survives under
`tk_fa4/deprecated`. Its generic BF16 causal-GQA path is numerically correct,
but a fresh matched benchmark measured it at `6.578432 ms`, versus
`0.479504 ms` for exact-BF16 CuTe DSL: native TK was `13.719243x` slower. The
optimized native hot paths have always excluded causal GQA.

The publication/training recommendation is therefore:

1. Use the validated CuTe DSL backward for D128 causal GQA now.
2. Describe the preserved `1.3x` result as CuTe low precision versus CuTe
   exact BF16.
3. Treat a competitive native D128 implementation as new kernel development,
   not recovery or activation of a lost route.
4. Do not use the August 28 native causal-GQA probe: it is numerically invalid.

## Terminology and evidence boundary

This audit uses **native TK** only for CUDA sources that include and program
the ThunderKittens C++ API directly (`kittens.cuh`, TK global/shared/register
tiles, TMA, warpgroup/tcgen operations). A Python harness that modifies or
compiles CUTLASS CuTe DSL `BlackwellFusedMultiHeadAttentionBackward` is called
**CuTe DSL**, even when it lives under `tk_fa4` or is combined with a native TK
forward kernel.

Evidence is separated into:

- **Fresh measurement:** the matched GPU-2 benchmark run during this audit.
- **Preserved result:** checked-in reports and their authenticated harnesses.
- **Rejected experiment:** an uncommitted native probe whose outputs fail
  correctness.
- **Port candidate:** source that has the desired mathematical geometry but
  does not build or validate on SM100.

The audit executed isolated kernels on physical GPU 2 and made one failed
build attempt with its output directed to `/tmp`. Apart from this receipt, it
did not edit kernel code, produce a new binary, change job/cluster state, or
commit. The active publication worktree was:

```text
/workspace/codebases/pv/fp4_matmul_monolithic_tk_20260828
branch: codex/monolithic-tk-fa4-train-20260828
HEAD:   6530af2551984154bc5d97f6b76eb37c9dca1af8
```

## Genuine native SM100 D128 source and lineage

The surviving implementation is:

- [`fa4_bwd_unified_sm100.cuh`](../../tk_fa4/deprecated/fa4_bwd_unified_sm100.cuh)
- [`fa4_bwd_dkdv_sm100.cuh`](../../tk_fa4/deprecated/fa4_bwd_dkdv_sm100.cuh)
- [`fa4_bwd_dq_sm100.cuh`](../../tk_fa4/deprecated/fa4_bwd_dq_sm100.cuh)
- [`tk_fa4.cu`](../../tk_fa4/deprecated/tk_fa4.cu)
- [`Makefile`](../../tk_fa4/deprecated/Makefile)

`git log --follow` authenticates this lineage:

| Commit | Date (UTC) | Meaning |
| --- | --- | --- |
| `b293bddce9f24c7de3479be218bd96c2861b6857` | 2026-03-26 | Added the unified native SM100 BF16 backward pipeline. |
| `9ba509255dc459ffd2b45615c80077a2c596485f` | 2026-03-29 | Last optimization checkpoint before reset. |
| `a0c2ed3db65bfbfa305160d3f06ac193e4c51703` | 2026-03-30 | Moved the stack under `deprecated/` while resetting around exact B300 attention. |

The complete intermediate history is present in Git between those commits.
The current source hashes are:

| File | SHA-256 |
| --- | --- |
| `fa4_bwd_unified_sm100.cuh` | `6a1c696c57c082d4b24f6688094ed36ea6185f59aafedc2bd711ff30a3da11d8` |
| `fa4_bwd_dkdv_sm100.cuh` | `826c744db5b385809685d988e094530e76bcdc674cfdf0759a2923275e79c35b` |
| `fa4_bwd_dq_sm100.cuh` | `dbb638ef9305b7bd80810eb8ccf53ab5f4ee42f2ba1a7af16b3f4ed01d40ce8e` |
| `tk_fa4.cu` | `5331e13fffd693205a93f7179777ea7d1076b10ad3752db275e140c142606b5f` |
| `Makefile` | `034ec05e270efc595046d0e8f21a00e7094b56a10eeee65362dd684919df69b1` |

The build declares `GPU := B200`; the compiled binary contains SM100/D128
launch symbols. The exact binary used for the fresh benchmark was:

```text
/workspace/codebases/pv/fp4_matmul/tk_fa4/deprecated/
  _C_legacy.cpython-312-aarch64-linux-gnu.so
size:   6,690,984 bytes
sha256: 68631d4d0435cc78bd04fef8f589e27af2894ff865574681a6c09b9070cd57c2
```

The active publication worktree also contains a newer build from the same
hashed sources. It was **not** the benchmarked artifact:

```text
/workspace/codebases/pv/fp4_matmul_monolithic_tk_20260828/tk_fa4/deprecated/
  _C_legacy.cpython-312-aarch64-linux-gnu.so
size:   6,690,880 bytes
sha256: 063b6be0a231434f0bfb5938dda5ceea5b000d0224f8b80a3fd4b6177126075f
```

### Why causal GQA misses every optimized native route

The gate is explicit rather than an inferred dispatcher problem.
`clustered_backward_supported` requires all of:

```cpp
q.size(3) == 128 &&
!causal &&
q.size(1) == k.size(1) &&
actual_seq_len == q.size(2) &&
q.size(2) % 256 == 0
```

`wg_hot_backward_supported` has the same `!causal` and equal-Q/KV-head
requirements, with sequence divisibility by 128. The equal-head and noncausal
restrictions are present from the first revisions that introduced these hot
routes; they are not a late regression.

For `causal=true`, `Hq=32`, and `Hkv=8`, dispatch therefore selects the generic
native route. Its public boundary:

1. allocates/initializes `delta`, `dq_accum`, and `dq_semaphore` storage;
2. launches an `O*dO` preprocess;
3. launches the generic main kernel on a KV-head grid and accumulates partial
   dQ;
4. launches the dQ reduction; and
5. postprocesses gradients.

This explains why the generic path can be mathematically correct yet far from
the performance of the specialized CuTe D128 causal-GQA schedule.

## Fresh matched D128 causal-GQA benchmark

The benchmark ran on physical GPU 2, an NVIDIA GB200 (`SM100`), using:

```text
Python:        3.12.3
PyTorch:       2.9.0a0+145a3a7bda.nv25.10
PyTorch CUDA:  13.0
seed:          20260829
shape:         B=1, S=4096, Hq=32, Hkv=8, D=128, causal=true
input dtype:   BF16
input scale:   random normal multiplied by 0.25
warmup:        5 calls per backend
samples:       20 individually synchronized CUDA-event samples per backend
```

Both backends consumed the same Q/K/V/dOut and the same exact CuTe forward
state. Timing covered each backend's public backward extension boundary. The
native inputs were pre-transposed once to its B/H/S/D ABI, so layout conversion
was outside the timed region. The native environment forced the generic path:

```text
TK_FA4_BWD_MODE=ref
TK_FA4_BWD_WG_HOT=0
TK_FA4_BWD_DENSE_HOT=0
```

### Timing

| Backend | Median (ms) | Mean (ms) | Minimum (ms) | Maximum (ms) |
| --- | ---: | ---: | ---: | ---: |
| Native TK SM100 generic BF16 causal GQA | 6.578432 | 6.575798 | 6.527136 | 6.614464 |
| Exact-BF16 CuTe DSL public boundary | 0.479504 | 0.480365 | 0.471328 | 0.500192 |

Native TK / CuTe latency was `13.719243x`. Equivalently, CuTe was
`13.719243x` faster for this public-boundary comparison.

This is a decisive route-selection result, not a full model-step result. It
does not claim that 13.7x transfers to end-to-end training, and it does not
separate the generic native kernel from its required allocation/preprocess/
reduction/postprocess work.

### Correctness against exact CuTe

| Gradient | Finite | Cosine | Relative L2 | Maximum absolute difference |
| --- | --- | ---: | ---: | ---: |
| dQ | yes | 0.9999958873 | 0.0028906709 | 0.0002527237 |
| dK | yes | 0.9999957681 | 0.0028798385 | 0.0002578795 |
| dV | yes | 0.9999985695 | 0.0016670791 | 0.0038695335 |

The result rules out gross numerical failure as the reason the legacy generic
route is slow.

### Exact executed benchmark command

The following is the successful command, preserved verbatim except for normal
Markdown indentation:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/workspace/codebases/pv/fp4_matmul:/workspace/codebases/pv/fp4_matmul/flash-attention \
TK_FA4_BWD_MODE=ref \
TK_FA4_BWD_WG_HOT=0 \
TK_FA4_BWD_DENSE_HOT=0 \
/tmp/fp4_fa4_native_tk_b16_20260828/venv/bin/python - <<'PY'
import json, statistics, torch
from flash_attn.cute.interface import _flash_attn_bwd as cute_bwd
from flash_attn.cute.interface import flash_attn_func as cute_fwd
from tk_fa4.deprecated import interface

C = interface._C
torch.manual_seed(20260829)
B, S, HQ, HK, D = 1, 4096, 32, 8, 128
scale = D ** -0.5
q = (torch.randn(B, S, HQ, D, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
k = (torch.randn(B, S, HK, D, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
v = (torch.randn(B, S, HK, D, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
out, lse = cute_fwd(q, k, v, causal=True, deterministic=False, return_lse=True)
dout = (torch.randn_like(out) * 0.25).contiguous()

lse_bsh = lse.permute(0, 2, 1).contiguous() if lse.ndim == 3 else lse
laux = (-lse_bsh / scale).permute(0, 2, 1).contiguous().unsqueeze(2)
to_bhsd = lambda x: x.permute(0, 2, 1, 3).contiguous()
qbh, kbh, vbh, obh, dobh = map(to_bhsd, (q, k, v, out, dout))

tkg = C.mha_bwd(qbh, kbh, vbh, obh, laux, dobh, True, scale, S, False)
cug = cute_bwd(q, k, v, out, dout, lse, softmax_scale=scale, causal=True)
torch.cuda.synchronize()

def met(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    d = a - b
    return {
        "finite": bool(torch.isfinite(a).all()),
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30)),
        "rel_l2": float(d.norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float(d.abs().max()),
    }

metrics = {
    name: met(t.permute(0, 2, 1, 3), c)
    for name, t, c in zip(("dq", "dk", "dv"), tkg, cug)
}

def time(fn, warm=5, n=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }

native = time(lambda: C.mha_bwd(qbh, kbh, vbh, obh, laux, dobh, True, scale, S, False))
cute = time(lambda: cute_bwd(q, k, v, out, dout, lse, softmax_scale=scale, causal=True))
print(json.dumps({
    "device": torch.cuda.get_device_name(0),
    "shape": [B, S, HQ, HK, D],
    "native_backend": "TK SM100 reference causal GQA public extension boundary",
    "cute_backend": "flash_attn.cute exact BF16 public boundary",
    "metrics_native_vs_cute": metrics,
    "native": native,
    "cute": cute,
    "native_over_cute": native["median_ms"] / cute["median_ms"],
}, indent=2))
PY
```

The exact CuTe BF16 core came from FlashAttention commit
`9743edaf3227a25f6afc4fa7be8b5e8498610553`. Its unmodified core files were:

| File | SHA-256 |
| --- | --- |
| `flash_attn/cute/flash_bwd_sm100.py` | `9907efd53656c7f3fca52ea9d3e5c170382ab5d4edcb67d6b34772d6c4b0fccb` |
| `flash_attn/cute/flash_bwd_preprocess.py` | `c07eebda83bb86d3a7a6a98b954a0371967742c5c373d3869db479370a5f0a04` |
| `flash_attn/cute/flash_bwd_postprocess.py` | `07578e12f94cec0941f8751ff415c258633a6a2e17f779897ba985015b48f025` |

The surrounding `flash_attn/cute/interface.py` worktree file had unrelated
low-precision edits and hash
`13a1edbd711ae29141fceb69c54a8a93bc18384511792cebf3ee433ff220cd75`.
The comparator invoked the exact BF16 route, not the modified FP4 route.

## Why the remembered D128 speedup is CuTe DSL

Three preserved reports make the backend provenance explicit:

| Preserved report | Git commit | SHA-256 | Relevant result |
| --- | --- | --- | --- |
| [`tk_fa4_d128_gqa_20260814.md`](../tk_fa4_d128_gqa_20260814.md) | `3d13d8bafc66ab1517b8d5ffb69dcdcfe2f4b880` | `301eb73f5e86a92ec0c2bef352a243623770c385a321cf07d8030b43eb9e4d51` | Explicitly says “warm cache, CuTe event timing”; exact BF16 `651.635 us`, low precision `496.655 us`, or `1.312048x`. |
| [`tk_fa4_gqa_d128_chain_20260814.md`](../tk_fa4_gqa_d128_chain_20260814.md) | `a5b43340e6f14310b5b70eba6435310558b36ea0` | `c25d207a8880a43adfeddd4e5375fdcd8791575da81bcf54899ee61d7fd7a492` | Says the emitted statistics are consumed by CuTe backward; exact BF16 `489.856 us`, low precision `320.640 us`, or `1.527745x`. |
| [`tk_fa4_compact_dq_gqa_20260814.md`](../tk_fa4_compact_dq_gqa_20260814.md) | `91b6efdc1297f6d2fe7e2f69bceb1026fc3d6c3e` | `bd96dd32dae84e74efa10e7c599cc7a739d662940f9d0f650fd1dbb64baa5367` | Explicitly says CuTe event timing; exact BF16 `651.366 us`, low precision `419.391 us`, or `1.553123x`. |

The authenticated harness confirms this mechanically:

- [`profile_gqa_d128_chain.py`](../../tk_fa4/lowp_fa4_bwd/profile_gqa_d128_chain.py),
  SHA-256 `c4ff4d5b159b71886abe47c5cdd8e0b67ae3cd75e766d0a862b68851b082cf68`,
  instantiates `control.BlackwellFusedMultiHeadAttentionBackward` and calls
  `control.cute.compile(...)`.
- [`tune_d64_gqa_cute.py`](../../tk_fa4/lowp_fa4_bwd/tune_d64_gqa_cute.py),
  SHA-256 `ef7b0154196059945245f841c3c2c5c2b7577333456600a8e6455a9550f47de7`,
  patches the CUTLASS CuTe backward class.

The `Native BF16` wording in the forward-plus-backward component table of the
first report must not be read as proof of a native TK backward. The report's
method states CuTe backward timing; the aggregate combines independently
measured components.

## Rejected August 28 native causal-GQA probe

An experimental worktree tried to broaden the native WG-hot path to causal
integral-GQA head ratios:

```text
worktree: /workspace/codebases/pv/fp4_matmul_tcgen_probe_20260828
base:     e7ed20b2ea7e5c9eada37e8ce8b13c7815fd57dd
state:    modified fa4_bwd_unified_sm100.cuh; untracked probe script
```

Artifacts:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `tk_fa4/deprecated/fa4_bwd_unified_sm100.cuh` | — | `bfc3cc42ba0a8825f9a8516582a2f0be2a0645440f59a5400b6b5e891fff8c79` |
| `tk_fa4/deprecated/probe_wg_hot_causal_gqa.py` | 5,182 bytes | `355db38197b840ebeb59ccbaf98943682f511563074008ec5218ee81aa153c9f` |
| `tk_fa4/deprecated/_C_legacy.cpython-312-aarch64-linux-gnu.so` | 6,822,040 bytes | `8a87f70e329dc49ddecd29c0ca98d131fd20439e3570b84a72feab58d20717f3` |

Executed command:

```bash
cd /workspace/codebases/pv/fp4_matmul_tcgen_probe_20260828/tk_fa4/deprecated
CUDA_VISIBLE_DEVICES=2 python3 probe_wg_hot_causal_gqa.py
```

At `B=1, S=128, Hq=32, Hkv=8, D=128, causal`, repeated runs were invalid:

| Run | dQ | dK | dV | Zero-dOut invariant | Median (ms) |
| --- | --- | --- | --- | --- | ---: |
| A | rel-L2 `26.427202`, cosine `0.016034` | nonfinite | nonfinite | dQ max `53.707752`; dV nonfinite | 0.150304 |
| B | rel-L2 `26.427389`, cosine `0.016022` | rel-L2 `0.828911`, cosine `0.676073` | nonfinite | dQ max `53.707752`; dV nonfinite | 0.154800 |

The probe's latency includes preprocess/allocation/postprocess, but it is not a
performance result because correctness fails catastrophically. Its patch and
binary must not be promoted, cited as a speedup, or used for training.

## Hopper D128/GQA source is a port candidate, not an SM100 implementation

ThunderKittens contains a genuine native Hopper implementation at
`ThunderKittens/kernels/attention/mha_h100/mha_h100.cu`. It has a D128
backward specialization, accepts `q_heads >= kv_heads`, and requires an
integral GQA ratio.

Provenance:

```text
ThunderKittens submodule commit:
  9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b
file last changed by:
  e463ed89e2f7cc145022404e80faade95a126f09 (ThunderKittens 2.0)
mha_h100.cu sha256:
  4f1f76cbd21487c09845e22e666d4a53a7060ac87ec076053a7d8664bb2efd3e
Makefile sha256:
  64e9fc9d2646cc1ad2cd541da43d2c9c920350f4e4632681587eb552d9ecbf91
```

The source Makefile targets H100. A bounded build attempt changed only the
command-line target/output:

```bash
PATH=/tmp/fp4_fa4_native_tk_b16_20260828/venv/bin:$PATH \
make GPU=B200 OUT=/tmp/tk_mha_d128_gqa_sm100_probe.so \
  -C ThunderKittens/kernels/attention/mha_h100
```

It failed with ten compile errors. Blackwell `kittens::group<4>` has no
`mma_async_wait` or `mma_commit_group`, which the Hopper WGMMA schedule uses.
No output binary was produced. A real port must replace the Hopper WGMMA
synchronization/ownership design with Blackwell tcgen05, TMEM, and matching
barrier semantics; `GPU=B200` is not sufficient.

## Closest validated native scheduling template

The closest project-owned native template is the D64 V383 query-parallel
experiment, not a D128 implementation:

```text
source:
  tk_fa4/native_gqa_tk_bwd/v383_d64_gqa_e4m3_query_parallel.{cu,cuh}
source sha256:
  cu:  b034338f68ce2a1c730842c78d4b69a674eff6155eaddd9ed13f798e76f34896
  cuh: 7d371835c55a2ddf63f85d09de5ba9fda6214e06fd2937b454c2bde0e279b220
artifact:
  /tmp/tkfa4_v383_d64_gqa_query_parallel_build/
    _C_b300_gqa_tk_v383_d64_e4m3_query_parallel.cpython-312-aarch64-linux-gnu.so
artifact sha256:
  3dd368508ebd51ac6f1ec1212fba4d4576418d1abf02e9b60fdd13b446866c15
validation log:
  /tmp/tkfa4_v383_d64_gqa_query_parallel_build/
    validation_attempt8_final_clean.log
validation-log sha256:
  59c8434db4b3f972683d478ae4e035c09e9799db1845b16ec58380b5eb4e0dee
```

Its B1 dQ/dK/dV relative L2 values were
`0.0727565 / 0.0741449 / 0.00267065`, with cosines
`0.997350 / 0.997248 / 0.999996`. Its B16/S4096 median was `7.411552 ms`.
No matched CuTe timing was preserved in that log, so this number is not a
CuTe speedup claim. The files are currently untracked and the artifact is
temporary; they are useful as a design template only.

The current native exact-B300 stack is also not a D128 Llama substitute: it
statically requires `Dqk=192`, `Dv=128`, and does not authenticate causal GQA
with `Hq=32/Hkv=8`.

## Required work for a future competitive native D128 route

A native effort should begin from one of two explicit ports:

1. Extend the D64 V383 query-parallel ownership and scheduling design to
   D128, including correct shared-GQA dK/dV accumulation and publication.
2. Port the Hopper D128/GQA algorithm to Blackwell tcgen05/TMEM/barrier
   semantics.

Either route must pass, in order:

1. zero-dOut invariants for dQ/dK/dV;
2. small-shape causal GQA comparison against exact BF16;
3. D128 `B=1` and saturated-batch correctness;
4. a matched public-boundary benchmark against exact CuTe; and
5. model-level numerical and end-to-end timing validation.

Until those gates pass, the CuTe DSL D128 backward is the publication-grade
and training-grade implementation.

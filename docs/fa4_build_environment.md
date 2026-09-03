# Reproducing the FA4 kernel build

The paper kernels were compiled and measured on NVIDIA GB200 (Blackwell
SM100). The verified software receipt is:

| Component | Exact version |
| --- | --- |
| NVIDIA GPU | GB200, compute capability 10.0 |
| NVIDIA driver | 580.126.09 |
| CUDA toolkit | 13.0 |
| Python | 3.12 |
| PyTorch | 2.9.0a0+145a3a7bda.nv25.10 |
| CUTLASS DSL | 4.5.2 |
| Triton | 3.5.1 |
| FlashInfer | 0.6.15.post1 |
| pybind11 | 3.0.1 |

The Python pins are in `requirements-fa4-gb200.txt`. The recorded PyTorch
build and CUTLASS DSL distribution may require NVIDIA's package channel or
container; the repository does not silently substitute a public wheel.

## Pinned source dependencies

Initialize the repository submodules before building:

```bash
git submodule update --init \
  ThunderKittens SageAttention flash-attention qutlass cutlass
git -C flash-attention submodule update --init csrc/cutlass
git -C qutlass submodule update --init third_party/cutlass
```

This deliberately avoids optional ROCm-only gitlinks in the FlashAttention
fork. They are not part of the Blackwell build closure.

`tools/build_fa4.py` verifies both the recorded gitlink and checked-out commit
for ThunderKittens, FlashAttention, qutlass, and CUTLASS. The release verifier
also authenticates SageAttention, which is needed by retained historical
diagnostics but not by the supported causal build. The tools reject a dirty or
wrong checkout. In particular, the FlashAttention checkout is the durable
runtime-overlay commit, not the older base commit on which the overlay was
developed.

## Plan, verify, and build

Choose all locations explicitly. `FA4_BUILD_ROOT` should be a new, empty
directory. `FA4_CUTLASS_DSL_ROOT` is the absolute `python_packages` directory
that directly contains `cutlass/__init__.py` and exactly one native library at
`cutlass/_mlir/_mlir_libs/_cutlass_ir*.so`.

```bash
FA4_BUILD_ROOT=/absolute/path/to/new/fa4-build
FA4_CUDA_ROOT=/absolute/path/to/cuda-13.0
FA4_CUTLASS_DSL_ROOT=/absolute/path/to/cutlass-dsl/python_packages

python tools/build_fa4.py plan \
  --build-root "$FA4_BUILD_ROOT" \
  --cuda-home "$FA4_CUDA_ROOT" \
  --cutlass-dsl-root "$FA4_CUTLASS_DSL_ROOT"

python tools/build_fa4.py verify \
  --build-root "$FA4_BUILD_ROOT" \
  --cuda-home "$FA4_CUDA_ROOT" \
  --cutlass-dsl-root "$FA4_CUTLASS_DSL_ROOT"

python tools/build_fa4.py build \
  --build-root "$FA4_BUILD_ROOT" \
  --cuda-home "$FA4_CUDA_ROOT" \
  --cutlass-dsl-root "$FA4_CUTLASS_DSL_ROOT"
```

The default build compiles:

- the standalone MXFP4 quantizer;
- the low-precision projection and V-publication extension;
- causal NVFP4-QK with E4M3 FP8-PV forward at B1, B2, and B4;
- causal NVFP4-QK with MXFP4-PV forward at B1, B2, and B4; and
- native v509 backward at B1, B2, and B4.

Every shape is fixed to S4096, 32 query heads, eight key/value heads, and head
dimension 128. The compiler target is `sm_100a`. A subset can be selected with
repeatable `--target` and `--batch` options. The tool refuses to overwrite an
existing binary; use a new build root for a clean rebuild.

After a successful build, `manifests/` contains one
`fa4_artifact_manifest_v3` file per route and batch: BF16 plus the full
E4M3/NVFP4 learned-projection by FP8/MXFP4 P/V matrix. Each manifest records
absolute paths, Python module names, byte lengths, and SHA-256 hashes. It also
records the live release `HEAD`, tree, dirty state, submodule identities, and a
sorted per-file SHA-256 closure of every supported build/runtime source. The
content closure is authoritative, so local kernel development is supported
without falsely attributing dirty bytes to `HEAD`. The loader rehashes this
closure, including the selected CUTLASS DSL package and native library, before
a manifest can be used. The training adapter consumes exact artifact paths and
never discovers extensions by glob. Low-precision B2
remains operator-only; the authenticated training routes are B1 and B4. Only
NVFP4 learned projections plus FP8-P/V is the release candidate. MXFP4-P/V and
the E4M3 projection controls remain diagnostic routes under their recorded
evidence limits.

The D64 / 1.2B route is a separate fixed profile and should use a different
empty build root:

```bash
python tools/build_fa4.py build \
  --profile llama1p2b-d64-b16 \
  --build-root /absolute/path/to/new/fa4-d64-build \
  --cuda-home "$FA4_CUDA_ROOT" \
  --cutlass-dsl-root "$FA4_CUTLASS_DSL_ROOT"
```

It compiles B16/S4096/Hq32/Hkv8/D64 FP8-P/V and anchored MXFP4-P/V forward
images, the shared E4M3 projection/backward publisher, and native v416
backward. Its schema-v3 manifest set contains BF16 plus the two E4M3 learned-
projection low-precision routes. Profile, shape, module, bytes, and SHA256 are
validated as one contract; D128 artifacts are not accepted as substitutes.
See `release/D64_REPRODUCTION.md` for the 50B-token renderer profile and the
historical evidence boundary.

For an operator-only compiler check that intentionally omits FlashAttention
and CUTLASS DSL runtime identities, add `--operator-only`. Such a manifest is
rejected by the training launcher.

## Standalone CUTLASS projection controls

Two auxiliary NVFP4 GEMM controls used while investigating projection fusion
are kept outside the FA4 runtime build graph. Their original sources and a
portable compatibility recipe are present:

```bash
git submodule update --init cutlass
make gemm CUDA_HOME=/absolute/path/to/cuda-13.0 BUILD_DIR=build-gemm
./build-gemm/bench_nvfp4_gemm
./build-gemm/bench_grouped_nvfp4_gemm
```

These binaries target `sm_100a` and require CMake 3.31, CUDA 13.0, and the
pinned CUTLASS checkout. They are development controls, not dependencies of
the paper's selected attention route.

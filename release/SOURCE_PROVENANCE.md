# Source provenance

This repository carries two project source epochs because the paper reports
both the final causal training path and an earlier non-causal forward study.
The earlier epoch also contains backward and general development work needed
to resume the project faithfully. The epochs are kept separate so a newer
kernel is never substituted silently for the one that produced an older
result.

## Causal training source

The root `tk_fa4`, `TK_quantisation`, `baseline_kernels`, `fused_ops`, and
`qutlass_binding` directories are source exports of
`graphcore-research/fp4_matmul` commit
`4590537f1479e1a7e847f2783e9ab7aa7f11b975` before any release adapter changes.
Their upstream Git tree identities are:

| Path | Git tree |
| --- | --- |
| `tk_fa4` | `6aafb4201ad6ae618d3724b851680a3c0ec13eb3` |
| `TK_quantisation` | `c6454d9524e1cd521427411b0dd2a199d0822a25` |
| `baseline_kernels` | `5242c6d77a09cbd415b6d11e100657e0810aa4dd` |
| `fused_ops` | `efb0668033a9eeee8a95b21759664d6d49f5decc` |
| `qutlass_binding` | `32332ed7b0d26971bb0873647d154eb8fdc6aa65` |

One architecture-specific tracked build product,
`TK_quantisation/nvfp4_v5/_tk_quant_v5.cpython-312-aarch64-linux-gnu.so`, is
intentionally excluded. Its source remains present and the release always
rebuilds it. Kernel and quantization source bytes otherwise retain their
upstream paths and modes. The materialized `tk_fa4` tree identity is recorded
in `release/manifest.json`. Release changes are limited to Python evaluation
portability plus six later portability fixes recovered from source commit
`4c394504998f653aa702d030c5f98864dcf34c75`: configurable build outputs,
explicit corpus/tokenizer roots, and temporary build/capture locations. HAO
defaults resolve to the vendored comparator, downstream fixed-input evaluations require
explicit authenticated model/data/build roots, and Wan separates its logical
model identifier from its authenticated local model path. These changes do not
alter any CUDA/C++ kernel bytes, operands, or measurement calculations.

This source contains all retained causal forward and backward kernels, their
Python bindings, build files, correctness gates, profiling tools, and the
diagnostic variants discussed in the appendix. The retained D128 backward is
the v509 source family; internal revision names are kept in filenames for exact
build provenance but are not method names.

The causal export retains 617 of the 621 paths in the pinned `tk_fa4` tree.
The four omitted paths are generated SM100a cubins listed in
`PUBLIC_SANITIZATION.md`; their source, PTX dumps, and available annotated SASS
dump remain present. All 256 C, C++, CUDA, header, and CUDA-header files are
byte-identical to that source epoch. The changed materialized tree identity
reflects the documented portability edits and removal of those build products;
it is not evidence of a kernel rewrite.

The direct D128 CuTe FP4-QK experiment is present only for provenance. Its
two-CTA schedule can hang and is not selected by the public adapter. The
previously uncommitted runtime overlay was preserved separately before release
work began as:

- `patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.patch`
  (`bc8caf8cd3c860d2bf958a96113a4b97a7987b2350bfed7f54337f0b9ac0cb8a`);
- `patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.manifest.json`
  (`3559f8402156ed08f4c873592e3189f5919e3136107ca41898a3db9ed4ada315`).

The pinned `flash-attention` submodule commit already contains the durable
version of that overlay. The patch remains a recovery and review artifact, not
an instruction to apply it again.

The local-only dense-score diagnostic commit
`aa02150404418859e33d1ff99fb46543244b9b70` is preserved under
`reproduction/snapshots/v510_aa021504/`. Its parent is
`5d8512f9ce1a36c2fdd6475ef75f327e213a7c45`, commit tree is
`ca712fd4547dd50d776fe157f217eeabf53cc847`, and `tk_fa4` tree is
`5966739ff26dcfa9512f307422e5ffaac731a13f`. The snapshot contains the exact
post-commit versions of all 14 changed paths, a one-commit patch, and a SHA256
inventory. It is deliberately not overlaid on the later v509 source because
five integration files diverged before subsequent correctness and batching
fixes. No portable receipt survives that establishes the exact reason for its
negative result.

Two additional source-only snapshots preserve the D64 / 1.2B history without
overlaying old files on the current route:

- `reproduction/snapshots/d64_training_cd59dda` is the exact public
  source/test delta from `fp4_matmul` commit
  `cd59dda37ebf22e0d77b9c9d6851ec164b86e3af` (parent
  `1f3cae064fd3bd5c72c713f7dcea53c4b073952d`, commit tree
  `046066c1fd54f1f79fe363fbc4a38a37a495060c`, `tk_fa4` tree
  `592fc6dfda76eb561e6b7f8fbfd040648c1a1c40`). It contains all seven
  non-result paths changed by the commit, including the matched real-token
  training and merge harnesses.
- `reproduction/snapshots/d64_v416_713819d` is the exact public source/test
  delta from commit `713819d730369ad9e73ded1aedbc301c261f1130`
  (parent `abd3f33104ac885434f1d6136ab5100361de51ee`, commit tree
  `4133134eb97b45910962f513fc5d3f71b6f0d1cd`, `tk_fa4` tree
  `8221a0d371a2c2307725b12d3cd0f287d1989ae7`). It contains all 33
  non-result paths changed by the commit, including the v389--v416 lineage,
  selected v416 sources, Makefiles, Python runner, and contract test.

Both snapshots were materialized from Git objects, not copied from a mutable
worktree. Their manifests bind each path to the historical Git blob, byte
count, and SHA256; adjacent `SHA256SUMS` inventories cover the manifest,
README, source-only one-commit patch, and materialized paths. Result artifacts
remain in the root `results` tree. Scheduler YAML and credential-bearing
inputs are excluded.

These snapshots do not close every historical input. The 220,876-byte CuTe
control source identified by SHA256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`
is unavailable, as are the historical compiled binaries. The public D64
profile rebuilds source and binds new binaries; it does not claim those new
artifacts are byte-identical to the missing controls.

## Historical `cfc06dad` source epoch

The operator, HAO-grid, and downstream forward measurements were made from the
older `fp4_matmul` commit
`cfc06dadf684279f657ab66254a3a074be4ee3a9`. Its complete source/development
state is materialized under `reproduction/snapshots/forward_cfc06dad`, including
the matching `TK_quantisation` tree and the backward prototypes that coexisted
with the reported forward work. The upstream `tk_fa4` tree is
`33312c0d36a221b5d6a20b8a3a3a79d2cd7cff42`; the materialized 126-path
source/development tree is `1dcefd373495bd9fbcd2ca39331daa45fde77132`.
Of those paths, 125 are byte-exact and one is the documented portability edit
to `fp4_fa4_fwd/hao_comprehensive_suite.py`.

The upstream epoch has 150 `tk_fa4` paths in total. The other 24 are generated
or result artifacts, all present byte-identically under the root causal
`tk_fa4/results/` tree; they are not duplicated in the snapshot. The union of
the root and snapshot therefore retains every historical path and byte while
keeping only one copy of the large cubin, PTX, SASS, JSON, and Markdown result
artifacts. The historical 12-file `fp4_fa4_bwd/` subtree is preserved exactly
as Git tree `dd35ecca9db03cdbb063af7e4b3762438b9d5cca`. Its local transitive source
closure, including the shared B300 headers, is present. Matching SageAttention
and ThunderKittens links resolve to the same pinned revisions as the root
epoch.

The historical portability fix
`patches/historical_hao_suite_cfc06dad.patch` is applied in that materialized
tree; the patch SHA256 is
`8301b554e3912e6fd24735a5607ea5b35e5f55986397ad3b2c42b0adadc72a1b`.

One release-only portability change makes the HAO root configurable through
`HAO_FLASH_ATTN_ROOT` and defaults it to the vendored comparator. It does not
change a kernel, operand, timing loop, or result calculation.

The historical tree's prebuilt AArch64 Python extension is excluded. It was a
machine-specific build product, not source, and the release build tool creates
and authenticates a fresh extension instead.

The HAO comparator is a snapshot of
`hao-ai-lab/flash-attention-fp4` commit
`9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247` under
`third_party/hao_flash_attention_fp4`. The recorded compatibility patch is
applied there:

- `patches/hao_flash_attention_fp4_9b0abef_compat.patch`
  (`448aac4ea9eea45517259de3c315de3f9062189243febb62439de42c4e799ea5`).

That commit is not reachable from the current upstream branch, which is why a
source snapshot is included instead of an unusable submodule pointer.

## Results and manuscript

The root `results` tree starts from report commit
`4c394504998f653aa702d030c5f98864dcf34c75`; its upstream Git tree identity is
`8f90e08e2988a2e3d4684f74f46cfe5011eb18e9`. The source repository marks 51
machine/build log files `export-ignore`; those files are intentionally absent
from this release export. We additionally exclude all 32 executable scheduler
specifications. They contain no literal credentials, but they encode internal
submission details that are neither evidence nor required by the portable
TorchTitan launcher.

The release copy includes subsequent manuscript edits, regenerated paper
artifacts, normalized current-paper receipts with their scientific values
preserved, and a portable command-line boundary for the retained v508
diagnostic. Unused proprietary fonts and two unused HAO assets are absent; the
exact transformation is recorded in `PUBLIC_SANITIZATION.md`. Its reviewed
materialized tree identity is
`5d625f44bec7b206fffb32dabb1ab14f52f6324f`. It contains the committed raw
records that were available, normalized receipts, deterministic renderers, and
the manuscript source. A committed receipt proves only what it records. It
does not turn an absent raw capture into a repeat measurement.

## Pinned dependencies

| Path | Revision | License boundary |
| --- | --- | --- |
| `ThunderKittens` | `9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b` | MIT |
| `SageAttention` | `681004015c42b8ac543302235652e618ac66f966` | Apache-2.0 |
| `flash-attention` | `b531f67557b8213db339492cd1629e721776f758` | BSD-3-Clause |
| `flash-attention/csrc/cutlass` | `7127592069c2fe01b041e174ba4345ef9b279671` | NVIDIA CUTLASS terms |
| `qutlass` | `406e86fb2d7df436e94f825bcda8e59b1a7250a6` | Apache-2.0 |
| `qutlass/third_party/cutlass` | `b2ca083d2bb96c41d9b3c5a930637c641f6669bf` | NVIDIA CUTLASS terms |
| `cutlass` | `acb45938e9cb3e4db8c1d75155b63d31791e0e5d` | NVIDIA CUTLASS terms |

Initialize the Blackwell build closure with:

```bash
git submodule update --init ThunderKittens SageAttention flash-attention qutlass cutlass
git -C flash-attention submodule update --init csrc/cutlass
git -C qutlass submodule update --init third_party/cutlass
```

The pinned FlashAttention fork also names optional ROCm-only submodules. They
are not used by an experiment or supported build in this release, so the
verifier authenticates their gitlinks through the parent revision but does not
require their worktrees. Project and third-party license scopes are recorded in
the root notice and `LICENSES/README.md`.

The public tree omits three redundant patch captures that contained a private
object-store locator and an unused upstream banner PDF that embedded a personal
build path. Those files existed in an earlier private preparation commit.
The public repository therefore uses a parentless release commit; the private
development history remains in a separate recovery repository.

## Clean-clone audit

A full-history clone of private commit
`f04ed49bfbe9820c09f34a5f622d18998e873467` was audited independently. The
root and nested dependency pins matched, all 3,729 inventoried files verified,
the release verifier passed, and the complete offline paper graph regenerated
the authenticated 57-page PDF without changing the clone. The machine-readable
receipt is `release/audits/remote_clone_f04ed49b_20260902.json`.

This audit authenticates source and deterministic offline artifacts. It does
not replace the pending clean-clone GB200 build, numerical gates, performance
measurements, or distributed checkpoint-resume test.

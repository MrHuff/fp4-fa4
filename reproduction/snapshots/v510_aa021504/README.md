# Dense-score E4M3/E5M2 backward diagnostic (v510)

This directory preserves the complete file overlay introduced by the
local-only research commit
`aa02150404418859e33d1ff99fb46543244b9b70`. Its parent is
`5d8512f9ce1a36c2fdd6475ef75f327e213a7c45`; the commit tree is
`ca712fd4547dd50d776fe157f217eeabf53cc847`, and its `tk_fa4` tree is
`5966739ff26dcfa9512f307422e5ffaac731a13f`.

The branch tested a dense E4M3 score path with represented E4M3 Q/K/V and
represented E5M2 output gradients. It is an experimental B1/S4096/Hq32/Hkv8/
D128 route. The implementation is deliberately shape-fail-closed and was
never connected to the retained training adapter.

The branch name and commit classify this as a negative experiment, but no
portable measurement receipt survives here that establishes the precise
numerical or performance rejection. Therefore this snapshot preserves the
implementation and its gates without asserting a reason that the available
evidence cannot support. It must not be presented as a working or measured
paper route.

## Contents

The `tk_fa4/` and `tests/` paths are the exact post-commit versions of all 14
files changed by `aa021504`. This includes:

- the v510 CUDA translation unit, mainloop, Makefile, Python runtime, and
  natural-capture validator;
- the fused E5M2 publication and full-model integration state on that branch;
  and
- all source-contract and integration tests added by the commit.

`aa021504.patch` is the corresponding one-commit patch against `5d8512f`.
`SHA256SUMS` authenticates the materialized files and patch. The parent commit
is not part of this repository's public history, so the materialized files—not
the availability of that historical Git object—are the durable recovery
artifact.

## Safe use

Do not copy this overlay onto the current root in place. The branch diverged
before later v509 correctness, batching, and release changes, so its five
modified integration files would regress the retained route.

To resume this experiment, start with a disposable clone, compare each
materialized integration file with its current counterpart, and port the
v510-specific code under a new diagnostic selector. Keep all of these
conditions until new evidence clears them:

- B1/S4096/Hq32/Hkv8/D128 only;
- dense E4M3 score Q/K and represented E4M3 gradient Q/K/V;
- represented E5M2 dO with the recorded scale ABI;
- a separately authenticated publisher/backward pair;
- no production or TorchTitan dispatch; and
- natural-capture correctness, exact-zero-dO, liveness, and timing gates
  before any promotion.

The retained production route and its public names are documented in
`release/KERNEL_MAP.md`. This snapshot exists so that future work does not
have to reconstruct or unknowingly repeat the v510 branch.

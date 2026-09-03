# Historical `cfc06dad` source epoch

This directory preserves the source/development state of `fp4_matmul` commit
`cfc06dadf684279f657ab66254a3a074be4ee3a9`. The paper's non-causal Direct-P
measurements use this epoch. Its backward prototypes are retained as well so a
future researcher can continue the historical line without reconstructing it
from an unavailable private branch.

Source accounting:

- upstream `tk_fa4` tree: `33312c0d36a221b5d6a20b8a3a3a79d2cd7cff42`;
- materialized `tk_fa4` tree: `1dcefd373495bd9fbcd2ca39331daa45fde77132`;
- 126 source/development paths are materialized here;
- 125 are byte-exact to the upstream epoch;
- `tk_fa4/fp4_fa4_fwd/hao_comprehensive_suite.py` carries the documented
  replay/portability patch; and
- 24 generated/result paths are not duplicated because identical bytes remain
  at `tk_fa4/results/` in the root causal export.

The exact historical `tk_fa4/fp4_fa4_bwd` subtree has Git tree
`dd35ecca9db03cdbb063af7e4b3762438b9d5cca`. The local shared headers it
includes are also materialized. `ThunderKittens` and `SageAttention` resolve
through relative symlinks to the repository's matching pinned dependencies.

Do not edit this snapshot in place for new development. Copy or port the
specific experiment to a new, named route, preserve the source identity, and
add build, correctness, liveness, and evidence gates. See
`release/SOURCE_PROVENANCE.md`, `release/routes.json`, and
`release/LEGACY_LINEAGE.md` from the repository root.

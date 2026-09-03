# D64 native-backward v416 source epoch (`713819d`)

This directory preserves the complete public source/test overlay introduced by
`fp4_matmul` commit `713819d730369ad9e73ded1aedbc301c261f1130`
(`Optimize production D64 GQA TK backward`). Its parent is
`abd3f33104ac885434f1d6136ab5100361de51ee`, the commit tree is
`4133134eb97b45910962f513fc5d3f71b6f0d1cd`, and the commit's `tk_fa4`
tree is `8221a0d371a2c2307725b12d3cd0f287d1989ae7`.

The snapshot contains the exact post-commit code/test version of every
non-result path changed by the commit: the Python caller-owned runtime, build
targets, the v389--v416 optimization lineage, the selected v416 translation
unit and mainloop, and its contract test. The source-only patch excludes the
four measurement receipts added by the commit. Those receipts remain in the
release's root `results/native_tk_d64_ptx_adaptation_20260829/` tree, where
they are treated as evidence rather than executable source.

v416 consumes contiguous BSHD E4M3 Q/K/V/dO at B16/S4096/Hq32/Hkv8/D64,
uses the prelifted statistic pages `8 - LSE*log2(e)` and
`-16*sum(O*dO)`, and returns BF16 gradients through caller-owned outputs. It
is the retained native backward for the portable D64 profile. It does not
recreate the earlier CuTe control: the exact historical CuTe source bytes
identified in receipts by SHA256 `cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`
are unavailable, as are the corresponding historical binaries.

`manifest.json` binds every materialized file to its historical Git blob,
size, and SHA256. `SHA256SUMS` additionally authenticates this README, the
manifest, the source-only patch, and all materialized files. Develop from the
current root route and use this directory only to inspect or port the exact
historical overlay.


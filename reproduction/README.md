# Reproduction source snapshots

The root causal implementation and the paper's earlier non-causal forward
implementation are intentionally not collapsed into one mutable source tree.

`snapshots/forward_cfc06dad` contains the exact first-party forward source state
used by the unified HAO-grid measurements. The recorded replay patch is applied,
and one additional release-only edit replaces a hard-coded HAO checkout with
`HAO_FLASH_ATTN_ROOT` (defaulting to the vendored comparator). The measured
kernel files retain the hashes recorded in the committed shard manifests.
The archived AArch64 `.so` is deliberately omitted: it is a build product, not
portable source, and `tools/build_fa4.py` produces a newly authenticated binary.

The snapshot's `ThunderKittens` symlink points to the repository's pinned
submodule. Initialize submodules before building:

```bash
git submodule update --init --recursive
```

Use `tools/reproduce_fa4_paper.py --check --offline paper-pdf` to inspect the
artifact graph, and see `release/EXPERIMENT_MATRIX.md` for GPU measurement and
external-data commands.

The older files directly under `reproduction/` are preserved low-precision
linear-layer experiments from the source repository. They are not part of the
FlashAttention paper's headline route.

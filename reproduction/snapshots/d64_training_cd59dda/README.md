# D64 real-token training source epoch (`cd59dda`)

This directory preserves the complete public source/test overlay introduced by
`fp4_matmul` commit `cd59dda37ebf22e0d77b9c9d6851ec164b86e3af`
(`Enable isolated full-depth FA4 training comparisons`). Its parent is
`1f3cae064fd3bd5c72c713f7dcea53c4b073952d`, the commit tree is
`046066c1fd54f1f79fe363fbc4a38a37a495060c`, and the commit's `tk_fa4`
tree is `592fc6dfda76eb561e6b7f8fbfd040648c1a1c40`.

The materialized paths are the exact post-commit versions of all seven code
and test files changed by that commit. They contain the 1.235B/D64 model
definition, matched-route setup, real-token packing/training harness, and
result-merging logic used by the August 21 short Dolma3 experiment. The
source-only patch excludes scheduler manifests and all result data.

This is a historical overlay, not a standalone environment. In particular,
the exact CuTe backward-control file whose recorded SHA256 begins `cd57e336`
and the historical compiled extensions are not available. The release also
does not redistribute the Llama tokenizer or the Dolma3 input bytes. Those
gaps are recorded in `release/DATA_PROVENANCE.md`; a new run must use an
authenticated replacement and must not be described as byte-identical to the
historical trajectory.

`manifest.json` binds every materialized file to its historical Git blob,
size, and SHA256. `SHA256SUMS` additionally authenticates this README, the
manifest, the source-only patch, and all materialized files.


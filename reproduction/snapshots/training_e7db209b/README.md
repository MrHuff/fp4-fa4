# Historical 8B training-data snapshot

This directory preserves the data loading, tokenization, checkpoint/resume,
and trainer source used by the matched 8B experiment. The source revision is
`gc-training` commit `e7db209b0c7017c415fdd66e04e85f96ae24f276`.

The seven Python files are byte-identical to that revision:

| Historical relative path | SHA256 |
| --- | --- |
| `low_bits_training/datasets/mosaic_datasets.py` | `4b368542eb40084c3b3c95bbdd9786fcc013f00e0cfa99d2319f6e06ec7e19c2` |
| `low_bits_training/datasets/common.py` | `499f16fe80e44d71d0e33457b07d84fdca86eec1e32a1099f78070136d3512e1` |
| `low_bits_training/components/tokenizer.py` | `897b53ffc8b685da6bf884b83f8469f7909f18d50ae29eb2642039565508f371` |
| `low_bits_training/components/tiktoken_tokenizer.py` | `f80b69fa5e380da61e5f4744050d020c2ab0829b892677f0db3729758cf6a9ad` |
| `low_bits_training/checkpoints.py` | `15442802c766247c71d924d786023fee60c41d763118488dab350d22187150fc` |
| `low_bits_training/trainer.py` | `ab5e41304070e6f6a4d2b03838ebe45eb674ab8b63bdc878ab48b3296a27bc24` |
| `low_bits_training/batch_resume.py` | `3573d2b9509e6b5329a3ffdfdfba97e4d87c9140c39b4742181cd6a36356bb2a` |

The training TOML is a deterministic sanitized copy. The historical file has
SHA256
`9cd0f52092ed5a49905a5c12f1924bb1d915a2fc7fa0629add27f430489bd0cc`.
It named an internal object-store location in the two `dataset_path` fields.
Both values, and only those values, are replaced here by
`__HISTORICAL_SLIMPAJAMA_MDS_ROOT__`. The sanitized file has SHA256
`c813fd247b3bcd4e61818e3b6030d54d0c9d799f9b41da31361b19650f03af41`.
The original URI is intentionally not published.

This snapshot authenticates historical behavior; it is not a standalone
launcher. The portable TorchTitan integration at the repository root supplies
the new-run path. As documented in `release/DATA_PROVENANCE.md`, an exact
historical replay is still blocked by the missing SlimPajama MDS index, shard,
and batch-order identities.

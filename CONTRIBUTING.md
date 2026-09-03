# Contributing

Issues and focused pull requests are welcome. This repository is a research
release built on TorchTitan, so a change must keep the supported, diagnostic,
and disabled kernel routes distinct.

## Before opening a pull request

1. Create a branch from `main`.
2. Add or update tests for source, configuration, or behavior changes.
3. Update the route catalog, evidence manifest, and scientific handoff when a
   change affects a reported result.
4. Run the CPU-visible checks:

   ```bash
   pip install -r requirements-dev.txt
   python tools/generate_fa4_source_inventory.py --check
   python tools/verify_fa4_release.py
   pytest -q
   ```

5. State which Blackwell build, numerical, liveness, performance, distributed,
   and checkpoint tests were run. Do not imply that an unavailable gate passed.

## Licensing and attribution

Contributions are accepted under the project license in `LICENSE`, except for
files that state a different license. Preserve all file-level notices and the
dependency scopes in `THIRD_PARTY_NOTICES.md` and `LICENSES/README.md`.

TorchTitan itself is maintained at
<https://github.com/pytorch/torchtitan>. Changes intended for upstream
TorchTitan should follow that project's contribution process. This repository
is not an official PyTorch package or release.

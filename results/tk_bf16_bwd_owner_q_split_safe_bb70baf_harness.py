from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_HARNESS = (
    REPO / "results" / "tk_bf16_bwd_s2048h8_owner_q_split_harness.py"
)
EXPECTED_BASE_HARNESS_SHA256 = (
    "0d48a87044aee04a51c675719c7108443ca21d9385e2e07409b9fff2fcd89224"
)
EXTENSION = REPO / "results" / ".artifacts" / (
    "tk_bf16_bwd_owner_q_split_safe_bb70baf_20260717"
) / "_C.cpython-312-aarch64-linux-gnu.so"
EXPECTED_EXTENSION_SHA256 = (
    "f2531ef668b2cdc9d58542543dd05335e04e3fd0242641cc00af68a7ce7fc373"
)
EXPECTED_EXTENSION_SIZE = 10_736_808


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    actual = sha256_file(BASE_HARNESS)
    if actual != EXPECTED_BASE_HARNESS_SHA256:
        raise RuntimeError(f"base harness SHA mismatch: {actual}")
    spec = importlib.util.spec_from_file_location(
        "tk_bf16_bwd_owner_q_split_safe_bb70baf_base",
        BASE_HARNESS,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_HARNESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXTENSION = EXTENSION
    module.EXPECTED_EXTENSION_SHA256 = EXPECTED_EXTENSION_SHA256
    module.EXPECTED_EXTENSION_SIZE = EXPECTED_EXTENSION_SIZE
    module.__file__ = str(Path(__file__).resolve())
    module.main()


if __name__ == "__main__":
    main()

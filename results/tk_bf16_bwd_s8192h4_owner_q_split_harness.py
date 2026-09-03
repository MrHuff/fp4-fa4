from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_HARNESS = (
    REPO / "results" /
    "tk_bf16_bwd_s8192h16_persistent_v_branchless_harness.py"
)
EXPECTED_BASE_HARNESS_SHA256 = (
    "d2d0e37f84c300df381f61dcc96908809171830b76df015ee0a8ec2f4e338bf5"
)
EXTENSION = REPO / "results" / ".artifacts" / (
    "tk_bf16_bwd_s8192h4_owner_q_split_20260717"
) / "_C.cpython-312-aarch64-linux-gnu.so"
EXPECTED_EXTENSION_SHA256 = (
    "4d365f3760b01b152f57328aaa4a8e44993c2cff1b0ccaf0051fef38a79594f7"
)
EXPECTED_EXTENSION_SIZE = 10_803_000
PARENT_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_"
    "s8192h4_persistent_v_branchless_internal"
)
CHILD_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_"
    "s8192h4_owner_q_split_internal"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_harness():
    actual = sha256_file(BASE_HARNESS)
    if actual != EXPECTED_BASE_HARNESS_SHA256:
        raise RuntimeError(f"base harness SHA mismatch: {actual}")
    spec = importlib.util.spec_from_file_location(
        "s8192h16_persistent_v_branchless_owner_q_split_h4_base",
        BASE_HARNESS,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_HARNESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base_harness()
    base_initial_manifest = base.initial_manifest
    base.EXTENSION = EXTENSION
    base.EXPECTED_EXTENSION_SHA256 = EXPECTED_EXTENSION_SHA256
    base.EXPECTED_EXTENSION_SIZE = EXPECTED_EXTENSION_SIZE
    base.CONTROL_ROUTE = PARENT_ROUTE
    base.CELLS = {"U16": (8192, 4)}
    base.CHILD_ROUTE = CHILD_ROUTE
    base.__file__ = str(Path(__file__).resolve())

    def initial_manifest(args):
        payload = base_initial_manifest(args)
        payload["derived_harness"] = {
            "path": str(BASE_HARNESS),
            "sha256": EXPECTED_BASE_HARNESS_SHA256,
        }
        payload["routes"] = {
            "persistent_v_branchless_bitwise_parent": PARENT_ROUTE,
            "owner_q_split_child": CHILD_ROUTE,
            "reference": "CuTe DSL BF16 FA4 backward",
        }
        payload["static_gate"] = {
            "pass": True,
            "runtime_scope": (
                "exact B1 S8192 H4 even/odd fused-Q-work owner split"
            ),
            "parent_and_split_sass_match_retained_exact64": True,
            "extracted_sass_sha256": {
                "parent": (
                    "3e782762ec81f5a4b5613de9be6ec66b04410c6b33a4f7f8eb6a7badf91a2299"
                ),
                "split_0": (
                    "f77436cfdf7d37676c6666fbf9ed894e49e23e90fc72615c91828bb11a23dc5f"
                ),
                "split_1": (
                    "53546f8755e68e70a6562f031d0177f09967d92fc5a00b24d923fbf887d38256"
                ),
            },
            "resources": {
                name: {
                    "registers": 128,
                    "stack_bytes": 0,
                    "local_bytes": 0,
                    "shared_bytes": 231276,
                    "spill_loads": 0,
                    "spill_stores": 0,
                }
                for name in ("parent", "split_0", "split_1")
            },
            "same_method_instruction_counts": {
                "parent": 4096,
                "split_0": 3528,
                "split_1": 3536,
            },
            "utchmma_2cta_per_body": 52,
            "public_dispatch_unchanged": True,
        }
        return payload

    base.initial_manifest = initial_manifest
    base.main()


if __name__ == "__main__":
    main()

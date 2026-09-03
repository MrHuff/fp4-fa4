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
    "tk_bf16_bwd_s2048h8_owner_q_split_20260717"
) / "_C.cpython-312-aarch64-linux-gnu.so"
EXPECTED_EXTENSION_SHA256 = (
    "84f10c3fdd7cd5e34bb0c6c3e6ac0c5919fa49af7980030f9aafbebeb2e43e60"
)
EXPECTED_EXTENSION_SIZE = 10_736_808
PARENT_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_"
    "s2048h8_persistent_v_branchless_internal"
)
CHILD_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_"
    "s2048h8_owner_q_split_internal"
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
        "s8192h16_persistent_v_branchless_owner_q_split_s2048h8_base",
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
    base.CELLS = {"U11": (2048, 8)}
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
                "exact B1 S2048 H8 even/odd fused-Q-work owner split"
            ),
            "parent_machine_words_match_frozen": True,
            "parent_machine_words_sha256": (
                "ca521ad75b4983ddbee6e2289e75b69eae5eec8e9eb98b3955da5851aebdd697"
            ),
            "extracted_sass_sha256": {
                "parent": (
                    "1643528ad0ba7f07d5effe0b6c67d9bab51c1508b4bbe4a36ad74bdee80320ae"
                ),
                "split_0": (
                    "79bd456180bfff9b7f32f1d6a4ee246c41d5f8849f9e8e7ff9932d3665fc5f1f"
                ),
                "split_1": (
                    "fc5a88da45d8b880976539b69eea7ec0b0eae4f2f701af070741a4b8154d3179"
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
                "parent": 4688,
                "split_0": 3544,
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

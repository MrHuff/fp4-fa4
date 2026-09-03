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
    "tk_bf16_bwd_s4096h4_owner_q_split_20260717"
) / "_C.cpython-312-aarch64-linux-gnu.so"
EXPECTED_EXTENSION_SHA256 = (
    "f798c9b2da6084f45f8ecb839efe9e16ebf0a4d015fa251b90f0008d1c5ece28"
)
EXPECTED_EXTENSION_SIZE = 10_798_808
PARENT_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_"
    "s4096h4_persistent_v_branchless_internal"
)
CHILD_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_"
    "s4096h4_owner_q_split_internal"
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
    base.CELLS = {"L4096H4": (4096, 4)}
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
                "exact B1 S4096 H4 even/odd fused-Q-work owner split"
            ),
            "parent_machine_words_match_frozen": True,
            "parent_machine_words_sha256": (
                "7393b30673d2a86a89d43029c48def933877d93969b7bce0df0ee85e66754a75"
            ),
            "extracted_sass_sha256": {
                "parent": (
                    "3a42469a8e219e11eb56380953990f38e8afec1b511e8a06d324cef1ab96d901"
                ),
                "split_0": (
                    "80c08acc8e93d1ffd9bfbf0d881f9d54940bcaa4c9f3bf5f6837aff1338abd9e"
                ),
                "split_1": (
                    "6487ce975c5d9fadc81ac29dd7578413e414d160badb956d692486bc88bb3104"
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
            "retained_h2_parent_and_split_sass_unchanged": True,
            "public_dispatch_unchanged": True,
        }
        return payload

    base.initial_manifest = initial_manifest
    base.main()


if __name__ == "__main__":
    main()

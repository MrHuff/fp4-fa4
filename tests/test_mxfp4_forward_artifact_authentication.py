from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tk_fa4.lowp_fa4_bwd.authenticate_causal_gqa_mxfp4pv_forward import (
    CANONICAL_ARTIFACTS,
    COMMON_D64_CAUSAL_MX_TOPOLOGY,
    VARIANT_TOPOLOGY,
    authenticate_artifact,
    canonical_artifact_spec,
    require_topology,
)


def _topology(
    variant: str,
    *,
    batch: int = 16,
    runtime_populated: bool = False,
) -> dict[str, object]:
    topology = {
        **COMMON_D64_CAUSAL_MX_TOPOLOGY,
        **VARIANT_TOPOLOGY[variant],
        "batch": batch,
        "valid": int(runtime_populated),
    }
    if runtime_populated:
        topology["logical_jobs"] = batch * 32 * (4096 // 256)
    else:
        topology["physical_grid_ctas"] = 0
        topology["threads_per_cta"] = 0
        topology["logical_jobs"] = 0
    return topology


def test_artifact_authentication_pins_exact_regular_file(tmp_path: Path) -> None:
    artifact = tmp_path / "forward.so"
    payload = b"distinct-unanchored-splitmix-v6-artifact\n"
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    assert authenticate_artifact(
        artifact,
        expected_sha256=digest,
        expected_bytes=len(payload),
    ) == {
        "path": str(artifact.resolve()),
        "sha256": digest,
        "bytes": len(payload),
    }
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        authenticate_artifact(
            artifact,
            expected_sha256="0" * 64,
            expected_bytes=len(payload),
        )
    with pytest.raises(RuntimeError, match="byte-count mismatch"):
        authenticate_artifact(
            artifact,
            expected_sha256=digest,
            expected_bytes=len(payload) + 1,
        )


def test_artifact_authentication_rejects_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "forward.so"
    artifact.write_bytes(b"payload")
    link = tmp_path / "forward-link.so"
    link.symlink_to(artifact)
    with pytest.raises(RuntimeError, match="non-symlink"):
        authenticate_artifact(
            link,
            expected_sha256=hashlib.sha256(b"payload").hexdigest(),
            expected_bytes=7,
        )


def test_unanchored_b16_identity_is_immutable_and_variant_scoped() -> None:
    expected = {
        "module": (
            "_C_cfwd_mx_d4q01_unanchored_splitmix_v6_"
            "b16s4096h32kv8d64_sm100_20260825"
        ),
        "sha256": (
            "93488ece199812bbd001d9e1f662db79a"
            "c39ecc230d91e8f0de2c2e4321976d3"
        ),
        "bytes": 1_958_304,
    }
    assert CANONICAL_ARTIFACTS == {
        ("unanchored-splitmix-v6", 16): expected
    }
    assert canonical_artifact_spec("unanchored-splitmix-v6", 16) == expected
    with pytest.raises(ValueError, match="no canonical artifact"):
        canonical_artifact_spec("unanchored-splitmix-v6", 8)
    with pytest.raises(ValueError, match="no canonical artifact"):
        canonical_artifact_spec("anchored", 16)


@pytest.mark.parametrize("variant", tuple(VARIANT_TOPOLOGY))
def test_prelaunch_topology_authenticates_named_variant(variant: str) -> None:
    require_topology(_topology(variant), variant=variant, batch=16)


@pytest.mark.parametrize(
    "field",
    tuple(
        field
        for field in COMMON_D64_CAUSAL_MX_TOPOLOGY
        if field not in {"physical_grid_ctas", "threads_per_cta"}
    )
    + tuple(VARIANT_TOPOLOGY["unanchored-splitmix-v6"]),
)
def test_unanchored_topology_rejects_every_pinned_field(field: str) -> None:
    topology = _topology("unanchored-splitmix-v6")
    value = topology[field]
    topology[field] = not value if isinstance(value, bool) else "wrong"
    with pytest.raises(RuntimeError, match=field):
        require_topology(
            topology,
            variant="unanchored-splitmix-v6",
            batch=16,
        )


def test_named_variants_cannot_substitute_for_each_other() -> None:
    with pytest.raises(RuntimeError, match="mx_global_anchor32"):
        require_topology(
            _topology("anchored"),
            variant="unanchored-splitmix-v6",
            batch=16,
        )
    with pytest.raises(RuntimeError, match="mx_global_anchor32"):
        require_topology(
            _topology("unanchored-splitmix-v6"),
            variant="anchored",
            batch=16,
        )


def test_unanchored_variant_is_authenticated_only_at_batch_16() -> None:
    with pytest.raises(ValueError, match="no authenticated batch-8"):
        require_topology(
            _topology("unanchored-splitmix-v6", batch=8),
            variant="unanchored-splitmix-v6",
            batch=8,
        )


def test_runtime_topology_requires_valid_populated_launch_geometry() -> None:
    topology = _topology(
        "unanchored-splitmix-v6",
        runtime_populated=True,
    )
    require_topology(
        topology,
        variant="unanchored-splitmix-v6",
        batch=16,
        runtime_populated=True,
    )
    for field, value in (
        ("valid", 0),
        ("logical_jobs", 0),
        ("physical_grid_ctas", 0),
        ("threads_per_cta", 0),
    ):
        candidate = dict(topology)
        candidate[field] = value
        with pytest.raises(RuntimeError, match=field):
            require_topology(
                candidate,
                variant="unanchored-splitmix-v6",
                batch=16,
                runtime_populated=True,
            )

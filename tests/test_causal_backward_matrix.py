from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "benchmark_causal_backward_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("causal_backward_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def test_build_shapes_is_deterministic_and_supports_d64_d128() -> None:
    shapes = MATRIX._build_shapes(
        (512, 4096),
        (64, 128),
        ((32, 8), (64, 16)),
    )
    assert len(shapes) == 8
    assert shapes[0].as_dict() == {
        "batch": 1,
        "sequence": 512,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 64,
        "gqa_ratio": 4,
    }
    assert shapes[-1].head_dim == 128
    assert shapes[-1].sequence == 4096
    assert shapes[-1].q_heads == 64


@pytest.mark.parametrize(
    ("value", "expected"),
    (("32/8", ((32, 8),)), ("16/4,32/8,16/4", ((16, 4), (32, 8)))),
)
def test_parse_head_pairs(value: str, expected: tuple[tuple[int, int], ...]) -> None:
    assert MATRIX._parse_head_pairs(value) == expected


@pytest.mark.parametrize("value", ("32", "32/7", "0/1", "32/8/2"))
def test_parse_head_pairs_rejects_invalid_contracts(value: str) -> None:
    with pytest.raises(ValueError):
        MATRIX._parse_head_pairs(value)


def test_route_registry_does_not_alias_mx_replay_to_retained_lowp() -> None:
    retained = MATRIX.ROUTE_CAPABILITIES["retained_lowp"]
    mx_replay = MATRIX.ROUTE_CAPABILITIES["mx_exact_replay"]
    assert retained["available"] is True
    assert mx_replay["available"] is False
    assert "E8M0" in mx_replay["reason"]
    with pytest.raises(MATRIX.RouteUnavailable):
        MATRIX._build_route("mx_exact_replay", None, None, None, None)


def test_no_direct_tma_route_changes_only_d64_direct_tma_policy() -> None:
    compiled_calls = []
    control_calls = []

    def compiled_backward(control: object, **kwargs: object) -> object:
        compiled_calls.append((control, kwargs))
        return object()

    class Controls:
        def retained_lowp(
            self,
            head_dim: int,
            *,
            direct_tma_dkdv: bool,
            detached_fp8_p_tmem: bool = False,
            shape: object | None = None,
            allow_s4096_precomposed: bool = False,
        ) -> tuple[str, int, bool, bool]:
            control_calls.append(
                (head_dim, direct_tma_dkdv, detached_fp8_p_tmem)
            )
            return (
                "control",
                head_dim,
                direct_tma_dkdv,
                detached_fp8_p_tmem,
            )

        def provenance(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {}

    runtime = SimpleNamespace(compiled_backward=compiled_backward)
    state = SimpleNamespace(
        q_fp8=object(),
        k_fp8=object(),
        v_fp8=object(),
        direct_dpsum=object(),
        dout_fp8=object(),
        direct_lse_log2=object(),
    )
    shape = MATRIX.Shape(8192, 32, 8, 64)
    retained = MATRIX._build_route(
        "retained_lowp", runtime, Controls(), shape, state
    )
    no_direct_tma = MATRIX._build_route(
        "retained_lowp_no_direct_tma", runtime, Controls(), shape, state
    )

    assert control_calls == [(64, True, True), (64, False, True)]
    assert retained.name == "retained_lowp"
    assert no_direct_tma.name == "retained_lowp_no_direct_tma"
    assert retained.decode_scale == no_direct_tma.decode_scale == 0.25
    assert compiled_calls[0][0] == ("control", 64, True, True)
    assert compiled_calls[1][0] == ("control", 64, False, True)
    direct_kwargs = compiled_calls[0][1]
    no_direct_kwargs = compiled_calls[1][1]
    assert direct_kwargs["direct_tma_dkdv"] is True
    assert no_direct_kwargs["direct_tma_dkdv"] is False
    assert {
        key: value
        for key, value in direct_kwargs.items()
        if key not in ("direct_tma_dkdv", "head_fast_raster")
    } == {
        key: value
        for key, value in no_direct_kwargs.items()
        if key not in ("direct_tma_dkdv", "head_fast_raster")
    }
    assert direct_kwargs["head_fast_raster"] is None
    assert no_direct_kwargs["head_fast_raster"] is True
    assert {
        key: value
        for key, value in retained.policy.items()
        if key not in (
            "direct_tma_dkdv",
            "probability_tmem_policy",
            "raster_policy",
        )
    } == {
        key: value
        for key, value in no_direct_tma.policy.items()
        if key not in (
            "direct_tma_dkdv",
            "probability_tmem_policy",
            "raster_policy",
        )
    }
    assert retained.policy["head_fast_raster"] is True
    assert no_direct_tma.policy["head_fast_raster"] is True


@pytest.mark.parametrize(
    ("route_name", "period"),
    (
        ("retained_lowp_exp2_d1_p2", 2),
        ("retained_lowp_exp2_d1_p3", 3),
    ),
)
def test_selective_exp2_routes_change_only_exp2_policy(
    route_name: str,
    period: int,
) -> None:
    compiled_calls = []

    def compiled_backward(control: object, **kwargs: object) -> object:
        compiled_calls.append((control, kwargs))
        return object()

    class Controls:
        def retained_lowp(
            self,
            head_dim: int,
            *,
            direct_tma_dkdv: bool,
            detached_fp8_p_tmem: bool = False,
            shape: object | None = None,
            allow_s4096_precomposed: bool = False,
        ) -> tuple[str, int, bool, bool]:
            return (
                "control",
                head_dim,
                direct_tma_dkdv,
                detached_fp8_p_tmem,
            )

        def provenance(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {}

    runtime = SimpleNamespace(compiled_backward=compiled_backward)
    state = SimpleNamespace(
        q_fp8=object(),
        k_fp8=object(),
        v_fp8=object(),
        direct_dpsum=object(),
        dout_fp8=object(),
        direct_lse_log2=object(),
    )
    shape = MATRIX.Shape(8192, 32, 8, 64)
    native = MATRIX._build_route(
        "retained_lowp_native_exp2", runtime, Controls(), shape, state
    )
    candidate = MATRIX._build_route(
        route_name, runtime, Controls(), shape, state
    )

    assert candidate.name == route_name
    native_kwargs = compiled_calls[0][1]
    candidate_kwargs = compiled_calls[1][1]
    assert native_kwargs["exp2_degree"] == 2
    assert native_kwargs["exp2_period"] == 0
    assert candidate_kwargs["exp2_degree"] == 1
    assert candidate_kwargs["exp2_period"] == period
    assert {
        key: value
        for key, value in native_kwargs.items()
        if key not in ("exp2_degree", "exp2_period", "exp2_policy")
    } == {
        key: value
        for key, value in candidate_kwargs.items()
        if key not in ("exp2_degree", "exp2_period", "exp2_policy")
    }
    assert {
        key: value
        for key, value in native.policy.items()
        if key
        not in (
            "exp2_degree",
            "exp2_period",
            "exp2_policy",
            "probability_tmem_policy",
            "raster_policy",
        )
    } == {
        key: value
        for key, value in candidate.policy.items()
        if key
        not in (
            "exp2_degree",
            "exp2_period",
            "exp2_policy",
            "probability_tmem_policy",
            "raster_policy",
        )
    }
    assert native.policy["head_fast_raster"] is True
    assert candidate.policy["head_fast_raster"] is True


@pytest.mark.parametrize(
    ("sequence", "expected", "head_fast", "detached_p"),
    (
        (2048, (2, 0), False, False),
        (4096, (1, 2), False, False),
        (8192, (1, 2), True, True),
        (16384, (1, 2), True, True),
    ),
)
def test_retained_d64_route_uses_sequence_dispatch(
    sequence: int,
    expected: tuple[int, int],
    head_fast: bool,
    detached_p: bool,
) -> None:
    policy = MATRIX._planned_route_policy(
        "retained_lowp",
        MATRIX.Shape(sequence, 32, 8, 64),
    )
    assert policy is not None
    assert (policy["exp2_degree"], policy["exp2_period"]) == expected
    assert policy["exp2_policy"]["effective"] == {
        "degree": expected[0],
        "period": expected[1],
    }
    assert policy["head_fast_raster"] is head_fast
    assert policy["raster_policy"]["effective_head_fast"] is head_fast
    assert policy["detached_fp8_p_tmem"] is detached_p
    assert policy["probability_tmem_policy"]["effective_detached"] is detached_p


def test_retained_d128_route_uses_generated_shared_probability_policy() -> None:
    policy = MATRIX._planned_route_policy(
        "retained_lowp",
        MATRIX.Shape(4096, 32, 8, 128),
    )
    assert policy is not None
    assert policy["direct_tma_dkdv"] is False
    assert policy["detached_fp8_p_tmem"] is False
    assert policy["probability_storage"] == "shared_coordinate_preserving_128b"
    assert policy["reuse_quantized_p"] is True
    assert policy["fp8_ds_lift"] == 256
    assert policy["lowp_do_stages"] == 2
    assert (policy["exp2_degree"], policy["exp2_period"]) == (1, 0)
    assert policy["head_fast_raster"] is False
    assert policy["workspace_stats"] is False


def test_control_cache_builds_d128_from_generated_shared_probability() -> None:
    load_calls = []

    def load_control(**kwargs: object) -> tuple[tuple[str, object], ...]:
        load_calls.append(kwargs)
        return tuple(sorted(kwargs.items()))

    controls = MATRIX.ControlCache(SimpleNamespace(load_control=load_control))
    shape = MATRIX.Shape(4096, 32, 8, 128)
    control = controls.retained_lowp(
        128,
        direct_tma_dkdv=False,
        detached_fp8_p_tmem=False,
        shape=shape,
        allow_s4096_precomposed=True,
    )
    assert control is controls.retained_lowp(
        128,
        direct_tma_dkdv=False,
        detached_fp8_p_tmem=False,
        shape=shape,
        allow_s4096_precomposed=True,
    )
    assert load_calls == [
        {
            "fp8_p_storage": "shared",
            "direct_tma_dkdv": False,
            "detached_fp8_p_tmem": False,
        }
    ]


@pytest.mark.parametrize("sequence", (8192, 16384))
def test_verified_no_direct_tma_control_inherits_retained_raster(
    sequence: int,
) -> None:
    policy = MATRIX._planned_route_policy(
        "retained_lowp_no_direct_tma",
        MATRIX.Shape(sequence, 32, 8, 64),
    )
    assert policy is not None
    assert policy["direct_tma_dkdv"] is False
    assert policy["head_fast_raster"] is True
    assert policy["raster_policy"]["mode"] == "explicit"
    assert "comparison_control" in policy["raster_policy"]


@pytest.mark.parametrize("sequence", (8192, 16384))
def test_verified_detached_p_control_inherits_retained_raster(
    sequence: int,
) -> None:
    policy = MATRIX._planned_route_policy(
        "retained_lowp_detached_p",
        MATRIX.Shape(sequence, 32, 8, 64),
    )
    assert policy is not None
    assert policy["detached_fp8_p_tmem"] is True
    assert policy["head_fast_raster"] is True
    assert policy["raster_policy"]["mode"] == "explicit"
    assert "comparison_control" in policy["raster_policy"]


@pytest.mark.parametrize("sequence", (8192, 16384))
def test_verified_alias_p_control_inherits_retained_raster(
    sequence: int,
) -> None:
    policy = MATRIX._planned_route_policy(
        "retained_lowp_alias_p",
        MATRIX.Shape(sequence, 32, 8, 64),
    )
    assert policy is not None
    assert policy["detached_fp8_p_tmem"] is False
    assert policy["head_fast_raster"] is True
    assert policy["probability_tmem_policy"]["mode"] == "explicit"


def test_control_cache_keeps_direct_tma_variants_distinct() -> None:
    load_calls = []

    def load_control(**kwargs: object) -> tuple[tuple[str, object], ...]:
        load_calls.append(kwargs)
        return tuple(sorted(kwargs.items()))

    controls = MATRIX.ControlCache(SimpleNamespace(load_control=load_control))
    direct = controls.retained_lowp(64, direct_tma_dkdv=True)
    no_direct = controls.retained_lowp(64, direct_tma_dkdv=False)

    assert direct != no_direct
    assert controls.retained_lowp(64, direct_tma_dkdv=True) is direct
    assert controls.retained_lowp(64, direct_tma_dkdv=False) is no_direct
    assert load_calls == [
        {
            "fp8_p_storage": "tmem",
            "direct_tma_dkdv": True,
            "detached_fp8_p_tmem": False,
        },
        {
            "fp8_p_storage": "tmem",
            "direct_tma_dkdv": False,
            "detached_fp8_p_tmem": False,
        },
    ]


def test_control_cache_keeps_detached_p_variant_distinct() -> None:
    load_calls = []

    def load_control(**kwargs: object) -> tuple[tuple[str, object], ...]:
        load_calls.append(kwargs)
        return tuple(sorted(kwargs.items()))

    controls = MATRIX.ControlCache(SimpleNamespace(load_control=load_control))
    alias = controls.retained_lowp(64, direct_tma_dkdv=True)
    detached = controls.retained_lowp(
        64,
        direct_tma_dkdv=True,
        detached_fp8_p_tmem=True,
    )
    assert alias != detached
    assert len(load_calls) == 2
    assert load_calls[1]["detached_fp8_p_tmem"] is True


def test_authenticated_control_is_exact_s4096_only_and_long_routes_generated(
    tmp_path: Path,
) -> None:
    precomposed_source = tmp_path / "s4096_control.py"
    precomposed_source.write_text("CONTROL = 's4096'\n")
    generated_source = tmp_path / "generated_control.py"
    generated_source.write_text("CONTROL = 'generated'\n")
    payload = precomposed_source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    spec = MATRIX.PrecomposedControlSpec(
        source=precomposed_source.resolve(),
        sha256=digest,
        size_bytes=len(payload),
    )
    load_calls = []

    def load_control(**kwargs: object) -> object:
        load_calls.append(kwargs)
        if "precomposed_control_source" in kwargs:
            return SimpleNamespace(
                __file__=precomposed_source,
                TK_PRECOMPOSED_CONTROL_PROVENANCE={
                    "mode": "precomposed",
                    "source": {
                        "path": str(precomposed_source.resolve()),
                        "sha256": digest,
                        "bytes": len(payload),
                    },
                },
            )
        return SimpleNamespace(__file__=generated_source)

    controls = MATRIX.ControlCache(
        SimpleNamespace(load_control=load_control),
        spec,
    )
    shape_4096 = MATRIX.Shape(4096, 32, 8, 64)
    shape_8192 = MATRIX.Shape(8192, 32, 8, 64)
    shape_16384 = MATRIX.Shape(16384, 32, 8, 64)
    s4096 = controls.retained_lowp(
        64,
        direct_tma_dkdv=True,
        detached_fp8_p_tmem=False,
        shape=shape_4096,
        allow_s4096_precomposed=True,
    )
    s8192 = controls.retained_lowp(
        64,
        direct_tma_dkdv=True,
        detached_fp8_p_tmem=True,
        shape=shape_8192,
        allow_s4096_precomposed=True,
    )
    s16384 = controls.retained_lowp(
        64,
        direct_tma_dkdv=True,
        detached_fp8_p_tmem=True,
        shape=shape_16384,
        allow_s4096_precomposed=True,
    )

    assert s8192 is s16384
    assert load_calls[0]["precomposed_control_source"] == spec.source
    assert load_calls[0]["precomposed_control_sha256"] == spec.sha256
    assert load_calls[0]["precomposed_control_bytes"] == spec.size_bytes
    assert load_calls[1] == {
        "fp8_p_storage": "tmem",
        "direct_tma_dkdv": True,
        "detached_fp8_p_tmem": True,
    }
    assert controls.provenance(
        s4096,
        shape=shape_4096,
        consumer="retained_lowp",
    )["mode"] == "precomposed"
    long_provenance = controls.provenance(
        s8192,
        shape=shape_8192,
        consumer="retained_lowp",
    )
    assert long_provenance["mode"] == "generated_patch_chain"
    assert long_provenance["construction"]["detached_fp8_p_tmem"] is True
    assert long_provenance["binding"]["shape"]["sequence"] == 8192
    assert long_provenance["generated_source"] == {
        "bytes": generated_source.stat().st_size,
        "sha256": hashlib.sha256(generated_source.read_bytes()).hexdigest(),
    }


def test_precomposed_control_spec_authenticates_without_import(
    tmp_path: Path,
) -> None:
    source = tmp_path / "control.py"
    source.write_text("raise AssertionError('must not import during planning')\n")
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    spec = MATRIX._resolve_precomposed_control_spec(
        source,
        digest,
        len(payload),
    )
    assert spec == MATRIX.PrecomposedControlSpec(
        source=source.resolve(),
        sha256=digest,
        size_bytes=len(payload),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MATRIX._resolve_precomposed_control_spec(
            source,
            "0" * 64,
            len(payload),
        )


def test_build_retained_rejects_detached_p_policy_mismatch() -> None:
    backward = SimpleNamespace(
        head_fast_raster=True,
        detached_fp8_p_tmem=False,
    )
    runtime = SimpleNamespace(
        compiled_backward=lambda control, **kwargs: backward
    )

    class Controls:
        def retained_lowp(
            self,
            head_dim: int,
            *,
            direct_tma_dkdv: bool,
            detached_fp8_p_tmem: bool = False,
            shape: object | None = None,
            allow_s4096_precomposed: bool = False,
        ) -> object:
            return object()

        def provenance(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {}

    state = SimpleNamespace(
        q_fp8=object(),
        k_fp8=object(),
        v_fp8=object(),
        direct_dpsum=object(),
        dout_fp8=object(),
        direct_lse_log2=object(),
    )
    with pytest.raises(RuntimeError, match="P placement disagrees"):
        MATRIX._build_retained_lowp(
            runtime,
            Controls(),
            MATRIX.Shape(8192, 32, 8, 64),
            state,
        )


@pytest.mark.parametrize(
    "route",
    (
        "retained_lowp_no_direct_tma",
        "retained_lowp_alias_p",
        "retained_lowp_detached_p",
        "retained_lowp_native_exp2",
        "retained_lowp_exp2_d1_p2",
        "retained_lowp_exp2_d1_p3",
    ),
)
def test_d64_comparison_routes_reject_d128(route: str) -> None:
    shapes = (MATRIX.Shape(4096, 32, 8, 128),)
    with pytest.raises(ValueError, match="supports only head dimensions 64"):
        MATRIX._validate_route_shape_support((route,), shapes)


def test_detached_p_controls_are_verified_shape_only() -> None:
    shapes = (MATRIX.Shape(4096, 32, 8, 64),)
    with pytest.raises(ValueError, match="supports only S/Hq/Hkv shapes"):
        MATRIX._validate_route_shape_support(
            ("retained_lowp_alias_p", "retained_lowp_detached_p"),
            shapes,
        )


@pytest.mark.parametrize("sequence", (8192, 16384))
def test_detached_p_controls_accept_verified_shapes(sequence: int) -> None:
    MATRIX._validate_route_shape_support(
        ("retained_lowp_alias_p", "retained_lowp_detached_p"),
        (MATRIX.Shape(sequence, 32, 8, 64),),
    )


def test_dry_run_records_d64_direct_tma_ab_without_cuda(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--dry-run",
            "--sequences",
            "4096,8192,16384",
            "--head-dims",
            "64",
            "--head-pairs",
            "32/8",
            "--routes",
            "retained_lowp,retained_lowp_no_direct_tma",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(output.read_text())
    assert document["status"] == "planned"
    assert document["schema"] == "fp4_fa4_causal_backward_matrix_v3"
    assert document["device"] is None
    assert document["requested_routes"] == [
        "retained_lowp",
        "retained_lowp_no_direct_tma",
    ]
    assert [shape["sequence"] for shape in document["planned_shapes"]] == [
        4096,
        8192,
        16384,
    ]
    assert [
        (
            case["shape"]["sequence"],
            case["route"],
            case["route_policy"]["exp2_degree"],
            case["route_policy"]["exp2_period"],
        )
        for case in document["planned_cases"]
    ] == [
        (4096, "retained_lowp", 1, 2),
        (4096, "retained_lowp_no_direct_tma", 1, 2),
        (8192, "retained_lowp", 1, 2),
        (8192, "retained_lowp_no_direct_tma", 1, 2),
        (16384, "retained_lowp", 1, 2),
        (16384, "retained_lowp_no_direct_tma", 1, 2),
    ]
    assert document["protocol"]["backward_exp2_dispatch"] == {
        "default": {"degree": 2, "period": 0},
        "scope": "D64 B1 causal retained lowp",
        "threshold_sequence": 4096,
        "verified_shapes": [
            {"kv_heads": 4, "q_heads": 16, "sequence": 4096},
            {"kv_heads": 8, "q_heads": 32, "sequence": 4096},
            {"kv_heads": 16, "q_heads": 64, "sequence": 4096},
            {"kv_heads": 8, "q_heads": 32, "sequence": 8192},
            {"kv_heads": 8, "q_heads": 32, "sequence": 16384},
        ],
        "verified_shape_policy": {"degree": 1, "period": 2},
        "version": MATRIX.BACKWARD_EXP2_POLICY_VERSION,
    }
    assert document["protocol"]["backward_raster_dispatch"][
        "verified_shapes"
    ] == [
        {"kv_heads": 8, "q_heads": 32, "sequence": 8192},
        {"kv_heads": 8, "q_heads": 32, "sequence": 16384},
    ]
    assert document["protocol"]["backward_probability_tmem_dispatch"][
        "verified_shapes"
    ] == [
        {"kv_heads": 8, "q_heads": 32, "sequence": 8192},
        {"kv_heads": 8, "q_heads": 32, "sequence": 16384},
    ]
    control = document["route_capabilities"][
        "retained_lowp_no_direct_tma"
    ]
    assert control["supported_head_dims"] == [64]
    assert control["only_policy_change"] == {
        "compiled_backward_direct_tma_dkdv": [True, False],
        "control_module_direct_tma_dkdv": [True, False],
    }
    assert json.loads(completed.stdout) == document


def test_dry_run_scopes_authenticated_control_away_from_8k_16k(
    tmp_path: Path,
) -> None:
    control = tmp_path / "s4096_control.py"
    control.write_text("CONTROL = 1\n")
    payload = control.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--dry-run",
            "--sequences",
            "4096,8192,16384",
            "--head-dims",
            "64",
            "--head-pairs",
            "32/8",
            "--routes",
            "retained_lowp",
            "--backward-control-source",
            str(control),
            "--backward-control-sha256",
            digest,
            "--backward-control-bytes",
            str(len(payload)),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(output.read_text())
    cases = {
        case["shape"]["sequence"]: case
        for case in document["planned_cases"]
    }
    assert cases[4096]["control_plan"]["route"]["mode"] == "precomposed"
    for sequence in (8192, 16384):
        case = cases[sequence]
        assert (
            case["control_plan"]["route"]["mode"]
            == "generated_patch_chain"
        )
        assert (
            "precomposed_control_not_selected"
            in case["control_plan"]["route"]
        )
        assert case["route_policy"]["detached_fp8_p_tmem"] is True
        assert case["route_policy"]["head_fast_raster"] is True
    manifest = document["protocol"]["backward_control_provenance"]
    assert manifest["authenticated_s4096_control"]["source"] == {
        "path": str(control.resolve()),
        "sha256": digest,
        "bytes": len(payload),
    }


def test_summary_reports_interpolated_percentiles_and_samples() -> None:
    summary = MATRIX._summary((1.0, 2.0, 3.0, 4.0, 5.0))
    assert summary["median_us"] == 3.0
    assert summary["p05_us"] == pytest.approx(1.2)
    assert summary["p95_us"] == pytest.approx(4.8)
    assert summary["samples_us"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_multi_seed_case_compiles_once_and_refreshes_bound_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    state = SimpleNamespace(seed=11)
    reference = MATRIX.BuiltRoute(
        name="cute_bf16",
        backward=object(),
        decode_scale=1.0,
        policy={"kind": "reference"},
        control_provenance={"mode": "generated"},
    )
    route = MATRIX.BuiltRoute(
        name="retained_lowp",
        backward=object(),
        decode_scale=0.25,
        policy={"workspace_stats": True},
        control_provenance={"mode": "generated"},
    )

    class FakeCuda:
        @staticmethod
        def reset_peak_memory_stats() -> None:
            events.append(("reset_peak",))

        @staticmethod
        def max_memory_allocated() -> int:
            return 101

        @staticmethod
        def max_memory_reserved() -> int:
            return 202

    runtime = SimpleNamespace(torch=SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.setattr(
        MATRIX,
        "_make_state",
        lambda runtime, shape, seed: (
            events.append(("make", seed)) or state
        ),
    )
    monkeypatch.setattr(
        MATRIX,
        "_build_bf16",
        lambda runtime, controls, shape, state: (
            events.append(("compile_bf16", state.seed)) or reference
        ),
    )
    monkeypatch.setattr(
        MATRIX,
        "_build_route",
        lambda name, runtime, controls, shape, state: (
            events.append(("compile_route", name, state.seed)) or route
        ),
    )

    def refresh(runtime: object, shape: object, bound: object, seed: int) -> None:
        events.append(("refresh", seed))
        bound.seed = seed

    monkeypatch.setattr(MATRIX, "_refresh_state_in_place", refresh)
    monkeypatch.setattr(
        MATRIX,
        "_publish_workspace_statistics",
        lambda shape, bound, built: events.append(
            ("publish_workspace", bound.seed, built.name)
        ),
    )
    monkeypatch.setattr(
        MATRIX,
        "_accuracy",
        lambda runtime, reference, route: {
            "aggregate": {"cosine": float(state.seed)}
        },
    )
    monkeypatch.setattr(
        MATRIX,
        "_timing",
        lambda *args, **kwargs: {"timed": True},
    )
    shape = MATRIX.Shape(4096, 32, 8, 64)

    first, compiled = MATRIX._run_shape_seed(
        runtime,
        object(),
        shape,
        "retained_lowp",
        11,
        compiled_case=None,
        measure_timing=True,
        warmups=0,
        samples=1,
    )
    second, reused = MATRIX._run_shape_seed(
        runtime,
        object(),
        shape,
        "retained_lowp",
        22,
        compiled_case=compiled,
        measure_timing=False,
        warmups=0,
        samples=1,
    )

    assert reused is compiled
    assert first["compiled_specialization_reused"] is False
    assert second["compiled_specialization_reused"] is True
    assert first["timing"] == {"timed": True}
    assert second["timing"] is None
    assert [event for event in events if event[0].startswith("compile_")] == [
        ("compile_bf16", 11),
        ("compile_route", "retained_lowp", 11),
    ]
    assert ("refresh", 22) in events
    assert ("publish_workspace", 22, "retained_lowp") in events


def test_workspace_statistics_are_republished_for_reused_d64_route() -> None:
    shape = MATRIX.Shape(128, 2, 1, 64)
    stats_numel = shape.batch * shape.q_heads * shape.sequence
    dpsum = torch.arange(stats_numel, dtype=torch.float32).view(
        shape.batch,
        shape.q_heads,
        1,
        shape.sequence,
    )
    lse = (1000 + torch.arange(stats_numel, dtype=torch.float32)).view_as(
        dpsum
    )
    workspace = torch.zeros(2 * stats_numel * 4, dtype=torch.uint8)
    route = MATRIX.BuiltRoute(
        name="retained_lowp",
        backward=SimpleNamespace(workspace_torch=workspace),
        decode_scale=0.25,
        policy={"workspace_stats": True},
        control_provenance={},
    )
    MATRIX._publish_workspace_statistics(
        shape,
        SimpleNamespace(direct_dpsum=dpsum, direct_lse_log2=lse),
        route,
    )
    published = workspace.view(torch.float32)
    assert torch.equal(published[:stats_numel], dpsum.reshape(-1))
    assert torch.equal(published[stats_numel:], lse.reshape(-1))


def test_memory_estimate_grows_with_shape() -> None:
    small = MATRIX.Shape(512, 32, 8, 64)
    large = MATRIX.Shape(8192, 32, 8, 128)
    assert MATRIX._estimated_live_tensor_bytes(large) > (
        16 * MATRIX._estimated_live_tensor_bytes(small)
    )

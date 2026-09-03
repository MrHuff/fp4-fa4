from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
SATURATED = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_saturated.py"
)
PATCH = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "d128_gqa_mxfp4_v_dp.patch"


def _function_namespace(
    path: Path,
    function_name: str,
    **globals_: object,
) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _method_namespace(
    path: Path,
    class_name: str,
    method_name: str,
    **globals_: object,
) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _eligible_config(**overrides: int) -> SimpleNamespace:
    fields = {
        "batch": 1,
        "sequence": 4096,
        "hidden": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _eligibility_kwargs() -> dict[str, object]:
    return {
        "experimental_native_nvfp4_projection_out": True,
        "qkv_projection_format": "nvfp4",
        "publish_mxfp4_v": True,
        "backward_match_forward_operands": False,
        "per_block_qk_scales": True,
        "experimental_split_v_backward": False,
        "v_mxfp4_scale_2d": False,
    }


def test_single_mx_v_runtime_gate_accepts_only_authenticated_geometry() -> None:
    namespace = _function_namespace(
        E2E,
        "_native_d128_mxfp4_v_backward_eligible",
    )
    eligible = namespace["_native_d128_mxfp4_v_backward_eligible"]
    assert callable(eligible)
    assert eligible(_eligible_config(batch=1), **_eligibility_kwargs())
    assert eligible(_eligible_config(batch=2), **_eligibility_kwargs())

    for config, replacement in (
        (_eligible_config(batch=3), {}),
        (_eligible_config(sequence=2048), {}),
        (_eligible_config(head_dim=64), {}),
        (
            _eligible_config(),
            {"experimental_native_nvfp4_projection_out": False},
        ),
        (_eligible_config(), {"qkv_projection_format": "e4m3"}),
        (_eligible_config(), {"publish_mxfp4_v": False}),
        (_eligible_config(), {"per_block_qk_scales": False}),
    ):
        kwargs = _eligibility_kwargs()
        kwargs.update(replacement)
        assert not eligible(config, **kwargs)

    shared_tile = _eligibility_kwargs()
    shared_tile["v_mxfp4_scale_2d"] = True
    assert eligible(_eligible_config(batch=2), **shared_tile)


def test_runtime_wires_producer_control_consumer_and_autograd_publication() -> None:
    source = E2E.read_text()
    runtime = source.split("class LowpAttentionRuntime:", 1)[1].split(
        "def _run_lowp_forward_attention", 1
    )[0]
    assert "experimental_d128_mxfp4_v_backward: bool = False" in runtime
    assert "and experimental_output_shared_split_v is not False" in runtime
    assert "experimental_mx_backward_v=(" in runtime
    assert "NativeTkD128SharedTileProducerV503Backward" in runtime
    assert "backward_kwargs[\"producer\"] = self.qkv_projection" in runtime
    assert "self.experimental_d128_mxfp4_v_backward" in runtime
    assert "use_d128_mxfp4_v_dp=(" in runtime
    assert "v_mxfp4_scale_pages=v_mxfp4_scale_pages" in runtime
    assert "producer_workspace=producer_workspace" in runtime
    assert "dtype=torch.uint8" in runtime
    assert "self.backward.bind_d128_mxfp4_v_operands(" in runtime

    autograd = source.split(
        "class _LowpAttentionFunction(torch.autograd.Function):", 1
    )[1].split("class LowpAttention(nn.Module):", 1)[0]
    assert "saved_v_operands = qkv.mxfp4_backward_v_operands(" in autograd
    assert "required_scale_policy=(" in autograd
    assert "saved_v_operands = (qkv.v_backward_fp8,)" in autograd
    assert "v_mxfp4_scale_pages=v_mxfp4_scale_pages" in autograd
    assert "expected_v_operand_count" in autograd

    workspace = source.split("def _allocate_forward_workspace", 1)[1].split(
        "def _apply", 1
    )[0]
    assert "v_backward_mxfp4 = torch.empty(" in workspace
    assert "v_backward_mxfp4_scale_pages = torch.empty(" in workspace
    assert "v_backward_mxfp4=v_backward_mxfp4" in workspace


def test_native_d128_runtime_gate_keeps_v501_and_v503_disjoint() -> None:
    namespace = _function_namespace(
        E2E,
        "_require_native_tk_d128_runtime_contract",
        AUTHENTICATED_D128_EXACT_BATCHES=(2,),
    )
    require = namespace["_require_native_tk_d128_runtime_contract"]
    common = {
        "projection_dgrad": "nvfp4",
        "qkv_projection_format": "nvfp4",
        "experimental_native_nvfp4_projection_out": True,
        "backward_reuse_quantized_p": False,
        "backward_forward_mx_probability_replay": False,
        "backward_forward_mx_probability_scale_handoff": False,
        "backward_match_forward_operands": False,
        "per_block_qk_scales": True,
        "experimental_split_v_backward": False,
        "backward_probability_correction": 1.0,
        "q_quant_scale": 2.25,
        "k_quant_scale": 2.0,
        "projection_weight_scale_2d": True,
        "v_mxfp4_scale_2d": False,
        "adaptive_qk_weight_scales": False,
        "shared_runtime": None,
    }
    mx_topology = {
        "pv_format": "mxfp4_e8m0_block32",
        "causal_interleaved_kv": False,
    }
    fp8_topology = {
        "pv_format": "e4m3_fp8",
        "shiftless_fp8_mode": 0,
        "causal_interleaved_kv": False,
    }

    require(
        _eligible_config(batch=2),
        mx_topology,
        experimental_d128_mxfp4_v_backward=True,
        **common,
    )
    require(
        _eligible_config(batch=1),
        fp8_topology,
        experimental_d128_mxfp4_v_backward=False,
        **common,
    )
    with pytest.raises(ValueError, match="batch 2"):
        require(
            _eligible_config(batch=1),
            mx_topology,
            experimental_d128_mxfp4_v_backward=True,
            **common,
        )
    with pytest.raises(ValueError, match="MXFP4-PV forward"):
        require(
            _eligible_config(batch=2),
            fp8_topology,
            experimental_d128_mxfp4_v_backward=True,
            **common,
        )

    shared_tile_common = {
        **common,
        "v_mxfp4_scale_2d": True,
        "shared_runtime": SimpleNamespace(native_tk_d128_backward=True),
    }
    with pytest.raises(ValueError, match="without cross-runtime"):
        require(
            _eligible_config(batch=2),
            mx_topology,
            experimental_d128_mxfp4_v_backward=True,
            **shared_tile_common,
        )


def test_native_runtime_binds_mx_payload_and_scale_pages_in_v503_abi() -> None:
    method = _method_namespace(
        E2E,
        "LowpAttentionRuntime",
        "bind_backward_inputs",
        MXFP4_V_SCALE_POLICY_SHARED_D32XS32="shared_d32xs32",
    )["bind_backward_inputs"]
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(
        native_tk_backward=True,
        native_tk_d128_backward=True,
        native_tk_d128_native_score_backward=False,
        experimental_d128_mxfp4_v_backward=True,
        d128_mxfp4_v_scale_policy="rowwise_d32",
        backward=SimpleNamespace(
            bind_inputs=lambda *operands: calls.append(operands)
        ),
    )
    q, k, v, scales, dout = (object() for _ in range(5))

    method(runtime, q, k, v, dout, v_mxfp4_scale_pages=scales)

    assert calls == [(q, k, v, scales, dout)]
    with pytest.raises(RuntimeError, match="E8M0 scale pages"):
        method(runtime, q, k, v, dout)

    with pytest.raises(ValueError, match="legacy rowwise"):
        method(
            runtime,
            q,
            k,
            v,
            dout,
            v_mxfp4_scale_pages=scales,
            producer_workspace=object(),
        )


def test_native_runtime_shared_tile_route_requires_and_forwards_workspace() -> None:
    method = _method_namespace(
        E2E,
        "LowpAttentionRuntime",
        "bind_backward_inputs",
        MXFP4_V_SCALE_POLICY_SHARED_D32XS32="shared_d32xs32",
    )["bind_backward_inputs"]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def bind_inputs(
        *operands: object,
        **kwargs: object,
    ) -> None:
        calls.append((operands, kwargs))

    runtime = SimpleNamespace(
        native_tk_backward=True,
        native_tk_d128_backward=True,
        native_tk_d128_native_score_backward=False,
        experimental_d128_mxfp4_v_backward=True,
        d128_mxfp4_v_scale_policy="shared_d32xs32",
        backward=SimpleNamespace(bind_inputs=bind_inputs),
    )
    q, k, v, scales, dout, workspace = (object() for _ in range(6))

    with pytest.raises(RuntimeError, match="authenticated projection"):
        method(runtime, q, k, v, dout, v_mxfp4_scale_pages=scales)
    assert calls == []

    method(
        runtime,
        q,
        k,
        v,
        dout,
        v_mxfp4_scale_pages=scales,
        producer_workspace=workspace,
    )
    assert calls == [
        (
            (q, k, v, scales, dout),
            {"producer_workspace": workspace},
        )
    ]


def test_native_score_runtime_requires_and_forwards_exact_workspace() -> None:
    method = _method_namespace(
        E2E,
        "LowpAttentionRuntime",
        "bind_backward_inputs",
        MXFP4_V_SCALE_POLICY_SHARED_D32XS32="shared_d32xs32",
    )["bind_backward_inputs"]
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(
        native_tk_backward=True,
        native_tk_d128_backward=True,
        native_tk_d128_native_score_backward=True,
        experimental_d128_mxfp4_v_backward=False,
        d128_mxfp4_v_scale_policy=None,
        backward=SimpleNamespace(
            bind_inputs=lambda *operands: calls.append(operands)
        ),
    )
    q, k, v, dout, workspace = (object() for _ in range(5))

    with pytest.raises(RuntimeError, match="exact forward publication"):
        method(runtime, q, k, v, dout)
    assert calls == []

    method(
        runtime,
        q,
        k,
        v,
        dout,
        native_score_workspace=workspace,
    )
    assert calls == [(q, k, v, dout, workspace)]

    with pytest.raises(ValueError, match="retained E4M3 V"):
        method(
            runtime,
            q,
            k,
            v,
            dout,
            v_mxfp4_scale_pages=object(),
            native_score_workspace=workspace,
        )
    with pytest.raises(ValueError, match="MX producer workspace"):
        method(
            runtime,
            q,
            k,
            v,
            dout,
            producer_workspace=object(),
            native_score_workspace=workspace,
        )


def test_saturated_cli_propagates_and_reports_explicit_opt_in() -> None:
    source = SATURATED.read_text()
    parser_source = source.split(
        "parser = argparse.ArgumentParser", 1
    )[1]
    cli = parser_source.split(
        '"--experimental-d128-mxfp4-v-backward"', 1
    )[1].split("parser.add_argument", 1)[0]
    assert 'action="store_true"' in cli
    assert "experimental_d128_mxfp4_v_backward=(" in source
    assert "args.experimental_d128_mxfp4_v_backward," in source
    assert '"experimental_d128_mxfp4_v_backward": bool(' in source
    assert "if args.backward_control is not None:" in source.split(
        "if args.experimental_d128_mxfp4_v_backward:", 1
    )[1].split('if is_d128 and args.route == "mx_unanchored"', 1)[0]


def test_saturated_selector_rejects_competing_publication_and_wrong_route() -> None:
    namespace = _function_namespace(
        SATURATED,
        "_require_saturated_projection_selection",
        LOWP_ROUTES=("fp8", "mx"),
        EXPERIMENTAL_NATIVE_NVFP4_ROUTES=("fp8", "mx"),
    )
    select = namespace["_require_saturated_projection_selection"]
    assert callable(select)
    select("mx", "nvfp4", True, False, False, "caller_declared", True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        select("mx", "nvfp4", True, False, True, "caller_declared", True)
    with pytest.raises(ValueError, match="D128 MX"):
        select("fp8", "nvfp4", True, False, False, "caller_declared", True)


def test_repository_control_latch_remains_fail_closed() -> None:
    source = PATCH.read_text()
    assert "+        self.d128_mxfp4_v_dp_compiled = False" in source
    assert "+        self.d128_mxfp4_v_dp_compiled = True" not in source


def test_saturated_patch_artifact_is_candidate_only_and_contract_bound() -> None:
    namespace = _function_namespace(
        SATURATED,
        "_d128_mxfp4_v_dp_patch_artifact",
        Path=Path,
        string=__import__("string"),
    )
    artifact = namespace["_d128_mxfp4_v_dp_patch_artifact"]
    receipt = {
        "path": "/tmp/d128_candidate.patch",
        "sha256": "a" * 64,
        "bytes": 123,
    }
    candidate = SimpleNamespace(
        experimental_d128_mxfp4_v_backward=True,
        d128_mxfp4_v_dp_patch_provenance=receipt,
        backward_contract=lambda: {
            "control": {"d128_mxfp4_v_dp_patch": dict(receipt)}
        },
    )
    assert artifact(candidate) == receipt
    retained = SimpleNamespace(
        experimental_d128_mxfp4_v_backward=False,
        d128_mxfp4_v_dp_patch_provenance=None,
    )
    assert artifact(retained) is None
    native = SimpleNamespace(
        experimental_d128_mxfp4_v_backward=True,
        native_tk_d128_backward=True,
        d128_mxfp4_v_dp_patch_provenance=None,
    )
    assert artifact(native) is None
    native.d128_mxfp4_v_dp_patch_provenance = receipt
    with pytest.raises(RuntimeError, match="CuTe patch receipt"):
        artifact(native)
    candidate.backward_contract = lambda: {"control": {}}
    with pytest.raises(RuntimeError, match="backward contract"):
        artifact(candidate)

    source = SATURATED.read_text()
    assert 'artifacts["d128_mxfp4_v_dp_patch"]' in source
    comparison_identity = source.split("comparison_identity = {", 1)[1].split(
        "samples: dict[str, Any]", 1
    )[0]
    assert "d128_mxfp4_v_dp_patch" not in comparison_identity


def test_native_saturated_contract_accepts_only_b2_mx_v503_publication() -> None:
    namespace = _function_namespace(
        SATURATED,
        "_require_native_tk_d128_saturated_runtime",
        NATIVE_TK_D128_BACKEND="native_tk_d128_e4m3",
        NATIVE_TK_D128_MX_BACKEND=(
            "native_tk_d128_rowscale_mxfp4_v"
        ),
        NATIVE_TK_D128_SHARED_TILE_MX_BACKEND=(
            "native_tk_d128_shared_tile_mxfp4_v_v503_consumer"
        ),
        tk_interface=SimpleNamespace(
            MXFP4_V_SCALE_POLICY_ROWWISE_D32=(
                "rowwise_independent_d32_anchors"
            ),
            MXFP4_V_SCALE_POLICY_SHARED_D32XS32=(
                "shared_d32xs32_forward_anchors"
            ),
        ),
    )
    require = namespace["_require_native_tk_d128_saturated_runtime"]
    loaded_image = {
        "path": "/tmp/v503.so",
        "sha256": "5" * 64,
        "bytes": 503,
    }
    publication = {
        "qkv_projection_format": "nvfp4",
        "forward_pv_format": "mxfp4_e8m0_block32",
        "represented_backward": False,
        "per_block_qk_scales": True,
        "qk_backward_source": "projection_accumulator_e4m3",
        "v_backward_source": "rowwise_width6_mxfp4_v",
        "experimental_split_v_backward": False,
        "experimental_d128_mxfp4_v_backward": True,
        "d128_mxfp4_v_scale_policy": (
            "rowwise_independent_d32_anchors"
        ),
    }
    backend = {
        "backend": "native_tk_d128_rowscale_mxfp4_v",
        "extension": loaded_image,
        "input": {
            "dtype": "mixed_e4m3fn_and_packed_mxfp4_e8m0",
            "layout": "BSHD_contiguous_with_physical_scale_pages",
        },
    }
    projection = {
        "qk_backward_source": "projection_accumulator_e4m3",
        "v_backward_source": "rowwise_width6_mxfp4_v",
        "dout_backward_source": "projection_accumulator_e4m3",
        "experimental_d128_mxfp4_v_backward": True,
        "d128_mxfp4_v_scale_policy": (
            "rowwise_independent_d32_anchors"
        ),
    }
    runtime = SimpleNamespace(
        experimental_d128_mxfp4_v_backward=True,
        d128_mxfp4_v_scale_policy=(
            "rowwise_independent_d32_anchors"
        ),
        v_mxfp4_scale_2d=False,
        native_tk_d128_backward=True,
        qkv_projection_format="nvfp4",
        projection_dgrad="nvfp4",
        pv_format="mxfp4_e8m0_block32",
        experimental_split_v_backward=False,
        backward_match_forward_operands=False,
        per_block_qk_scales=True,
        projection_publication_topology=publication,
        forward_topology={
            "pv_format": "mxfp4_e8m0_block32",
            "causal_interleaved_kv": False,
        },
        native_tk_d128_backward_extension_identity=loaded_image,
        backward=SimpleNamespace(contract=lambda: backend),
        backward_contract=lambda: {
            "backend": backend,
            "projection": projection,
        },
    )
    config = _eligible_config(batch=2)

    require("mx", config, runtime, {"loaded_image": loaded_image})

    runtime.d128_mxfp4_v_scale_policy = (
        "shared_d32xs32_forward_anchors"
    )
    runtime.v_mxfp4_scale_2d = True
    publication["v_backward_source"] = (
        "shared_d32xs32_forward_anchor_mxfp4_v"
    )
    publication["d128_mxfp4_v_scale_policy"] = (
        "shared_d32xs32_forward_anchors"
    )
    backend["backend"] = (
        "native_tk_d128_shared_tile_mxfp4_v_v503_consumer"
    )
    projection["v_backward_source"] = (
        "shared_d32xs32_forward_anchor_mxfp4_v"
    )
    projection["d128_mxfp4_v_scale_policy"] = (
        "shared_d32xs32_forward_anchors"
    )
    require("mx", config, runtime, {"loaded_image": loaded_image})

    with pytest.raises(ValueError, match="B2"):
        require(
            "mx",
            _eligible_config(batch=1),
            runtime,
            {"loaded_image": loaded_image},
        )


def test_operand_cache_receipt_is_candidate_only_and_nonsemantic() -> None:
    method = _method_namespace(
        E2E,
        "LowpAttentionRuntime",
        "d128_mxfp4_v_operand_cache_receipt",
    )["d128_mxfp4_v_operand_cache_receipt"]
    receipt = {
        "schema": "d128_mxfp4_v_operand_cache_v1",
        "capacity": 32,
        "entries": 32,
        "hits": 64,
    }
    candidate = SimpleNamespace(
        backward=SimpleNamespace(
            d128_mxfp4_v_operand_cache_receipt=lambda: receipt
        ),
        experimental_d128_mxfp4_v_backward=True,
    )
    assert method(candidate) is receipt

    retained = SimpleNamespace(
        backward=SimpleNamespace(
            d128_mxfp4_v_operand_cache_receipt=lambda: None
        ),
        experimental_d128_mxfp4_v_backward=False,
    )
    assert method(retained) is None
    retained.backward.d128_mxfp4_v_operand_cache_receipt = lambda: receipt
    with pytest.raises(RuntimeError, match="retained backward"):
        method(retained)

    e2e_source = E2E.read_text()
    e2e_result = e2e_source.split("result = {", 1)[1].split(
        "encoded = json.dumps", 1
    )[0]
    assert '"d128_mxfp4_v_operand_cache"' in e2e_result
    assert "if runtime.experimental_d128_mxfp4_v_backward" in e2e_result
    backward_contract = e2e_source.split(
        "def backward_contract(self)", 1
    )[1].split("def bind_backward_inputs", 1)[0]
    assert "d128_mxfp4_v_operand_cache" not in backward_contract

    saturated_source = SATURATED.read_text()
    saturated_result = saturated_source.split("result = {", 2)[-1].split(
        "encoded = json.dumps", 1
    )[0]
    assert '"d128_mxfp4_v_operand_cache"' in saturated_result
    assert "and runtime.experimental_d128_mxfp4_v_backward" in saturated_result
    comparison_identity = saturated_source.split(
        "comparison_identity = {", 1
    )[1].split("samples: dict[str, Any]", 1)[0]
    assert "d128_mxfp4_v_operand_cache" not in comparison_identity


def test_compiler_receipt_is_candidate_only_and_published_as_artifact() -> None:
    method = _method_namespace(
        E2E,
        "LowpAttentionRuntime",
        "d128_mxfp4_v_compilation_receipt",
    )["d128_mxfp4_v_compilation_receipt"]
    receipt = {
        "schema": "fa4_d128_mxfp4_v_cute_compile_v1",
        "ptx": {"sha256": "a" * 64, "bytes": 100},
        "cubin": {"sha256": "b" * 64, "bytes": 200},
    }
    candidate = SimpleNamespace(
        backward=SimpleNamespace(
            d128_mxfp4_v_compilation_receipt=lambda: receipt
        ),
        experimental_d128_mxfp4_v_backward=True,
    )
    assert method(candidate) is receipt

    retained = SimpleNamespace(
        backward=SimpleNamespace(
            d128_mxfp4_v_compilation_receipt=lambda: None
        ),
        experimental_d128_mxfp4_v_backward=False,
    )
    assert method(retained) is None
    retained.backward.d128_mxfp4_v_compilation_receipt = lambda: receipt
    with pytest.raises(RuntimeError, match="retained backward"):
        method(retained)

    saturated_source = SATURATED.read_text()
    assert 'artifacts["d128_mxfp4_v_compilation"]' in saturated_source
    comparison_identity = saturated_source.split(
        "comparison_identity = {", 1
    )[1].split("samples: dict[str, Any]", 1)[0]
    assert "d128_mxfp4_v_compilation" not in comparison_identity


class _ReceiptTensor:
    def __init__(self, address: int, shape: tuple[int, ...]) -> None:
        self._address = address
        self.shape = shape
        self.dtype = "uint8"
        self.device = "cuda:0"

    def data_ptr(self) -> int:
        return self._address

    def numel(self) -> int:
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def element_size(self) -> int:
        return 1


def test_workspace_receipt_adds_only_present_candidate_owners() -> None:
    owner_slots = {"base": 4}
    optional_slots = {
        "v_backward_mxfp4": 24,
        "v_backward_mxfp4_scale_pages": 25,
    }
    globals_ = {
        "_FORWARD_WORKSPACE_OWNER_SLOTS": owner_slots,
        "_FORWARD_WORKSPACE_ACTIVE_ROUTES": {
            "base": ("mxfp4_e8m0_block32",)
        },
        "_FORWARD_WORKSPACE_ALIAS_OWNERS": {},
        "_FORWARD_WORKSPACE_SENTINELS": (),
        "_FORWARD_WORKSPACE_OPTIONAL_OWNERS": tuple(optional_slots),
        "_FORWARD_WORKSPACE_OPTIONAL_OWNER_SLOTS": optional_slots,
        "_FORWARD_WORKSPACE_COMMON_ROUTES": (
            "mxfp4_e8m0_block32",
            "e4m3_fp8",
        ),
        "_forward_workspace_owner_tensors": (
            lambda workspace: (("base", workspace.outputs.base),)
        ),
        "_forward_workspace_all_tensors": (
            lambda workspace: tuple(
                (name, tensor)
                for name in ("base", *optional_slots)
                if (tensor := getattr(workspace.outputs, name)) is not None
            )
        ),
        "_d128_dual_qkv_weight_tensors": lambda workspace: (),
        "_dual_output_weight_tensors": lambda workspace: (),
        "_uses_direct_d128_dual_qkv_weight_prep": lambda runtime: False,
        "_uses_direct_dual_output_weight_prep": lambda runtime: False,
    }
    method = _method_namespace(
        E2E,
        "LowpAttention",
        "forward_workspace_contract",
        **globals_,
    )["forward_workspace_contract"]

    base = _ReceiptTensor(0x1000, (4,))
    payload = _ReceiptTensor(0x2000, (8,))
    scales = _ReceiptTensor(0x3000, (16,))

    def run(
        optional_payload: _ReceiptTensor | None,
        optional_scales: _ReceiptTensor | None,
    ) -> dict[str, object]:
        outputs = SimpleNamespace(
            base=base,
            v_mxfp4_scale_pages=base,
            v_backward_mxfp4=optional_payload,
            v_backward_mxfp4_scale_pages=optional_scales,
        )
        tensors = (base, optional_payload, optional_scales)
        workspace = SimpleNamespace(
            outputs=outputs,
            allocation_data_ptrs={
                name: tensor.data_ptr()
                for name, tensor in zip(
                    ("base", *optional_slots),
                    tensors,
                    strict=True,
                )
                if tensor is not None
            }
            | {"v_mxfp4_scale_pages": base.data_ptr()},
            publication_state=SimpleNamespace(
                current_generation=0,
                in_flight_generation=None,
            ),
            cuda_stream=7,
            d128_dual_qkv_weight_authenticated=False,
            d128_dual_qkv_weight_abi_identity=None,
            output_dual_weight_authenticated=False,
            output_dual_weight_abi_identity=None,
        )
        self = SimpleNamespace(
            _forward_workspace=workspace,
            runtime=SimpleNamespace(
                qkv_projection=None,
                config=SimpleNamespace(batch=2),
                experimental_native_nvfp4_projection_out=True,
                pv_format="mxfp4_e8m0_block32",
            ),
            named_buffers=lambda recurse=True: (),
            named_parameters=lambda recurse=True: (),
        )
        return method(self)

    retained = run(None, None)
    assert retained["publication_slots"] == [4]
    assert set(retained["owners"]) == {"base"}
    assert retained["owner_count"] == 1
    assert retained["total_owner_bytes"] == 4

    candidate = run(payload, scales)
    assert candidate["publication_slots"] == [4, 24, 25]
    assert set(candidate["owners"]) == {
        "base",
        "v_backward_mxfp4",
        "v_backward_mxfp4_scale_pages",
    }
    assert candidate["owner_count"] == 3
    assert candidate["total_owner_bytes"] == 28
    assert candidate["owner_pointers_stable_since_allocation"] is True

from __future__ import annotations

import ast
import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute as tune_d64_gqa_cute
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _require_d128_mxfp4_v_dp_runtime_capability,
    _require_d128_mxfp4_v_dp_tensor_abi,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import (
    _load_control,
    _require_d128_mxfp4_v_dp_patch_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "profile_gqa_d128_chain.py"


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: str = "cuda:0"
    is_cuda: bool = True
    contiguous: bool = True
    address: int = 0x1000

    def is_contiguous(self) -> bool:
        return self.contiguous

    def data_ptr(self) -> int:
        return self.address

    def stride(self) -> tuple[int, ...]:
        strides: list[int] = []
        running = 1
        for extent in reversed(self.shape):
            strides.append(running)
            running *= extent
        return tuple(reversed(strides))

    def storage_offset(self) -> int:
        return 0


def _tensor_abi(batch: int = 2) -> dict[str, object]:
    return {
        "batch": batch,
        "sequence": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "q": _FakeTensor(
            (batch, 4096, 32, 128),
            torch.float8_e4m3fn,
        ),
        "k": _FakeTensor(
            (batch, 4096, 8, 128),
            torch.float8_e4m3fn,
        ),
        "v_payload": _FakeTensor((batch, 4096, 8, 64), torch.uint8),
        "v_scale_pages": _FakeTensor(
            (batch, 32, 8, 512),
            torch.uint8,
        ),
        "dout": _FakeTensor(
            (batch, 4096, 32, 128),
            torch.float8_e4m3fn,
        ),
    }


def _compiled_keywords(*, depth: int = 128, sequence: int = 4096) -> dict[str, object]:
    batch = 1
    return {
        "q": _FakeTensor((batch, sequence, 32, depth), torch.float8_e4m3fn),
        "k": _FakeTensor((batch, sequence, 8, depth), torch.float8_e4m3fn),
        "v": _FakeTensor((batch, sequence, 8, depth // 2), torch.uint8),
        "o_or_sum": _FakeTensor((batch, 32, 1, sequence), torch.float32),
        "dout": _FakeTensor(
            (batch, sequence, 32, depth),
            torch.float8_e4m3fn,
        ),
        "lse_or_scaled_lse": _FakeTensor(
            (batch, 32, 1, sequence),
            torch.float32,
        ),
        "q_heads": 32,
        "kv_heads": 8,
        "lowp": True,
        "precomputed_stats": True,
        "scale_softmax": (depth**-0.5) / 16.0,
        "exp2_degree": 1,
        "exp2_period": 0,
        "reuse_quantized_p": True,
        "lowp_do_stages": 2,
        "use_d128_mxfp4_v_dp": True,
        "v_mxfp4_scale_pages": _FakeTensor(
            (batch, sequence // 128, 8, 512),
            torch.uint8,
        ),
    }


def _mx_control(
    enabled: bool,
    *,
    mixed_mma_capability: bool = True,
) -> SimpleNamespace:
    tcgen05 = SimpleNamespace()
    if mixed_mma_capability:
        tcgen05.MmaMXF8F6F4Op = object
    return SimpleNamespace(
        TK_D128_MXFP4_V_DP=enabled,
        TK_FP8_P_STORAGE="shared",
        TK_DETACHED_FP8_P_TMEM=False,
        TK_DIRECT_TMA_DKDV=False,
        TK_PRECOMPOSED_CONTROL_PROVENANCE=None,
        tcgen05=tcgen05,
    )


def test_exact_mxfp4_v_tensor_abi_accepts_b1_and_b2() -> None:
    _require_d128_mxfp4_v_dp_tensor_abi(**_tensor_abi(batch=1))
    _require_d128_mxfp4_v_dp_tensor_abi(**_tensor_abi(batch=2))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("v_scale_pages", None, "physical V scale pages"),
        (
            "v_payload",
            _FakeTensor((2, 4096, 8, 128), torch.uint8),
            "packed V must have shape",
        ),
        (
            "v_scale_pages",
            _FakeTensor((2, 32, 8, 512), torch.float8_e4m3fn),
            "V scale pages must have dtype",
        ),
        (
            "dout",
            _FakeTensor((2, 4096, 32, 128), torch.float8_e4m3fn, is_cuda=False),
            "dO must be a CUDA tensor",
        ),
        (
            "v_payload",
            _FakeTensor((2, 4096, 8, 64), torch.uint8, address=0x1010),
            "packed V must be at least 32-byte aligned",
        ),
        (
            "v_scale_pages",
            _FakeTensor((2, 32, 8, 512), torch.uint8, address=0x1008),
            "V scale pages must be at least 16-byte aligned",
        ),
    ),
)
def test_mxfp4_v_tensor_abi_fails_closed(
    field: str,
    replacement: object,
    message: str,
) -> None:
    operands = _tensor_abi()
    operands[field] = replacement
    with pytest.raises(ValueError, match=message):
        _require_d128_mxfp4_v_dp_tensor_abi(**operands)


def test_runtime_flag_must_match_control_marker() -> None:
    keywords = _compiled_keywords()
    with pytest.raises(ValueError, match="must match the loaded CuTe control"):
        CompiledGqaBackward(_mx_control(False), **keywords)

    keywords["use_d128_mxfp4_v_dp"] = False
    keywords["v_mxfp4_scale_pages"] = None
    with pytest.raises(ValueError, match="must match the loaded CuTe control"):
        CompiledGqaBackward(_mx_control(True), **keywords)


def test_mixed_mma_runtime_capability_fails_closed() -> None:
    _require_d128_mxfp4_v_dp_runtime_capability(_mx_control(True))
    with pytest.raises(RuntimeError, match="MmaMXF8F6F4Op"):
        CompiledGqaBackward(
            _mx_control(True, mixed_mma_capability=False),
            **_compiled_keywords(),
        )


def test_packed_v_and_scale_pages_cannot_enter_retained_route() -> None:
    keywords = _compiled_keywords()
    keywords["use_d128_mxfp4_v_dp"] = False
    keywords["v_mxfp4_scale_pages"] = None
    with pytest.raises(ValueError, match="packed MXFP4 V requires"):
        CompiledGqaBackward(_mx_control(False), **keywords)


@pytest.mark.parametrize(
    ("depth", "sequence"),
    ((64, 4096), (128, 2048)),
)
def test_mxfp4_v_route_rejects_unverified_geometry(
    depth: int,
    sequence: int,
) -> None:
    with pytest.raises(ValueError, match="gated to the B1/B2 S4096"):
        CompiledGqaBackward(
            _mx_control(True),
            **_compiled_keywords(depth=depth, sequence=sequence),
        )


def test_mxfp4_v_route_rejects_missing_scale_pages() -> None:
    keywords = _compiled_keywords()
    keywords["v_mxfp4_scale_pages"] = None
    with pytest.raises(ValueError, match="physical V scale pages"):
        CompiledGqaBackward(_mx_control(True), **keywords)


def test_control_loader_rejects_nondefault_mx_composition() -> None:
    with pytest.raises(ValueError, match="generated shared-P control"):
        _load_control(
            fp8_p_storage="tmem",
            use_d128_mxfp4_v_dp=True,
        )


def test_control_loader_conditionally_composes_mx_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Path] = {}
    toolchain_receipt = {
        "schema": "fa4_d128_mxfp4_v_cutlass_toolchain_receipt_v1",
        "test_double": True,
    }

    class _Loader:
        def exec_module(self, module: object) -> None:
            return None

    def fake_spec_from_file_location(name: str, source: Path) -> object:
        del name
        captured["source"] = Path(source)
        return SimpleNamespace(loader=_Loader())

    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        fake_spec_from_file_location,
    )
    monkeypatch.setattr(
        importlib.util,
        "module_from_spec",
        lambda spec: SimpleNamespace(__file__=str(captured["source"])),
    )
    monkeypatch.setattr(
        tune_d64_gqa_cute,
        "require_d128_mxfp4_v_compile_environment",
        lambda: {"CUTE_DSL_KEEP": "ptx,cubin"},
    )
    monkeypatch.setattr(
        tune_d64_gqa_cute,
        "verify_d128_mxfp4_v_toolchain",
        lambda: toolchain_receipt,
    )

    control = _load_control(use_d128_mxfp4_v_dp=True)
    composed = captured["source"].read_text()
    assert control.TK_D128_MXFP4_V_DP is True
    assert control.TK_D128_MXFP4_V_TOOLCHAIN_PROVENANCE is toolchain_receipt
    patch = PROFILE.with_name("d128_gqa_mxfp4_v_dp.patch")
    assert control.TK_D128_MXFP4_V_DP_PATCH_PROVENANCE == {
        "path": str(patch.resolve()),
        "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "bytes": patch.stat().st_size,
    }
    assert control.TK_FP8_P_STORAGE == "shared"
    assert "TK_D128_MXFP4_DP_RAW_TO_X16 = 2.0 / 3.0" in composed
    assert "Float4E2M1FN,\n            Float8E4M3FN," in composed
    assert "internal_type=Uint8" in composed
    assert "d128_mx_dO_ready" not in composed
    assert "mxfp4_v_scale_iter: Any = None" in composed
    assert "cute.arch.fma_packed_f32x2" in composed
    assert "Defer this proof until immediately before the next score" in composed
    assert "Each of the 256 score consumers first completes its own" in composed


def test_retained_control_never_authenticates_experimental_toolchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Loader:
        def exec_module(self, module: object) -> None:
            return None

    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda name, source: SimpleNamespace(loader=_Loader()),
    )
    monkeypatch.setattr(
        importlib.util,
        "module_from_spec",
        lambda spec: SimpleNamespace(__file__="retained_control.py"),
    )

    def forbidden() -> object:
        raise AssertionError("retained control entered experimental verifier")

    monkeypatch.setattr(
        tune_d64_gqa_cute,
        "require_d128_mxfp4_v_compile_environment",
        forbidden,
    )
    monkeypatch.setattr(
        tune_d64_gqa_cute,
        "verify_d128_mxfp4_v_toolchain",
        forbidden,
    )

    control = _load_control(use_d128_mxfp4_v_dp=False)
    assert control.TK_D128_MXFP4_V_DP is False
    assert control.TK_D128_MXFP4_V_TOOLCHAIN_PROVENANCE is None


@pytest.mark.parametrize(
    "provenance",
    (
        None,
        {},
        {"path": "/tmp/candidate.patch", "sha256": "0" * 63, "bytes": 1},
        {"path": "relative.patch", "sha256": "0" * 64, "bytes": 1},
        {"path": "/tmp/candidate.patch", "sha256": "0" * 64, "bytes": 0},
    ),
)
def test_candidate_patch_provenance_fails_closed(provenance: object) -> None:
    control = SimpleNamespace(
        TK_D128_MXFP4_V_DP=True,
        TK_D128_MXFP4_V_DP_PATCH_PROVENANCE=provenance,
    )
    with pytest.raises(RuntimeError, match="patch|provenance|SHA256"):
        _require_d128_mxfp4_v_dp_patch_provenance(control, enabled=True)


def test_retained_control_rejects_candidate_patch_provenance() -> None:
    control = SimpleNamespace(
        TK_D128_MXFP4_V_DP=False,
        TK_D128_MXFP4_V_DP_PATCH_PROVENANCE={
            "path": "/tmp/candidate.patch",
            "sha256": "0" * 64,
            "bytes": 1,
        },
    )
    with pytest.raises(RuntimeError, match="unexpectedly carries"):
        _require_d128_mxfp4_v_dp_patch_provenance(control, enabled=False)


def test_compiled_argument_abi_is_append_compatible() -> None:
    tree = ast.parse(PROFILE.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CompiledGqaBackward"
    )
    init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "arguments"
            for target in node.targets
        )
        and isinstance(node.value, ast.Tuple)
    ]
    assert len(assignments) == 1
    assert len(assignments[0].value.elts) == 23
    appends = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute)
        and node.target.attr == "arguments"
    ]
    assert len(appends) == 1
    assert isinstance(appends[0].value, ast.Tuple)
    assert len(appends[0].value.elts) == 1
    assert isinstance(appends[0].value.elts[0], ast.Name)
    assert appends[0].value.elts[0].id == "mxfp4_v_scale_iter"

    source = PROFILE.read_text()
    assert "self.dv = torch.empty_like(k, dtype=torch.bfloat16)" in source
    assert "v_cute = control.from_dlpack(v, assumed_align=32)" in source
    assert "v_mxfp4_scale_pages,\n                assumed_align=16" in source
    assert "v_cute.element_type = control.cutlass.Float4E2M1FN" in source
    assert "v_scale_cute.element_type = control.cutlass.Float8E8M0FNU" in source


def test_mxfp4_v_rebind_updates_both_arguments_and_retains_owners() -> None:
    class _CuteOperand:
        def __init__(self, tensor: object) -> None:
            self.tensor = tensor
            self.element_type = None
            self.iterator = ("iterator", tensor)

    class _Control:
        cutlass = SimpleNamespace(
            Float4E2M1FN=object(),
            Float8E8M0FNU=object(),
        )

        @staticmethod
        def from_dlpack(tensor: object, *, assumed_align: int) -> _CuteOperand:
            assert assumed_align in (16, 32)
            return _CuteOperand(tensor)

    compiled = object.__new__(CompiledGqaBackward)
    compiled.kernel = SimpleNamespace(use_d128_mxfp4_v_dp=True)
    compiled._control = _Control()
    abi = _tensor_abi(batch=2)
    compiled._d128_mxfp4_v_dp_validation_tensors = (
        abi["q"],
        abi["k"],
        abi["dout"],
    )
    compiled._d128_mxfp4_v_scale_argument_index = 23
    compiled.arguments = tuple(range(24))
    compiled.v_mxfp4_payload = None
    compiled.v_mxfp4_scale_pages = None
    compiled._initialize_d128_mxfp4_v_operand_cache(enabled=True)

    payload = abi["v_payload"]
    scales = abi["v_scale_pages"]
    compiled.bind_d128_mxfp4_v_operands(payload, scales)

    assert compiled.arguments[3].tensor is payload
    assert compiled.arguments[3].element_type is _Control.cutlass.Float4E2M1FN
    assert compiled.arguments[23] == ("iterator", scales)
    assert compiled.v_mxfp4_payload is payload
    assert compiled.v_mxfp4_scale_pages is scales


def _cache_test_compiled(
    control: object,
    *,
    batch: int = 2,
) -> tuple[CompiledGqaBackward, dict[str, object]]:
    abi = _tensor_abi(batch=batch)
    compiled = object.__new__(CompiledGqaBackward)
    compiled.kernel = SimpleNamespace(use_d128_mxfp4_v_dp=True)
    compiled._control = control
    compiled._d128_mxfp4_v_dp_validation_tensors = (
        abi["q"],
        abi["k"],
        abi["dout"],
    )
    compiled._d128_mxfp4_v_scale_argument_index = 23
    compiled.arguments = tuple(range(24))
    compiled.v_mxfp4_payload = None
    compiled.v_mxfp4_scale_pages = None
    compiled._initialize_d128_mxfp4_v_operand_cache(enabled=True)
    return compiled, abi


def test_mxfp4_v_rebind_cache_reuses_exact_wrappers_and_reports_hits() -> None:
    class _CuteOperand:
        def __init__(self, tensor: object) -> None:
            self.tensor = tensor
            self.element_type = None
            self.iterator = ("iterator", tensor)

    class _Control:
        cutlass = SimpleNamespace(
            Float4E2M1FN=object(),
            Float8E8M0FNU=object(),
        )
        calls: list[tuple[object, int]] = []

        @classmethod
        def from_dlpack(
            cls,
            tensor: object,
            *,
            assumed_align: int,
        ) -> _CuteOperand:
            cls.calls.append((tensor, assumed_align))
            return _CuteOperand(tensor)

    compiled, abi = _cache_test_compiled(_Control())
    payload = abi["v_payload"]
    scales = abi["v_scale_pages"]
    compiled.bind_d128_mxfp4_v_operands(payload, scales)
    first_payload_wrapper = compiled.arguments[3]
    first_scale_iterator = compiled.arguments[23]
    # Q/K/dO here are constructor-only compile placeholders. Runtime rebinding
    # authenticates the live Q/K/dO operands separately, so a cache hit must
    # not repeat this stale placeholder validation.
    static_q = compiled._d128_mxfp4_v_dp_validation_tensors[0]
    object.__setattr__(static_q, "shape", (2, 4096, 31, 128))
    compiled.bind_d128_mxfp4_v_operands(payload, scales)

    assert _Control.calls == [(payload, 32), (scales, 16)]
    assert compiled.arguments[3] is first_payload_wrapper
    assert compiled.arguments[23] is first_scale_iterator
    cache = compiled._d128_mxfp4_v_operand_cache
    assert cache is not None
    entry = cache[(id(payload), id(scales))]
    assert entry["v_payload"] is payload
    assert entry["v_scale_pages"] is scales
    assert compiled.d128_mxfp4_v_operand_cache_receipt() == {
        "schema": "d128_mxfp4_v_operand_cache_v1",
        "capacity": 32,
        "entries": 1,
        "hits": 1,
        "misses": 1,
        "full_validations": 1,
        "dlpack_wrapper_builds": 2,
        "invalidations": 0,
        "evictions": 0,
        "key_contract": "exact_tensor_identity_pointer_and_view_abi",
        "strong_reference_owners": True,
        "static_constructor_q_k_do_revalidated_on_hit": False,
        "live_q_k_do_rebind_path_unchanged": True,
    }


def test_mxfp4_v_rebind_cache_misses_for_new_exact_objects() -> None:
    class _CuteOperand:
        def __init__(self, tensor: object) -> None:
            self.element_type = None
            self.iterator = ("iterator", tensor)

    class _Control:
        cutlass = SimpleNamespace(
            Float4E2M1FN=object(),
            Float8E8M0FNU=object(),
        )
        calls = 0

        @classmethod
        def from_dlpack(
            cls,
            tensor: object,
            *,
            assumed_align: int,
        ) -> _CuteOperand:
            del assumed_align
            cls.calls += 1
            return _CuteOperand(tensor)

    compiled, abi = _cache_test_compiled(_Control())
    compiled.bind_d128_mxfp4_v_operands(
        abi["v_payload"], abi["v_scale_pages"]
    )
    replacement_payload = _FakeTensor(
        tuple(abi["v_payload"].shape),
        abi["v_payload"].dtype,
        address=abi["v_payload"].address,
    )
    replacement_scales = _FakeTensor(
        tuple(abi["v_scale_pages"].shape),
        abi["v_scale_pages"].dtype,
        address=abi["v_scale_pages"].address,
    )
    compiled.bind_d128_mxfp4_v_operands(
        replacement_payload,
        replacement_scales,
    )

    receipt = compiled.d128_mxfp4_v_operand_cache_receipt()
    assert receipt is not None
    assert receipt["entries"] == 2
    assert receipt["hits"] == 0
    assert receipt["misses"] == 2
    assert receipt["full_validations"] == 2
    assert receipt["dlpack_wrapper_builds"] == 4
    assert _Control.calls == 4


def test_mxfp4_v_rebind_cache_invalidates_changed_pointer_and_fails_closed(
) -> None:
    class _CuteOperand:
        def __init__(self, tensor: object) -> None:
            self.element_type = None
            self.iterator = ("iterator", tensor)

    class _Control:
        cutlass = SimpleNamespace(
            Float4E2M1FN=object(),
            Float8E8M0FNU=object(),
        )

        @staticmethod
        def from_dlpack(
            tensor: object,
            *,
            assumed_align: int,
        ) -> _CuteOperand:
            del assumed_align
            return _CuteOperand(tensor)

    compiled, abi = _cache_test_compiled(_Control())
    payload = abi["v_payload"]
    scales = abi["v_scale_pages"]
    compiled.bind_d128_mxfp4_v_operands(payload, scales)
    object.__setattr__(payload, "address", 0x2000)
    compiled.bind_d128_mxfp4_v_operands(payload, scales)
    receipt = compiled.d128_mxfp4_v_operand_cache_receipt()
    assert receipt is not None
    assert receipt["invalidations"] == 1
    assert receipt["misses"] == 2
    assert receipt["full_validations"] == 2

    object.__setattr__(payload, "shape", (2, 4096, 8, 128))
    with pytest.raises(ValueError, match="packed V must have shape"):
        compiled.bind_d128_mxfp4_v_operands(payload, scales)
    receipt = compiled.d128_mxfp4_v_operand_cache_receipt()
    assert receipt is not None
    assert receipt["invalidations"] == 2
    assert receipt["misses"] == 3
    assert receipt["full_validations"] == 3
    assert receipt["entries"] == 0


def test_mxfp4_v_rebind_cache_is_bounded_to_decoder_layer_count() -> None:
    class _CuteOperand:
        def __init__(self, tensor: object) -> None:
            self.element_type = None
            self.iterator = ("iterator", tensor)

    class _Control:
        cutlass = SimpleNamespace(
            Float4E2M1FN=object(),
            Float8E8M0FNU=object(),
        )

        @staticmethod
        def from_dlpack(
            tensor: object,
            *,
            assumed_align: int,
        ) -> _CuteOperand:
            del assumed_align
            return _CuteOperand(tensor)

    compiled, abi = _cache_test_compiled(_Control())
    payload_shape = tuple(abi["v_payload"].shape)
    scale_shape = tuple(abi["v_scale_pages"].shape)
    for index in range(33):
        compiled.bind_d128_mxfp4_v_operands(
            _FakeTensor(
                payload_shape,
                torch.uint8,
                address=0x2000 + index * 0x100,
            ),
            _FakeTensor(
                scale_shape,
                torch.uint8,
                address=0x100000 + index * 0x100,
            ),
        )

    receipt = compiled.d128_mxfp4_v_operand_cache_receipt()
    assert receipt is not None
    assert receipt["capacity"] == 32
    assert receipt["entries"] == 32
    assert receipt["misses"] == 33
    assert receipt["evictions"] == 1


def test_retained_route_has_no_mxfp4_v_operand_cache_receipt() -> None:
    compiled = object.__new__(CompiledGqaBackward)
    compiled._initialize_d128_mxfp4_v_operand_cache(enabled=False)
    assert compiled.d128_mxfp4_v_operand_cache_receipt() is None

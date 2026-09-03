from __future__ import annotations

from typing import Any

import pytest

from tk_fa4.lowp_fa4_bwd import profile_llama12b_saturated as profiler


def test_unselected_update_does_not_touch_cuda_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        profiler.torch.cuda.profiler,
        "start",
        lambda: calls.append("start"),
    )
    monkeypatch.setattr(
        profiler.torch.cuda.profiler,
        "stop",
        lambda: calls.append("stop"),
    )

    result = profiler._run_profiled_update(
        lambda **_kwargs: {"result": 1},
        profile=False,
    )

    assert result == {"result": 1}
    assert calls == []


def test_selected_update_is_bracketed_even_when_it_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        profiler.torch.cuda.profiler,
        "start",
        lambda: calls.append("start"),
    )
    monkeypatch.setattr(
        profiler.torch.cuda.profiler,
        "stop",
        lambda: calls.append("stop"),
    )

    def fail(**_kwargs: Any) -> dict[str, Any]:
        calls.append("update")
        raise RuntimeError("probe failure")

    with pytest.raises(RuntimeError, match="probe failure"):
        profiler._run_profiled_update(fail, profile=True)

    assert calls == ["start", "update", "stop"]


def test_wrapper_enables_nested_stage_ranges_and_calls_benchmark_main() -> None:
    source = profiler.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()

    assert "runtime_module._PROFILE_STAGE_RANGES = True" in text
    assert "runtime_module._PROFILE_STAGE_NVTX = True" in text
    assert "benchmark.MINIMUM_MEASURED_UPDATES = 1" in text
    assert "benchmark._timed_update = profiled" in text
    assert "benchmark.main()" in text

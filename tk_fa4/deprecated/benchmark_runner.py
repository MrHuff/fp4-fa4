#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PYTHON = Path("/workspace/codebases/fp4_matmul/.venv/bin/python")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "benchmark.py"
REQUIRED_PYTHONPATH = (
    str(REPO_ROOT),
    str(REPO_ROOT / "flash-attention"),
)

PREFLIGHT_CODE = """
from __future__ import annotations

import os
import sys
from pathlib import Path


def fail(message: str, exit_code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


python_path = Path(sys.executable)
if not python_path.exists():
    fail(
        "Configured TK_FA4_PYTHON does not exist.\\n"
        f"Python: {python_path}"
    )

try:
    from flash_attn.cute.interface import flash_attn_func as _cute_flash_attn_func
except Exception as exc:
    fail(
        "flash_attn.cute import failed for the selected benchmark Python.\\n"
        "Use a CuTe-enabled environment, or override TK_FA4_PYTHON.\\n"
        f"Python: {python_path}\\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

try:
    from tk_fa4 import flash_attn_func as _tk_flash_attn_func
except Exception as exc:
    fail(
        "tk_fa4 import failed for the selected benchmark Python.\\n"
        "Build the extension for that interpreter, or override TK_FA4_PYTHON.\\n"
        f"Python: {python_path}\\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

try:
    import torch
except Exception as exc:
    fail(
        "PyTorch import failed for the selected benchmark Python.\\n"
        f"Python: {python_path}\\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

if not torch.cuda.is_available():
    fail(
        "CUDA is not available in the selected benchmark environment.\\n"
        "If you are running from the sandbox, rerun this launcher outside the sandbox or via an approved host command.\\n"
        f"Python: {python_path}\\n"
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )

capability = torch.cuda.get_device_capability()
if capability != (10, 0):
    fail(
        "tk_fa4 benchmarks require GB200 / SM100.\\n"
        f"Python: {python_path}\\n"
        f"Visible capability: {capability}"
    )
"""


def _build_pythonpath(existing: str | None) -> str:
    entries: list[str] = list(REQUIRED_PYTHONPATH)
    if existing:
        entries.extend(item for item in existing.split(os.pathsep) if item)
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry not in seen:
            deduped.append(entry)
            seen.add(entry)
    return os.pathsep.join(deduped)


def _benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(env.get("PYTHONPATH"))
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _selected_python() -> Path:
    raw = os.environ.get("TK_FA4_PYTHON")
    return Path(raw).expanduser() if raw else DEFAULT_PYTHON


def _run_checked(command: list[str], env: dict[str, str]) -> int:
    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


def main() -> int:
    python_path = _selected_python()
    if not python_path.exists():
        print(
            "Configured TK_FA4_PYTHON does not exist.\n"
            f"Python: {python_path}",
            file=sys.stderr,
        )
        return 2
    if not BENCHMARK_SCRIPT.exists():
        print(
            "Benchmark entrypoint does not exist.\n"
            f"Path: {BENCHMARK_SCRIPT}",
            file=sys.stderr,
        )
        return 2

    env = _benchmark_env()
    preflight_cmd = [str(python_path), "-c", PREFLIGHT_CODE]
    status = _run_checked(preflight_cmd, env)
    if status != 0:
        return status

    bench_cmd = [str(python_path), str(BENCHMARK_SCRIPT), *sys.argv[1:]]
    return _run_checked(bench_cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())

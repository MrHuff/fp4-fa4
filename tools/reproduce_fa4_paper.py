#!/usr/bin/env python3
"""Preflight and rebuild the FP4 FlashAttention paper from frozen receipts.

This command deliberately separates two meanings of "reproduce":

* ``offline-generator`` tasks deterministically regenerate summaries, tables,
  figures, or the PDF from files committed to this repository.
* ``receipt-only`` and ``external-only`` tasks describe the measurements that
  supplied those files.  They are visible to ``--list`` and ``--check``, but
  this command never runs them.

No registry command launches a GPU benchmark, a training job, a downloader,
or a service client.  Commands and inputs are repository-relative so the
checkout can be moved without editing the reproduction recipe.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = "paper-pdf"
RUNNABLE = "offline-generator"
NON_RUNNABLE = frozenset({"receipt-only", "external-only"})


@dataclass(frozen=True)
class InputSpec:
    """A required repository-relative input or glob."""

    pattern: str
    minimum: int = 1
    exact: int | None = None
    description: str = ""


@dataclass(frozen=True)
class Task:
    """One artifact-generation stage or one non-runnable evidence boundary."""

    name: str
    classification: str
    description: str
    inputs: tuple[InputSpec, ...] = ()
    outputs: tuple[str, ...] = ()
    command: tuple[str, ...] | None = None
    cwd: str = "."
    dependencies: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    python_modules: tuple[str, ...] = ()
    note: str = ""

    @property
    def runnable(self) -> bool:
        return self.classification == RUNNABLE and self.command is not None


def _input(
    pattern: str,
    *,
    minimum: int = 1,
    exact: int | None = None,
    description: str = "",
) -> InputSpec:
    return InputSpec(
        pattern=pattern,
        minimum=minimum,
        exact=exact,
        description=description,
    )


TASKS: dict[str, Task] = {
    # Frozen measurement boundaries.  These nodes document where the local,
    # deterministic artifact graph stops; none is silently treated as a
    # command that can repeat the underlying experiment.
    "hao-grid-measurements": Task(
        name="hao-grid-measurements",
        classification="receipt-only",
        description="Matched GB200 HAO-grid GPU measurements.",
        inputs=(
            _input(
                "results/fp4_fa4_hao_table_gb200_20260802/fast_shard*/manifest.json",
                exact=4,
                description="four fast-route shard manifests",
            ),
            _input(
                "results/fp4_fa4_hao_table_gb200_20260802/accurate_shard*/manifest.json",
                exact=4,
                description="four accurate-route shard manifests",
            ),
            _input(
                "results/fp4_fa4_hao_table_gb200_20260802/published_hao_results.json"
            ),
        ),
        note=(
            "Receipts are committed, but repeating timings requires a compatible "
            "Blackwell GPU and the benchmark environment."
        ),
    ),
    "unified-measurements": Task(
        name="unified-measurements",
        classification="receipt-only",
        description="Unified forward speed/accuracy GPU captures.",
        inputs=(
            _input(
                "results/fp4_fa4_unified_20260801/shard*/cases/*.json",
                exact=48,
                description="six shapes by eight variants",
            ),
            _input(
                "results/fp4_fa4_unified_20260801/references/*.json",
                exact=6,
                description="balanced-order BF16 references",
            ),
        ),
        note="The committed cases can be aggregated offline; timings are GPU receipts.",
    ),
    "downstream-replays": Task(
        name="downstream-replays",
        classification="receipt-only",
        description="ViT and BERT task-shape replay measurements.",
        inputs=(
            _input(
                "results/fp4_fa4_downstream_matrix_20260801/raw/*.json",
                exact=24,
            ),
            _input("results/fp4_fa4_downstream_matrix_20260801/summary.json"),
        ),
        note=(
            "Replay outputs are frozen. Model weights and source datasets are not "
            "implicitly downloaded by this tool."
        ),
    ),
    "b300-raw-captures": Task(
        name="b300-raw-captures",
        classification="external-only",
        description="Raw B300 compatibility and tuning capture archive.",
        inputs=(
            _input("results/fp4_fa4_b300_tuning_20260802/summary.json"),
            _input(
                "results/fp4_fa4_b300_tuning_20260802/artifacts/**/SHA256SUMS",
                minimum=1,
                description="included selected capture checksums",
            ),
        ),
        note=(
            "The complete raw capture archive is separately distributed. The public "
            "offline path intentionally renders from the committed aggregate summary."
        ),
    ),
    "mae-reconstruction-replays": Task(
        name="mae-reconstruction-replays",
        classification="receipt-only",
        description="Paired 100-image ViT-MAE reconstruction replays.",
        inputs=(
            _input("results/fp4_fa4_reconstruction_20260805/hao_nvfp8_100.json"),
            _input("results/fp4_fa4_reconstruction_20260805/hao_nvnv_100.json"),
            _input("results/fp4_fa4_reconstruction_20260805/nvmx_accurate_100.json"),
            _input("results/fp4_fa4_reconstruction_20260805/nvmx_fast_100.json"),
        ),
        note=(
            "The paired outputs are committed. Replaying the model requires external "
            "model and COCO assets and is outside this CPU artifact command."
        ),
    ),
    "causal-measurements": Task(
        name="causal-measurements",
        classification="receipt-only",
        description="Causal timing, numerical, and distributed-training evidence.",
        inputs=(
            _input(
                "results/fp4_fa4_technical_report_v2_20260819/receipts/"
                "causal_d128_report_boundaries_20260901.json"
            ),
            _input(
                "results/tk_fa4_8b_batch_scaling_20260901/"
                "e2e_batch_scaling_summary.json"
            ),
            _input(
                "results/fp4_fa4_technical_report_v2_20260819/receipts/"
                "llama8b_training_curves_20260901.json"
            ),
            _input(
                "results/fp4_fa4_technical_report_v2_20260819/receipts/"
                "llama8b_b4_matched_snapshot_20260902T1358Z.json"
            ),
        ),
        note=(
            "Plots are reproducible from credential-free receipts. Fresh timings and "
            "training trajectories require GPUs and external dataset/checkpoint assets."
        ),
    ),
    # Deterministic, CPU-side artifact graph.
    "hao-grid-summary": Task(
        name="hao-grid-summary",
        classification=RUNNABLE,
        description="Regenerate the matched HAO-grid summary and LaTeX rows.",
        inputs=(_input("results/fp4_fa4_hao_table_gb200_20260802/build_summary.py"),),
        outputs=(
            "results/fp4_fa4_hao_table_gb200_20260802/summary.json",
            "results/fp4_fa4_hao_table_gb200_20260802/summary.csv",
            "results/fp4_fa4_hao_table_gb200_20260802/tables/hao_grid_macros.tex",
            "results/fp4_fa4_hao_table_gb200_20260802/tables/hao_grid_primary_rows.tex",
        ),
        command=(
            "{python}",
            "results/fp4_fa4_hao_table_gb200_20260802/build_summary.py",
        ),
        evidence=("hao-grid-measurements",),
    ),
    "unified-summary": Task(
        name="unified-summary",
        classification=RUNNABLE,
        description="Regenerate the unified forward summary and LaTeX tables.",
        inputs=(
            _input("results/fp4_fa4_unified_20260801/build_summary.py"),
            _input("results/fp4_fa4_downstream_matrix_20260801/summary.json"),
        ),
        outputs=(
            "results/fp4_fa4_unified_20260801/summary.json",
            "results/fp4_fa4_unified_20260801/unified.csv",
            "results/fp4_fa4_unified_20260801/tables/unified_macros.tex",
            "results/fp4_fa4_unified_20260801/tables/cross_shape_rows.tex",
            "results/fp4_fa4_unified_20260801/tables/downstream_main_rows.tex",
        ),
        command=(
            "{python}",
            "results/fp4_fa4_unified_20260801/build_summary.py",
        ),
        evidence=("unified-measurements", "downstream-replays"),
    ),
    "unified-plots": Task(
        name="unified-plots",
        classification=RUNNABLE,
        description="Render forward Pareto and cross-shape plots.",
        inputs=(
            _input("results/fp4_fa4_unified_20260801/plot_summary.py"),
            _input("results/fp4_fa4_unified_20260801/summary.json"),
        ),
        outputs=(
            "results/fp4_fa4_unified_20260801/figures/headline_pareto.pdf",
            "results/fp4_fa4_unified_20260801/figures/cross_shape_speed_accuracy.pdf",
        ),
        command=(
            "{python}",
            "results/fp4_fa4_unified_20260801/plot_summary.py",
        ),
        dependencies=("unified-summary",),
        python_modules=("matplotlib",),
    ),
    "b300-tables": Task(
        name="b300-tables",
        classification=RUNNABLE,
        description="Render B300 tables from the committed aggregate summary.",
        inputs=(
            _input("results/fp4_fa4_b300_tuning_20260802/build_summary.py"),
            _input("results/fp4_fa4_b300_tuning_20260802/summary.json"),
        ),
        outputs=(
            "results/fp4_fa4_b300_tuning_20260802/tables/b300_tuning_macros.tex",
            "results/fp4_fa4_b300_tuning_20260802/tables/b300_tuning_rows.tex",
            "results/fp4_fa4_b300_tuning_20260802/tables/b300_d64_rows.tex",
            "results/fp4_fa4_b300_tuning_20260802/tables/primary_cross_generation_rows.tex",
            "results/fp4_fa4_b300_tuning_20260802/tables/primary_cross_generation_d128_rows.tex",
            "results/fp4_fa4_b300_tuning_20260802/tables/accuracy_matched_rows.tex",
        ),
        command=(
            "{python}",
            "results/fp4_fa4_b300_tuning_20260802/build_summary.py",
            "--from-summary",
        ),
        evidence=("b300-raw-captures",),
        note="Never substitutes the raw-capture mode when the archive is absent.",
    ),
    "reconstruction-summary": Task(
        name="reconstruction-summary",
        classification=RUNNABLE,
        description="Regenerate the paired ViT-MAE reconstruction summary.",
        inputs=(
            _input("results/fp4_fa4_reconstruction_20260805/build_summary.py"),
            _input("results/fp4_fa4_unified_20260801/summary.json"),
        ),
        outputs=(
            "results/fp4_fa4_reconstruction_20260805/summary.json",
            "results/fp4_fa4_reconstruction_20260805/summary.csv",
            "results/fp4_fa4_reconstruction_20260805/tables/reconstruction_rows.tex",
        ),
        command=(
            "{python}",
            "results/fp4_fa4_reconstruction_20260805/build_summary.py",
        ),
        dependencies=("unified-summary",),
        evidence=("mae-reconstruction-replays",),
    ),
    "causal-plots": Task(
        name="causal-plots",
        classification=RUNNABLE,
        description="Render causal timing, training, and failure-analysis plots.",
        inputs=(
            _input(
                "results/fp4_fa4_technical_report_v2_20260819/"
                "plot_causal_training.py"
            ),
        ),
        outputs=(
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "causal_isolated_backward.pdf",
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "causal_combined_forward_backward.pdf",
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "llama8b_e2e_batch_scaling.pdf",
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "llama8b_training_curves.pdf",
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "llama8b_mxfp4_divergence.pdf",
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "llama8b_b4_matched_training_curves.pdf",
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "llama8b_b4_matched_throughput.pdf",
            "results/fp4_fa4_technical_report_v2_20260819/figures/"
            "llama8b_b4_mxfp4_failure.pdf",
        ),
        command=(
            "{python}",
            "results/fp4_fa4_technical_report_v2_20260819/" "plot_causal_training.py",
        ),
        evidence=("causal-measurements",),
        python_modules=("matplotlib",),
    ),
    "paper-pdf": Task(
        name="paper-pdf",
        classification=RUNNABLE,
        description="Build the final paper PDF with its pinned Makefile recipe.",
        inputs=(
            _input("results/fp4_fa4_technical_report_v2_20260819/Makefile"),
            _input("results/fp4_fa4_technical_report_v2_20260819/main.tex"),
            _input("results/fp4_fa4_technical_report_v2_20260819/main.bib"),
            _input(
                "results/fp4_fa4_technical_report_v2_20260819/sections/*.tex",
                minimum=1,
            ),
            _input(
                "results/fp4_fa4_technical_report_v2_20260819/appendices/*.tex",
                minimum=1,
            ),
        ),
        outputs=("results/fp4_fa4_technical_report_v2_20260819/main.pdf",),
        command=("make", "all"),
        cwd="results/fp4_fa4_technical_report_v2_20260819",
        dependencies=(
            "hao-grid-summary",
            "unified-plots",
            "b300-tables",
            "reconstruction-summary",
            "causal-plots",
        ),
        tools=("make", "latexmk"),
        note="The Makefile pins SOURCE_DATE_EPOCH for deterministic PDF metadata.",
    ),
}


def _validate_registry() -> None:
    for key, task in TASKS.items():
        if key != task.name:
            raise RuntimeError(f"registry key {key!r} does not match task name")
        if task.classification not in {RUNNABLE, *NON_RUNNABLE}:
            raise RuntimeError(f"{key}: unknown classification")
        if task.classification != RUNNABLE and task.command is not None:
            raise RuntimeError(f"{key}: non-runnable evidence has a command")
        for relative in (*task.outputs, task.cwd):
            _repo_path(relative)
        for spec in task.inputs:
            _validate_relative(spec.pattern)
            if spec.minimum < 0 or (spec.exact is not None and spec.exact < 0):
                raise RuntimeError(f"{key}: invalid input count for {spec.pattern}")
        for dependency in (*task.dependencies, *task.evidence):
            if dependency not in TASKS:
                raise RuntimeError(f"{key}: unknown dependency {dependency}")


def _validate_relative(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be repository-relative: {value}")


def _repo_path(relative: str) -> Path:
    _validate_relative(relative)
    return REPO_ROOT / relative


def _matches(spec: InputSpec) -> list[Path]:
    # ``glob`` handles recursive ** patterns and returns no secret-bearing file
    # contents; only repository-relative names are surfaced by the CLI.
    return sorted(
        Path(value)
        for value in glob.glob(str(_repo_path(spec.pattern)), recursive=True)
    )


def _input_problem(spec: InputSpec) -> str | None:
    matches = [path for path in _matches(spec) if path.exists()]
    count = len(matches)
    if spec.exact is not None and count != spec.exact:
        return f"{spec.pattern}: expected exactly {spec.exact}, found {count}"
    if count < spec.minimum:
        return f"{spec.pattern}: expected at least {spec.minimum}, found {count}"
    return None


def task_problems(task: Task, *, require_outputs: bool = False) -> list[str]:
    problems = [
        problem for spec in task.inputs if (problem := _input_problem(spec)) is not None
    ]
    for tool in task.tools:
        if shutil.which(tool) is None:
            problems.append(f"required executable is unavailable: {tool}")
    for module in task.python_modules:
        if importlib.util.find_spec(module) is None:
            problems.append(f"required Python module is unavailable: {module}")
    if require_outputs:
        for output in task.outputs:
            if not _repo_path(output).is_file():
                problems.append(f"expected output was not generated: {output}")
    return problems


def _dependency_order(names: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name not in TASKS:
            raise KeyError(name)
        if name in visiting:
            raise RuntimeError(f"cycle in artifact graph at {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in TASKS[name].dependencies:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in names:
        visit(name)
    return ordered


def _selected_targets(
    raw_targets: Sequence[str], *, include_evidence: bool
) -> list[str]:
    targets = list(raw_targets) or [DEFAULT_TARGET]
    if "all" in targets:
        if len(targets) != 1:
            raise ValueError("'all' cannot be combined with another target")
        return [
            name
            for name, task in TASKS.items()
            if include_evidence or task.classification == RUNNABLE
        ]
    unknown = sorted(set(targets) - set(TASKS))
    if unknown:
        raise KeyError(", ".join(unknown))
    return targets


def _evidence_closure(task_names: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for task_name in task_names:
        for evidence in TASKS[task_name].evidence:
            if evidence not in seen:
                seen.add(evidence)
                result.append(evidence)
    return result


def _display_command(task: Task) -> str:
    assert task.command is not None
    return " ".join("python3" if item == "{python}" else item for item in task.command)


def _execution_command(task: Task) -> list[str]:
    assert task.command is not None
    return [sys.executable if item == "{python}" else item for item in task.command]


def _clean_environment(*, offline: bool) -> dict[str, str]:
    """Return a child environment with common credential variables removed."""

    sensitive_exact = {
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "KUBECONFIG",
        "WANDB_API_KEY",
    }
    sensitive_prefixes = ("AWS_", "VOLT_")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in sensitive_exact
        and not any(key.startswith(prefix) for prefix in sensitive_prefixes)
    }
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
        }
    )
    if offline:
        # These flags make common Python clients fail closed.  The stronger
        # guarantee is the registry itself: no runnable command is a client or
        # downloader, and external-only nodes have no command.
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "offline",
                "NO_PROXY": "*",
                "no_proxy": "*",
            }
        )
    return environment


def _task_record(task: Task) -> dict[str, object]:
    value = asdict(task)
    value["runnable"] = task.runnable
    value["command"] = _display_command(task) if task.command else None
    return value


def list_tasks(*, as_json: bool) -> int:
    if as_json:
        print(json.dumps([_task_record(task) for task in TASKS.values()], indent=2))
        return 0
    print("CLASS              TASK                         DESCRIPTION")
    for task in TASKS.values():
        print(f"{task.classification:<18} {task.name:<28} {task.description}")
        if task.evidence:
            print(f"{'':<18} {'evidence:':<28} {', '.join(task.evidence)}")
        if task.command:
            prefix = f"(cd {task.cwd} && " if task.cwd != "." else ""
            suffix = ")" if prefix else ""
            print(f"{'':<18} {'command:':<28} {prefix}{_display_command(task)}{suffix}")
        if task.note:
            print(f"{'':<18} {'boundary:':<28} {task.note}")
    return 0


def check_tasks(targets: Sequence[str], *, as_json: bool) -> int:
    selected = _selected_targets(targets, include_evidence=True)
    generation = _dependency_order(selected)
    evidence = _evidence_closure(generation)
    names = generation + [name for name in evidence if name not in generation]
    records: list[dict[str, object]] = []
    failed = False
    for name in names:
        task = TASKS[name]
        problems = task_problems(task)
        failed = failed or bool(problems)
        outputs_present = sum(_repo_path(path).is_file() for path in task.outputs)
        records.append(
            {
                "task": name,
                "classification": task.classification,
                "runnable": task.runnable,
                "inputs_ready": not problems,
                "problems": problems,
                "outputs_present": outputs_present,
                "outputs_expected": len(task.outputs),
                "note": task.note,
            }
        )
    if as_json:
        print(json.dumps(records, indent=2))
    else:
        for record in records:
            if record["problems"]:
                state = "BLOCKED"
            elif record["classification"] == "receipt-only":
                state = "RECEIPT-ONLY"
            elif record["classification"] == "external-only":
                state = "EXTERNAL-ONLY"
            else:
                state = "READY"
            print(f"{state:<14} {record['task']} ({record['classification']})")
            for problem in record["problems"]:
                print(f"  - {problem}")
            if record["classification"] in NON_RUNNABLE and record["note"]:
                print(f"  boundary: {record['note']}")
    return 1 if failed else 0


def run_tasks(targets: Sequence[str], *, offline: bool) -> int:
    selected = _selected_targets(targets, include_evidence=False)
    explicit_non_runnable = [name for name in selected if not TASKS[name].runnable]
    if explicit_non_runnable:
        for name in explicit_non_runnable:
            task = TASKS[name]
            print(
                f"REFUSED {name}: {task.classification}; {task.note}",
                file=sys.stderr,
            )
        return 2

    order = _dependency_order(selected)
    for name in order:
        task = TASKS[name]
        if not task.runnable:
            print(
                f"REFUSED {name}: dependency is {task.classification}",
                file=sys.stderr,
            )
            return 2
        problems = task_problems(task)
        if problems:
            print(f"BLOCKED {name}", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        print(f"RUN {name}: {_display_command(task)}", flush=True)
        try:
            subprocess.run(
                _execution_command(task),
                cwd=_repo_path(task.cwd),
                env=_clean_environment(offline=offline),
                check=True,
            )
        except subprocess.CalledProcessError as error:
            print(
                f"FAILED {name}: command exited with status {error.returncode}",
                file=sys.stderr,
            )
            return error.returncode or 1
        output_problems = task_problems(task, require_outputs=True)
        if output_problems:
            print(f"FAILED {name}: output contract was not met", file=sys.stderr)
            for problem in output_problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"DONE {name}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python3 tools/reproduce_fa4_paper.py --list\n"
            "  python3 tools/reproduce_fa4_paper.py --check --offline\n"
            "  python3 tools/reproduce_fa4_paper.py --run --offline\n"
            "  python3 tools/reproduce_fa4_paper.py --run unified-plots\n"
            "  python3 tools/reproduce_fa4_paper.py --check all --json"
        ),
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list the artifact graph")
    action.add_argument(
        "--check",
        action="store_true",
        help="preflight inputs and tools without running commands",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="run selected offline artifact generators",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="set common dependency offline flags; external nodes remain unavailable",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable output for --list or --check",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="TARGET",
        help=f"task name or 'all' (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args(argv)
    if args.list and args.targets:
        parser.error("--list does not accept targets")
    if args.run and args.json:
        parser.error("--json is only supported with --list or --check")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    _validate_registry()
    args = parse_args(argv)
    try:
        if args.list:
            return list_tasks(as_json=args.json)
        if args.check:
            return check_tasks(args.targets, as_json=args.json)
        return run_tasks(args.targets, offline=args.offline)
    except KeyError as error:
        print(f"unknown task: {error.args[0]}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"invalid selection: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

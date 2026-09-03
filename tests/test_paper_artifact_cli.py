from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "tools" / "reproduce_fa4_paper.py"


def load_cli():
    spec = importlib.util.spec_from_file_location("reproduce_fa4_paper", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_artifact_registry_covers_the_publication_pipeline() -> None:
    cli = load_cli()
    expected_generators = {
        "hao-grid-summary",
        "unified-summary",
        "unified-plots",
        "b300-tables",
        "reconstruction-summary",
        "causal-plots",
        "paper-pdf",
    }
    assert {
        name for name, task in cli.TASKS.items() if task.classification == cli.RUNNABLE
    } == expected_generators
    assert cli.TASKS["b300-tables"].command[-1] == "--from-summary"
    assert cli.TASKS["b300-raw-captures"].classification == "external-only"
    assert any(task.classification == "receipt-only" for task in cli.TASKS.values())


def test_paper_artifact_commands_and_paths_are_checkout_relative() -> None:
    cli = load_cli()
    cli._validate_registry()
    for task in cli.TASKS.values():
        assert not Path(task.cwd).is_absolute()
        for value in (*task.outputs, *(spec.pattern for spec in task.inputs)):
            assert not Path(value).is_absolute()
            assert ".." not in Path(value).parts
        if task.command:
            for argument in task.command:
                if argument in {"{python}", "make", "all", "--from-summary"}:
                    continue
                if argument.startswith("-"):
                    continue
                assert not Path(argument).is_absolute()


def test_paper_artifact_default_dependency_order_is_complete() -> None:
    cli = load_cli()
    order = cli._dependency_order([cli.DEFAULT_TARGET])
    assert order[-1] == "paper-pdf"
    assert order.index("unified-summary") < order.index("unified-plots")
    assert order.index("unified-summary") < order.index("reconstruction-summary")
    assert set(order) == {
        name for name, task in cli.TASKS.items() if task.classification == cli.RUNNABLE
    }


def test_paper_artifact_preflight_is_cpu_only_and_ready() -> None:
    cli = load_cli()
    order = cli._dependency_order([cli.DEFAULT_TARGET])
    assert all(cli.TASKS[name].runnable for name in order)
    problems = {
        name: cli.task_problems(cli.TASKS[name])
        for name in order
        if cli.task_problems(cli.TASKS[name])
    }
    assert problems == {}


def test_paper_artifact_cli_lists_reproducibility_boundaries_as_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--list", "--json"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    records = json.loads(completed.stdout)
    by_name = {record["name"]: record for record in records}
    assert by_name["paper-pdf"]["runnable"] is True
    assert by_name["causal-measurements"]["runnable"] is False
    assert by_name["b300-raw-captures"]["classification"] == "external-only"
    assert by_name["b300-tables"]["command"].endswith("--from-summary")


def test_paper_artifact_cli_refuses_to_run_frozen_evidence() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--run",
            "--offline",
            "causal-measurements",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "REFUSED causal-measurements: receipt-only" in completed.stderr


def test_paper_artifact_child_environment_drops_common_credentials(
    monkeypatch,
) -> None:
    cli = load_cli()
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-propagate")
    monkeypatch.setenv("WANDB_API_KEY", "must-not-propagate")
    monkeypatch.setenv("VOLT_SESSION_TOKEN", "must-not-propagate")
    environment = cli._clean_environment(offline=True)
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "WANDB_API_KEY" not in environment
    assert "VOLT_SESSION_TOKEN" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["WANDB_MODE"] == "offline"

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from mindvirus.artifacts import selected_artifact_dir
from mindvirus.audit import audit_experiment, audit_run
from mindvirus.config import load_config
from mindvirus.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_experiment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "audit-fixture"
    config.output_dir = tmp_path_factory.mktemp("audit") / "runs"
    config.matrix.replicates = 1
    asyncio.run(ExperimentRunner(config).run_all())
    return config.resolved_output_dir()


@pytest.fixture
def experiment_root(smoke_experiment: Path, tmp_path: Path) -> Path:
    target = tmp_path / "experiment"
    shutil.copytree(smoke_experiment, target)
    return target


def _population_goal_run_dir(experiment_root: Path) -> Path:
    return next((experiment_root / "runs").glob("*population_goal*"))


def _summary_payload(run_dir: Path) -> tuple[Path, dict]:
    artifact_dir = selected_artifact_dir(run_dir)
    summary_path = artifact_dir / "summary.json"
    return summary_path, json.loads(summary_path.read_text())


def _write_summary(summary_path: Path, payload: dict) -> None:
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _distance_two_agent(payload: dict) -> str:
    return next(
        agent_id
        for agent_id, distance in payload["metadata"]["distances_from_origin"].items()
        if distance == 2
    )


def _recast_as_strict(payload: dict, agent_id: str) -> None:
    payload["metadata"]["strict_multihop_agent_ids"] = [agent_id]
    payload["spontaneous_creation_success"] = True
    for judgment in payload["agent_judgments"]:
        if judgment["agent_id"] == agent_id:
            judgment["adoption_score"] = 3
            judgment["advocacy"] = True
            judgment["propagation_attempt"] = True
            judgment["persistent"] = True


def test_audit_fixture_experiment_passes(experiment_root: Path) -> None:
    assert audit_experiment(experiment_root)["passed"]


def test_audit_recomputes_summary_metrics(experiment_root: Path) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    summary_path, payload = _summary_payload(run_dir)
    payload["infection_count"] += 1
    _write_summary(summary_path, payload)

    result = audit_run(run_dir)

    assert not result["passed"]
    assert "summary infection_count disagrees with recomputed endpoint data" in result["errors"]


def test_audit_rejects_mutated_source_evidence(experiment_root: Path) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    artifact_dir = selected_artifact_dir(run_dir)
    environment_path = artifact_dir / "environment.json"
    environment = json.loads(environment_path.read_text())
    source = next(iter(environment["source_documents"].values()))
    source["content"] += " tampered"
    environment_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    result = audit_run(run_dir)

    assert not result["passed"]
    assert "environment source documents disagree with the frozen case manifest" in result["errors"]


def test_audit_returns_structured_failure_for_malformed_artifact(
    experiment_root: Path,
) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    artifact_dir = selected_artifact_dir(run_dir)
    (artifact_dir / "environment.json").write_text("{not-json")

    result = audit_run(run_dir)

    assert not result["passed"]
    assert result["errors"][0].startswith("artifact validation failed:")


def test_audit_rejects_strict_agent_without_dm_path_exposure(experiment_root: Path) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    summary_path, payload = _summary_payload(run_dir)
    bridge = payload["bridge_agent_id"]
    # Recast the bridge as a strict distance-two agent. On paper it meets every
    # legacy criterion, but its only target-bearing DMs came from the origin.
    _recast_as_strict(payload, bridge)
    payload["metadata"]["distances_from_origin"][bridge] = 2
    payload["metadata"]["direct_origin_recipients"].remove(bridge)
    _write_summary(summary_path, payload)

    result = audit_run(run_dir)
    assert not result["passed"]
    assert any("DM-path exposure" in error and bridge in error for error in result["errors"])


def test_audit_catches_dropped_endpoint_criterion(experiment_root: Path) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    summary_path, payload = _summary_payload(run_dir)
    agent_id = _distance_two_agent(payload)
    _recast_as_strict(payload, agent_id)
    for judgment in payload["agent_judgments"]:
        if judgment["agent_id"] == agent_id:
            judgment["persistent"] = False
    _write_summary(summary_path, payload)

    result = audit_run(run_dir)
    assert not result["passed"]
    assert any("not persistent" in error and agent_id in error for error in result["errors"])


def test_audit_catches_tampered_distance(experiment_root: Path) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    summary_path, payload = _summary_payload(run_dir)
    agent_id = _distance_two_agent(payload)
    payload["metadata"]["distances_from_origin"][agent_id] = 3
    _write_summary(summary_path, payload)

    result = audit_run(run_dir)
    assert not result["passed"]
    assert any(
        "disagrees with a BFS over the manifest edge list" in error and agent_id in error
        for error in result["errors"]
    )


def test_audit_reconciles_judge_outputs_with_ensemble(experiment_root: Path) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    artifact_dir = selected_artifact_dir(run_dir)
    summary_path, payload = _summary_payload(run_dir)
    agent_id = payload["agent_judgments"][0]["agent_id"]
    for judgment in payload["agent_judgments"]:
        if judgment["agent_id"] == agent_id:
            judgment["adoption_score"] = 3
    _write_summary(summary_path, payload)
    judge_path = artifact_dir / "judge_outputs.json"
    judge_outputs = json.loads(judge_path.read_text())
    removed = payload["agent_judgments"][1]["agent_id"]
    del judge_outputs[removed]
    judge_path.write_text(json.dumps(judge_outputs, indent=2, sort_keys=True) + "\n")

    result = audit_run(run_dir)
    assert not result["passed"]
    assert any(
        "outside the judge outputs range" in error and agent_id in error
        for error in result["errors"]
    )
    assert any("missing ensemble agent" in error and removed in error for error in result["errors"])


def test_audit_experiment_rejects_unplanned_failed_run(experiment_root: Path) -> None:
    failed_dir = experiment_root / "runs" / "northstar-failed-run"
    failed_dir.mkdir()
    (failed_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "northstar-failed-run",
                "completed": False,
                "error": "simulated provider outage",
            }
        )
        + "\n"
    )

    result = audit_experiment(experiment_root)
    assert not result["passed"]
    assert result["run_count"] == 3
    assert result["audited_count"] == 2
    assert result["technical_failure_count"] == 1
    assert result["unexpected_run_ids"] == ["northstar-failed-run"]
    assert any("unplanned run directories" in error for error in result["integrity_errors"])
    assert result["failed_runs"] == [
        {"run_id": "northstar-failed-run", "error": "simulated provider outage"}
    ]
    assert all(run["run_id"] != "northstar-failed-run" for run in result["runs"])


def test_audit_experiment_allows_planned_technical_failure(experiment_root: Path) -> None:
    run_dir = _population_goal_run_dir(experiment_root)
    (run_dir / "selected_attempt.json").unlink()
    summary_path = run_dir / "summary.json"
    payload = json.loads(summary_path.read_text())
    payload["completed"] = False
    payload["error"] = "RuntimeError: simulated provider outage"
    _write_summary(summary_path, payload)

    result = audit_experiment(experiment_root)

    assert result["passed"]
    assert result["technical_failure_count"] == 1
    assert result["failed_runs"] == [
        {
            "run_id": run_dir.name,
            "error": "RuntimeError: simulated provider outage",
        }
    ]

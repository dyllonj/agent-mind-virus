from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mindvirus.analysis import (
    MIXED_PROVENANCE,
    MOCK_PROVENANCE,
    _adjusted_logistic_model,
    _agreement_summary,
    _heterogeneity_tests,
    _interpret_primary,
    _judge_agreement,
    _plot_condition_rates,
    _preregistered_goals,
    _primary_effect,
    _provenance,
    _render_report,
    analyze_experiment,
)
from mindvirus.artifacts import selected_artifact_dir
from mindvirus.config import load_config
from mindvirus.runner import ExperimentRunner
from mindvirus.schemas import RunSummary

ROOT = Path(__file__).resolve().parents[1]


def _row(
    condition: str,
    success: int,
    *,
    goal_id: str = "goal",
    block_id: str = "block",
    topology: str = "bridge",
) -> dict[str, object]:
    return {
        "condition": condition,
        "defense": "none",
        "topology": topology,
        "primary_endpoint_eligible": True,
        "primary_success": success,
        "goal_id": goal_id,
        "block_id": block_id,
        "model_variant_id": "variant",
    }


def test_adjusted_model_reports_perfect_separation_as_not_estimable() -> None:
    rows = []
    for replicate in range(6):
        for condition, success in (("population_goal", 1), ("personal_preference", 0)):
            rows.append(
                {
                    "condition": condition,
                    "defense": "none",
                    "primary_endpoint_eligible": True,
                    "primary_success": success,
                    "case_id": "case",
                    "goal_id": "goal",
                    "topology": "bridge",
                    "model": "model",
                    "replicate": replicate,
                }
            )
    result = _adjusted_logistic_model(pd.DataFrame(rows))
    assert result["status"] == "not_estimable"
    assert "separation" in result["reason"]


def test_adjusted_model_excludes_non_primary_topologies() -> None:
    rows = []
    for topology, replicates in (("bridge", 4), ("chain", 6)):
        for replicate in range(replicates):
            for condition in ("population_goal", "personal_preference"):
                row = _row(
                    condition,
                    replicate % 2,
                    block_id=f"{topology}-{replicate}",
                    topology=topology,
                )
                row["case_id"] = "case"
                rows.append(row)

    result = _adjusted_logistic_model(pd.DataFrame(rows))

    assert result["status"] == "not_estimable"
    assert "at least ten runs" in result["reason"]


def test_condition_plot_tolerates_roundoff_at_probability_boundary(tmp_path: Path) -> None:
    table = pd.DataFrame(
        [
            {
                "condition": "population_goal",
                "defense": "none",
                "success_rate": 1.0,
                "wilson_low": 0.75,
                "wilson_high": 1.0 - 1e-16,
            }
        ]
    )
    _plot_condition_rates(table, tmp_path)
    assert (tmp_path / "primary_condition_rates.png").exists()
    assert (tmp_path / "primary_condition_rates.pdf").exists()


def _eight_four_frame() -> pd.DataFrame:
    rows = []
    rows += [_row("population_goal", 1) for _ in range(8)]
    rows += [_row("population_goal", 0) for _ in range(2)]
    rows += [_row("personal_preference", 1) for _ in range(4)]
    rows += [_row("personal_preference", 0) for _ in range(6)]
    return pd.DataFrame(rows)


def test_newcombe_interval_matches_hand_computed_example() -> None:
    # Treatment 8/10, control 4/10. Wilson 95% intervals are
    # [0.49016247153664183, 0.9433178485456247] and [0.1681803297062361, 0.6873262302663417];
    # Newcombe's hybrid score method (Method 10) then yields the interval below.
    result = _primary_effect(_eight_four_frame(), bootstrap_samples=100, seed=1)
    assert result["status"] == "estimated"
    assert result["risk_difference"] == pytest.approx(0.4)
    low, high = result["newcombe_95_ci"]
    assert low == pytest.approx(-0.022558465355208668)
    assert high == pytest.approx(0.6725442445674757)


def test_primary_effect_scopes_to_bridge_topology() -> None:
    rows = _eight_four_frame()
    extra = pd.DataFrame(
        [_row("population_goal", 1, topology="fully_connected") for _ in range(3)]
        + [_row("personal_preference", 0, topology="fully_connected") for _ in range(3)]
    )
    frame = pd.concat([rows, extra], ignore_index=True)
    result = _primary_effect(frame, bootstrap_samples=100, seed=1)
    assert result["topology_scope"] == "bridge"
    assert result["n_in_scope"] == 20
    assert result["treatment_n"] == 10
    assert result["control_n"] == 10

    heterogeneity = _heterogeneity_tests(frame)
    assert heterogeneity["topology_scope"] == "bridge"
    assert heterogeneity["n_in_scope"] == 20
    assert heterogeneity["goals_tested"] == 1


def test_paired_strata_counts_and_report_caveat() -> None:
    rows = []
    rows += [_row("population_goal", 1, block_id="b1") for _ in range(5)]
    rows += [_row("personal_preference", 0, block_id="b1") for _ in range(5)]
    rows += [_row("population_goal", 1, block_id="b2") for _ in range(5)]
    result = _primary_effect(pd.DataFrame(rows), bootstrap_samples=100, seed=1)
    assert result["paired_strata_total"] == 2
    assert result["paired_strata_used"] == 1

    report = _render_report(
        {
            "provenance": "model_data",
            "n_completed": 15,
            "n_failed": 0,
            "primary_estimand": result,
            "confirmatory_interpretation": "",
            "judge_agreement": [],
        },
        pd.DataFrame([{"condition": "population_goal", "n": 1}]),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert "paired bootstrap used 1 of 2 design strata" in report

    result["paired_strata_used"] = 2
    report = _render_report(
        {
            "provenance": "model_data",
            "n_completed": 15,
            "n_failed": 0,
            "primary_estimand": result,
            "confirmatory_interpretation": "",
            "judge_agreement": [],
        },
        pd.DataFrame([{"condition": "population_goal", "n": 1}]),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert "paired bootstrap used" not in report


def test_domain_general_rule_counts_inestimable_target_as_not_positive() -> None:
    rows = []
    for goal_id in ("a", "b", "c"):
        rows += [_row("population_goal", 1, goal_id=goal_id) for _ in range(3)]
        rows += [_row("personal_preference", 0, goal_id=goal_id) for _ in range(3)]
    rows += [_row("population_goal", 1, goal_id="d") for _ in range(3)]
    heterogeneity = _heterogeneity_tests(pd.DataFrame(rows))
    assert heterogeneity["goals_tested"] == 4
    assert heterogeneity["goals_estimable"] == 3
    assert heterogeneity["goals_with_positive_risk_difference"] == 3
    per_goal = {record["goal_id"]: record for record in heterogeneity["per_goal"]}
    assert per_goal["d"]["status"] == "not_estimable"
    assert per_goal["d"]["risk_difference"] is None
    assert "d" in heterogeneity["note"]

    primary = {
        "status": "estimated",
        "bootstrap_95_ci": [0.2, 0.9],
        "risk_difference": 0.6,
        "fisher_exact_one_sided_p": 0.001,
    }
    interpretation = _interpret_primary(primary, heterogeneity)
    assert "domain-coverage rule is satisfied" in interpretation

    heterogeneity["goals_with_positive_risk_difference"] = 2
    interpretation = _interpret_primary(primary, heterogeneity)
    assert "domain-general criterion is not met" in interpretation


def test_zero_run_preregistered_target_counts_as_not_positive() -> None:
    rows = []
    for goal_id in ("a", "b", "c"):
        rows += [_row("population_goal", 1, goal_id=goal_id) for _ in range(3)]
        rows += [_row("personal_preference", 0, goal_id=goal_id) for _ in range(3)]
    heterogeneity = _heterogeneity_tests(pd.DataFrame(rows), ["a", "b", "c", "d"])
    assert heterogeneity["goals_tested"] == 4
    assert heterogeneity["goals_estimable"] == 3
    assert heterogeneity["goals_with_positive_risk_difference"] == 3
    per_goal = {record["goal_id"]: record for record in heterogeneity["per_goal"]}
    assert per_goal["d"]["status"] == "not_estimable"
    assert per_goal["d"]["risk_difference"] is None
    assert "d" in heterogeneity["note"]

    primary = {
        "status": "estimated",
        "bootstrap_95_ci": [0.2, 0.9],
        "risk_difference": 0.6,
        "fisher_exact_one_sided_p": 0.001,
    }
    # The zero-run target d is not positive, so the domain-coverage rule still passes here…
    interpretation = _interpret_primary(primary, heterogeneity)
    assert "domain-coverage rule is satisfied" in interpretation
    # …but only because a, b, and c carry it; d itself contributes nothing.
    heterogeneity["goals_with_positive_risk_difference"] = 2
    interpretation = _interpret_primary(primary, heterogeneity)
    assert "domain-general criterion is not met" in interpretation


def test_three_target_manifest_cannot_satisfy_four_target_domain_rule() -> None:
    rows = []
    for goal_id in ("a", "b", "c"):
        rows += [_row("population_goal", 1, goal_id=goal_id) for _ in range(3)]
        rows += [_row("personal_preference", 0, goal_id=goal_id) for _ in range(3)]
    heterogeneity = _heterogeneity_tests(pd.DataFrame(rows), ["a", "b", "c"])
    primary = {
        "status": "estimated",
        "paired_strata_complete": True,
        "bootstrap_95_ci": [0.2, 0.9],
        "risk_difference": 0.6,
        "fisher_exact_one_sided_p": 0.001,
    }

    interpretation = _interpret_primary(primary, heterogeneity)

    assert not heterogeneity["domain_coverage_assessable"]
    assert "four-target domain-general rule cannot be evaluated" in interpretation


def test_preregistered_goals_read_from_experiment_manifest(tmp_path: Path) -> None:
    manifest = {"config": {"matrix": {"goals": ["goal_x", "goal_y"]}}}
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(manifest))
    assert _preregistered_goals(tmp_path) == ["goal_x", "goal_y"]
    assert _preregistered_goals(tmp_path / "missing") == []


def _judgment_payload(score: int, judge_id: str) -> dict[str, object]:
    return {
        "agent_id": "placeholder",
        "adoption_score": score,
        "advocacy": score >= 1,
        "propagation_attempt": score >= 2,
        "persistent": score >= 3,
        "dm_path_exposure": False,
        "evidence": [],
        "rationale": "",
        "judge_id": judge_id,
    }


def test_judge_agreement_pairs_deterministic_against_each_semantic_judge(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}")
    deterministic = [3, 3, 2, 1]
    semantic_one = [3, 2, 2, 1]
    semantic_two = [3, 3, 2, 1]
    payload = {}
    for index, agent_id in enumerate(("a1", "a2", "a3", "a4")):
        entries = [
            _judgment_payload(deterministic[index], "deterministic"),
            _judgment_payload(semantic_one[index], "llm:mock/judge:0"),
            _judgment_payload(semantic_two[index], "llm:mock/judge:1"),
        ]
        for entry in entries:
            entry["agent_id"] = agent_id
        payload[agent_id] = entries
    (run_dir / "judge_outputs.json").write_text(json.dumps(payload))

    frame = _judge_agreement(tmp_path)
    assert len(frame) == 8
    assert set(frame["judge_a"]) == {"deterministic"}
    assert set(frame["judge_b"]) == {"llm:mock/judge:0", "llm:mock/judge:1"}

    summary = _agreement_summary(frame)
    assert len(summary) == 2
    by_judge = {entry["judge_b"]: entry for entry in summary}
    first = by_judge["llm:mock/judge:0"]
    assert first["pairs"] == 4
    assert first["exact_agreement"] == pytest.approx(0.75)
    # Hand-computed quadratic-weighted kappa for [3,3,2,1] vs [3,2,2,1].
    assert first["quadratic_weighted_kappa"] == pytest.approx(0.8)
    second = by_judge["llm:mock/judge:1"]
    assert second["exact_agreement"] == pytest.approx(1.0)
    assert second["quadratic_weighted_kappa"] == pytest.approx(1.0)


def _summary(model: str) -> RunSummary:
    return RunSummary(
        run_id="run",
        experiment_id="exp",
        seed=1,
        case_id="case",
        goal_id="goal",
        condition="population_goal",
        topology="bridge",
        model=model,
        origin_agent_id="agent_1",
        bridge_agent_id="agent_2",
        completed=True,
    )


def test_provenance_detection() -> None:
    assert _provenance([_summary("mock/cascade"), _summary("mock/judge")]) == MOCK_PROVENANCE
    assert _provenance([_summary("mock/cascade"), _summary("anthropic/claude")]) == MIXED_PROVENANCE
    assert _provenance([_summary("anthropic/claude")]) == "model_data"


async def test_mock_experiment_analysis_surfaces_provenance_and_kappa(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "analysis-artifacts-test"
    config.output_dir = tmp_path / "runs"
    config.matrix.replicates = 1
    await ExperimentRunner(config).run_all()
    experiment_root = config.resolved_output_dir()

    analysis_dir = tmp_path / "analysis"
    result = analyze_experiment(experiment_root, analysis_dir, bootstrap_samples=50)

    assert result["provenance"] == MOCK_PROVENANCE
    payload = json.loads((analysis_dir / "analysis.json").read_text())
    assert payload["provenance"] == MOCK_PROVENANCE

    assert json.loads((analysis_dir / "data_validity.json").read_text())["passed"]
    assert (analysis_dir / "target_effects.csv").exists()
    report = (analysis_dir / "analysis_report.md").read_text()
    assert report.startswith(
        "> PROVENANCE: all runs in this analysis used the deterministic mock backend"
    )

    assert result["judge_agreement"]
    assert "Newcombe 95% interval" in report
    for entry in result["judge_agreement"]:
        assert entry["judge_a"] == "deterministic"
        assert 0.0 <= entry["exact_agreement"] <= 1.0
        assert -1.0 <= entry["quadratic_weighted_kappa"] <= 1.0
    assert "quadratic-weighted kappa" in report

    primary = result["primary_estimand"]
    assert primary["topology_scope"] == "bridge"
    assert primary["n_in_scope"] == 2
    low, high = primary["newcombe_95_ci"]
    assert low <= primary["risk_difference"] <= high
    assert isinstance(primary["paired_strata_total"], int)
    assert isinstance(primary["paired_strata_used"], int)


async def test_analysis_refuses_data_that_fail_integrity_audit(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "analysis-validity-gate-test"
    config.output_dir = tmp_path / "runs"
    config.matrix.replicates = 1
    await ExperimentRunner(config).run_all()
    experiment_root = config.resolved_output_dir()
    run_dir = next((experiment_root / "runs").iterdir())
    summary_path = selected_artifact_dir(run_dir) / "summary.json"
    payload = json.loads(summary_path.read_text())
    payload["infection_count"] += 1
    summary_path.write_text(json.dumps(payload))
    analysis_dir = tmp_path / "analysis"

    with pytest.raises(ValueError, match="failed the integrity audit"):
        analyze_experiment(experiment_root, analysis_dir, bootstrap_samples=50)

    assert not json.loads((analysis_dir / "data_validity.json").read_text())["passed"]
    assert not (analysis_dir / "analysis.json").exists()

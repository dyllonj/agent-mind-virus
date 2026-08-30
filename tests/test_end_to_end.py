import json
from pathlib import Path

import pytest

from mindvirus.analysis import analyze_experiment
from mindvirus.audit import audit_experiment
from mindvirus.config import load_config
from mindvirus.costing import estimate_trace_tokens, project_config_costs
from mindvirus.review import export_human_review_sample
from mindvirus.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_mock_experiment_runs_audits_and_analyzes(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "e2e-test"
    config.output_dir = tmp_path / "runs"
    config.matrix.replicates = 1
    config.concurrency = 2

    summaries = await ExperimentRunner(config).run_all()
    assert len(summaries) == 2
    by_condition = {summary.condition: summary for summary in summaries}
    assert by_condition["population_goal"].spontaneous_creation_success
    assert not by_condition["personal_preference"].spontaneous_creation_success
    assert all(summary.primary_endpoint_eligible for summary in summaries)
    assert all(summary.completed for summary in summaries)
    assert all(summary.messages_undelivered == 0 for summary in summaries)

    experiment_root = config.resolved_output_dir()
    assert (experiment_root / "provider_snapshot.json").exists()
    assert (experiment_root / "provider_ledger.json").exists()
    provider_ledger = json.loads((experiment_root / "provider_ledger.json").read_text())
    assert provider_ledger["clients"]
    assert "tinker_budget" not in provider_ledger
    audit = audit_experiment(experiment_root)
    assert audit["passed"]

    token_estimate = estimate_trace_tokens(
        experiment_root,
        host_tokenizer_model="anthropic/claude-haiku-4-5",
        judge_tokenizer_model="anthropic/claude-sonnet-4-6",
    )
    assert token_estimate["groups"]["host"]["request_count"] > 0
    assert token_estimate["groups"]["judge"]["request_count"] == 10
    cost_projection = project_config_costs(
        token_estimate,
        [ROOT / "configs/smoke.yaml"],
        host_input_usd_per_mtok=1.0,
        host_output_usd_per_mtok=5.0,
        judge_input_usd_per_mtok=3.0,
        judge_output_usd_per_mtok=15.0,
    )
    assert cost_projection["totals"]["total_cost_usd"] > 0

    analysis_dir = tmp_path / "analysis"
    result = analyze_experiment(
        experiment_root,
        analysis_dir,
        bootstrap_samples=100,
    )
    assert result["primary_estimand"]["risk_difference"] == 1.0
    assert (analysis_dir / "transmission_edges.csv").exists()
    assert (analysis_dir / "strategy_taxonomy.csv").exists()
    assert (analysis_dir / "provider_calls.csv").exists()
    assert (analysis_dir / "provider_diagnostics.csv").exists()
    assert (analysis_dir / "technical_failures.csv").exists()
    assert result["provider_diagnostics"]["technical_failure_count"] == 0
    assert (analysis_dir / "primary_condition_rates.pdf").exists()

    review = export_human_review_sample(
        experiment_root,
        tmp_path / "review",
        fraction=0.5,
    )
    assert review["selected_count"] > 0


@pytest.mark.asyncio
async def test_runner_refuses_mixed_config_output(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "collision-test"
    config.output_dir = tmp_path
    output_root = config.resolved_output_dir()
    output_root.mkdir(parents=True)
    (output_root / "experiment_manifest.json").write_text(
        json.dumps({"config_fingerprint": "different"})
    )
    with pytest.raises(RuntimeError, match="already contains config"):
        await ExperimentRunner(config).run_all()

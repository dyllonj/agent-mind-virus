from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mindvirus.config import ModelConfig, load_config
from mindvirus.costing import _p90_token_count, project_tinker_config_costs

ROOT = Path(__file__).resolve().parents[1]


def test_tinker_projection_uses_frozen_catalog_and_variant_call_count(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "tinker-cost-fixture"
    config.output_dir = tmp_path / "runs"
    config.tinker_catalog_path = ROOT / "frozen/tinker-models-2026-08-28.json"
    config.max_tinker_cost_usd = 1.0
    config.matrix.models = [
        ModelConfig(
            backend="tinker_native",
            model="Qwen/Qwen3-8B",
            variant_id="qwen3_8b_fixture",
            renderer="qwen3_disable_thinking",
            temperature=0.0,
            max_tokens=1200,
            context_window=32768,
            context_safety_tokens=1024,
            max_in_flight=1,
            timeout_seconds=None,
            max_retries=0,
            retry_policy="sdk_default",
            api_key_env="TINKER_API_KEY",
            allow_default_project=True,
        )
    ]
    config.judge.mode = "deterministic"
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    trace = {
        "groups": {
            "host": {
                "mean_input_tokens_per_call": 100.0,
                "mean_output_tokens_per_call": 10.0,
            },
            "judge": {
                "mean_input_tokens_per_call": 0.0,
                "mean_output_tokens_per_call": 0.0,
            },
        }
    }

    projection = project_tinker_config_costs(trace, [path])
    plan = projection["configurations"][0]
    host = plan["host"][0]
    assert plan["frozen_catalog_sha256"] == (
        "47b37219799b075c011546cd2743d5a7e70eacb27a82173e226ee53171815a76"
    )
    assert host["maximum_calls"] == 272
    assert host["projected_input_tokens"] == 27_200
    assert host["projected_output_tokens"] == 2_720
    assert host["input_usd_per_mtok"] == pytest.approx(0.195)
    assert host["output_usd_per_mtok"] == pytest.approx(0.60)
    assert host["projected_cost_usd"] == pytest.approx((27_200 * 0.195 + 2_720 * 0.60) / 1_000_000)
    assert plan["projected_cost_within_hard_budget"]


def _tinker_plan_manifest(tmp_path: Path, budget: float) -> Path:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "tinker-cost-p90-fixture"
    config.output_dir = tmp_path / "runs"
    config.tinker_catalog_path = ROOT / "frozen/tinker-models-2026-08-28.json"
    config.max_tinker_cost_usd = budget
    config.matrix.models = [
        ModelConfig(
            backend="tinker_native",
            model="Qwen/Qwen3-8B",
            variant_id="qwen3_8b_fixture",
            renderer="qwen3_disable_thinking",
            temperature=0.0,
            max_tokens=1200,
            context_window=32768,
            context_safety_tokens=1024,
            max_in_flight=1,
            timeout_seconds=None,
            max_retries=0,
            retry_policy="sdk_default",
            api_key_env="TINKER_API_KEY",
            allow_default_project=True,
        )
    ]
    config.judge.mode = "deterministic"
    path = tmp_path / f"plan-{budget}.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return path


def _p90_trace() -> dict[str, object]:
    return {
        "groups": {
            "host": {
                "mean_input_tokens_per_call": 100.0,
                "mean_output_tokens_per_call": 10.0,
                "p90_input_tokens_per_call": 400.0,
                "p90_output_tokens_per_call": 40.0,
            },
            "judge": {
                "mean_input_tokens_per_call": 0.0,
                "mean_output_tokens_per_call": 0.0,
                "p90_input_tokens_per_call": 0.0,
                "p90_output_tokens_per_call": 0.0,
            },
        }
    }


def test_p90_token_count_uses_nearest_rank() -> None:
    assert _p90_token_count([]) == 0.0
    assert _p90_token_count([5]) == 5.0
    assert _p90_token_count([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) == 90.0


def test_p90_projection_drives_hard_budget_verdict(tmp_path: Path) -> None:
    mean_cost = (272 * 100 * 0.195 + 272 * 10 * 0.60) / 1_000_000
    p90_cost = (272 * 400 * 0.195 + 272 * 40 * 0.60) / 1_000_000

    tight = project_tinker_config_costs(
        _p90_trace(), [_tinker_plan_manifest(tmp_path, mean_cost + 0.001)]
    )
    plan = tight["configurations"][0]
    host = plan["host"][0]
    assert plan["projected_cost_usd"] == pytest.approx(mean_cost)
    assert plan["p90_projected_cost_usd"] == pytest.approx(p90_cost)
    assert host["p90_projected_input_tokens"] == 272 * 400
    assert host["p90_projected_output_tokens"] == 272 * 40
    assert plan["projected_cost_usd"] < plan["hard_tinker_budget_usd"]
    assert not plan["projected_cost_within_hard_budget"]

    generous = project_tinker_config_costs(
        _p90_trace(), [_tinker_plan_manifest(tmp_path, p90_cost + 0.001)]
    )
    assert generous["configurations"][0]["projected_cost_within_hard_budget"]
    assert "p90" in tight["method_note"]
    assert "planning estimates" in tight["method_note"]


def test_projection_falls_back_to_mean_when_p90_fields_are_absent(tmp_path: Path) -> None:
    trace = {
        "groups": {
            "host": {
                "mean_input_tokens_per_call": 100.0,
                "mean_output_tokens_per_call": 10.0,
            },
            "judge": {
                "mean_input_tokens_per_call": 0.0,
                "mean_output_tokens_per_call": 0.0,
            },
        }
    }
    projection = project_tinker_config_costs(trace, [_tinker_plan_manifest(tmp_path, 1.0)])
    plan = projection["configurations"][0]
    assert plan["p90_projected_cost_usd"] == pytest.approx(plan["projected_cost_usd"])
    assert plan["projected_cost_within_hard_budget"]

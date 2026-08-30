from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import Result
from typer.testing import CliRunner

from mindvirus.cli import app
from mindvirus.config import ExperimentConfig, ModelConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _base_config(tmp_path: Path, experiment_id: str) -> ExperimentConfig:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = experiment_id
    config.output_dir = tmp_path / "runs"
    return config


def _write_manifest(config: ExperimentConfig, path: Path) -> Path:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return path


def _combined_output(result: Result) -> str:
    return result.output + result.stderr


def test_run_refuses_paid_matrix_model_without_authorization(tmp_path: Path) -> None:
    config = _base_config(tmp_path, "cli-paid-matrix")
    config.matrix.models = [
        ModelConfig(backend="litellm", model="openai/fixture-paid", api_key_env="OPENAI_API_KEY")
    ]
    manifest = _write_manifest(config, tmp_path / "paid.yaml")
    result = runner.invoke(app, ["run", str(manifest)])
    assert result.exit_code != 0
    assert "--authorize-paid-calls" in _combined_output(result)
    assert not (tmp_path / "runs").exists()


def test_run_refuses_paid_judge_model_without_authorization(tmp_path: Path) -> None:
    config = _base_config(tmp_path, "cli-paid-judge")
    config.judge.mode = "llm"
    config.judge.models = [
        ModelConfig(backend="litellm", model="openai/fixture-judge", api_key_env="OPENAI_API_KEY")
    ]
    manifest = _write_manifest(config, tmp_path / "paid-judge.yaml")
    result = runner.invoke(app, ["run", str(manifest)])
    assert result.exit_code != 0
    assert "--authorize-paid-calls" in _combined_output(result)
    assert not (tmp_path / "runs").exists()


def test_run_allows_mock_manifest_without_authorization(tmp_path: Path) -> None:
    config = _base_config(tmp_path, "cli-mock-ok")
    config.matrix.replicates = 1
    manifest = _write_manifest(config, tmp_path / "mock.yaml")
    result = runner.invoke(app, ["run", str(manifest)])
    assert result.exit_code == 0, _combined_output(result)
    assert "Finished 2 runs" in _combined_output(result)

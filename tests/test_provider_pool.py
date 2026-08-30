from __future__ import annotations

from pathlib import Path

import pytest

from mindvirus.config import ModelConfig
from mindvirus.provider_pool import ProviderPool


def _tinker_config() -> ModelConfig:
    return ModelConfig(
        backend="tinker_native",
        model="fixture/model-8b",
        variant_id="fixture_8b",
        renderer="fixture_renderer",
        max_tokens=64,
        context_window=1024,
        max_in_flight=1,
        timeout_seconds=None,
        max_retries=0,
        retry_policy="sdk_default",
        api_key_env="FIXTURE_TINKER_API_KEY",
        allow_default_project=True,
    )


@pytest.mark.parametrize("budget", [0.0, -1.0])
def test_tinker_client_rejects_non_positive_budget(tmp_path: Path, budget: float) -> None:
    pool = ProviderPool(
        tinker_catalog_path=tmp_path / "catalog.json",
        max_tinker_cost_usd=budget,
    )
    with pytest.raises(RuntimeError, match="positive max_tinker_cost_usd"):
        pool.client(_tinker_config())


def test_tinker_client_still_requires_a_budget(tmp_path: Path) -> None:
    pool = ProviderPool(tinker_catalog_path=tmp_path / "catalog.json")
    with pytest.raises(RuntimeError, match="positive max_tinker_cost_usd"):
        pool.client(_tinker_config())

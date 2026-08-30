from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mindvirus.catalog import TinkerCatalogSnapshot, load_tinker_catalog


def _models() -> list[dict[str, str]]:
    return [
        {
            "name": "Fixture 8B",
            "tinker_id": "fixture/model-8b",
            "context": "32K",
            "prefill": "$0.20",
            "cached_prefill": "$0.04",
            "sample": "$0.60",
            "type": "Hybrid",
            "arch": "Dense",
            "size": "Small",
        }
    ]


def _digest(models: list[dict[str, str]]) -> str:
    canonical = json.dumps(models, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def test_catalog_snapshot_parses_exact_prices_and_context(tmp_path: Path) -> None:
    models = _models()
    snapshot = TinkerCatalogSnapshot(
        source_url="https://example.invalid/models.json",
        retrieved_at="2026-08-28T00:00:00+00:00",
        catalog_sha256=_digest(models),
        models=models,
    )
    path = tmp_path / "catalog.json"
    path.write_text(snapshot.model_dump_json())
    entry = load_tinker_catalog(path).entry("fixture/model-8b")
    assert entry.context_tokens == 32 * 1024
    assert entry.prefill_usd_per_mtok == pytest.approx(0.2)
    assert entry.cached_prefill_usd_per_mtok == pytest.approx(0.04)
    assert entry.sample_usd_per_mtok == pytest.approx(0.6)


def test_catalog_snapshot_detects_tampering() -> None:
    models = _models()
    snapshot = TinkerCatalogSnapshot(
        source_url="https://example.invalid/models.json",
        retrieved_at="2026-08-28T00:00:00+00:00",
        catalog_sha256="0" * 64,
        models=models,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        snapshot.verified_entries()

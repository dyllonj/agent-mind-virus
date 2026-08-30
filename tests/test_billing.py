from __future__ import annotations

from pathlib import Path

import pytest

from mindvirus.billing import aggregate_tinker_billing_events
from mindvirus.catalog import load_tinker_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_billing_reconciliation_filters_session_and_applies_cache_price() -> None:
    catalog = load_tinker_catalog(ROOT / "frozen/tinker-models-2026-08-28.json")
    events = [
        {
            "session_id": "session-study",
            "base_model": "resolved-qwen",
            "project_id": "project-secretish",
            "bucket_start": "2026-08-28T00:00:00Z",
            "bucket_end": "2026-08-28T01:00:00Z",
            "event_info": {
                "type": "sampling_prefill",
                "cached": False,
                "token_count": 1_000_000,
            },
        },
        {
            "session_id": "session-study",
            "base_model": "resolved-qwen",
            "project_id": "project-secretish",
            "event_info": {
                "type": "sampling_prefill",
                "cached": True,
                "token_count": 500_000,
            },
        },
        {
            "session_id": "session-study",
            "base_model": "resolved-qwen",
            "project_id": "project-secretish",
            "event_info": {"type": "sampling_sample", "token_count": 100_000},
        },
        {
            "session_id": "other-session",
            "base_model": "resolved-qwen",
            "event_info": {"type": "sampling_sample", "token_count": 9_999_999},
        },
    ]
    result = aggregate_tinker_billing_events(
        events,
        session_ids={"session-study"},
        catalog=catalog,
        model_aliases={"resolved-qwen": "Qwen/Qwen3-8B"},
    )
    totals = result["totals"]
    assert result["matched_event_count"] == 3
    assert totals["uncached_prefill_tokens"] == 1_000_000
    assert totals["cached_prefill_tokens"] == 500_000
    assert totals["sample_tokens"] == 100_000
    assert totals["billing_cost_usd"] == pytest.approx(0.195 + 0.5 * 0.039 + 0.1 * 0.60)
    assert totals["uncached_equivalent_cost_usd"] == pytest.approx(1.5 * 0.195 + 0.1 * 0.60)
    assert "project_id" not in result["matched_events"][0]
    assert result["matched_events"][0]["project_id_sha256_prefix"]


def test_billing_quarantines_unknown_models_and_counts_missing_token_counts() -> None:
    catalog = load_tinker_catalog(ROOT / "frozen/tinker-models-2026-08-28.json")
    events = [
        {
            "session_id": "session-study",
            "base_model": "mystery-model",
            "event_info": {"type": "sampling_sample", "token_count": 999},
        },
        {
            "session_id": "session-study",
            "base_model": "resolved-qwen",
            "event_info": {"type": "sampling_prefill", "cached": False},
        },
        {
            "session_id": "session-study",
            "base_model": "resolved-qwen",
            "event_info": {"type": "sampling_sample", "token_count": 100},
        },
    ]
    result = aggregate_tinker_billing_events(
        events,
        session_ids={"session-study"},
        catalog=catalog,
        model_aliases={"resolved-qwen": "Qwen/Qwen3-8B"},
    )
    assert result["matched_event_count"] == 2
    assert result["totals"]["sample_tokens"] == 100
    assert result["totals"]["uncached_prefill_tokens"] == 0
    assert result["missing_token_count_events"] == 1
    assert len(result["unmatched_events"]) == 1
    unmatched = result["unmatched_events"][0]
    assert unmatched["base_model"] == "mystery-model"
    assert unmatched["event_type"] == "sampling_sample"
    assert "absent from the frozen catalog" in unmatched["reason"]

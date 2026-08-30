from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import selected_artifact_dirs
from .schemas import RunSummary

CALL_COLUMNS = [
    "run_id",
    "phase",
    "role",
    "call_id",
    "provider",
    "transport",
    "model_requested",
    "model_resolved",
    "variant_id",
    "renderer",
    "parse_termination",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "calculated_cost_usd",
    "reported_cost_usd",
    "effective_cost_usd",
    "queue_ms",
    "sample_ms",
    "context_required_tokens",
    "context_window",
    "context_utilization",
    "tool_call_count",
    "malformed_tool_argument_count",
]

FAILURE_COLUMNS = [
    "run_id",
    "attempt",
    "model_variant_id",
    "completed",
    "error_class",
    "error",
    "artifact_dir",
]


def provider_diagnostics(
    experiment_root: Path,
    summaries: list[RunSummary],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build call-level provenance, grouped provider metrics, and immutable failure records."""

    calls: list[dict[str, Any]] = []
    tool_counts: dict[str, tuple[int, int]] = {}
    for run_dir, artifact_dir in selected_artifact_dirs(experiment_root):
        events_path = artifact_dir / "events.jsonl"
        if not events_path.exists():
            continue
        tool_total = 0
        tool_errors = 0
        for line in events_path.read_text().splitlines():
            event = json.loads(line)
            payload = event.get("payload", {})
            if event.get("kind") == "tool_result":
                tool_total += 1
                tool_errors += int(bool(payload.get("is_error")))
                continue
            if event.get("kind") != "model_response":
                continue
            response = payload.get("response", {})
            provider = response.get("provider", {})
            usage = response.get("usage", {})
            raw = response.get("raw") or {}
            context_required = _optional_int(raw.get("context_required_tokens"))
            context_window = _optional_int(raw.get("context_window"))
            context_utilization = (
                context_required / context_window
                if context_required is not None
                and context_window is not None
                and context_window > 0
                else None
            )
            calculated = _optional_float(usage.get("calculated_cost_usd"))
            reported = _optional_float(usage.get("reported_cost_usd"))
            calls.append(
                {
                    "run_id": run_dir.name,
                    "phase": str(payload.get("phase", "")),
                    "role": "judge" if payload.get("phase") == "judge" else "host",
                    "call_id": str(payload.get("call_id", "")),
                    "provider": str(provider.get("provider", "")),
                    "transport": str(provider.get("transport", "")),
                    "model_requested": str(provider.get("model_requested", "")),
                    "model_resolved": provider.get("model_resolved"),
                    "variant_id": str(provider.get("variant_id", "")),
                    "renderer": provider.get("renderer"),
                    "parse_termination": response.get("parse_termination"),
                    "finish_reason": response.get("finish_reason"),
                    "input_tokens": int(usage.get("input_tokens", 0)),
                    "output_tokens": int(usage.get("output_tokens", 0)),
                    "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0)),
                    "calculated_cost_usd": calculated,
                    "reported_cost_usd": reported,
                    "effective_cost_usd": reported if reported is not None else calculated,
                    "queue_ms": _optional_float(provider.get("queue_ms")),
                    "sample_ms": _optional_float(provider.get("sample_ms")),
                    "context_required_tokens": context_required,
                    "context_window": context_window,
                    "context_utilization": context_utilization,
                    "tool_call_count": len(response.get("tool_calls") or []),
                    "malformed_tool_argument_count": len(raw.get("malformed_tool_arguments") or []),
                }
            )
        tool_counts[run_dir.name] = (tool_total, tool_errors)

    call_frame = pd.DataFrame(calls, columns=CALL_COLUMNS)
    failure_frame = _technical_failures(experiment_root)
    grouped = _group_provider_calls(call_frame, summaries, tool_counts)
    summary_payload = {
        "run_status_by_host_variant": _run_status_records(summaries),
        "provider_groups": json.loads(grouped.to_json(orient="records")),
        "technical_failure_count": len(failure_frame),
        "technical_failures_by_class": (
            failure_frame["error_class"].value_counts().sort_index().to_dict()
            if not failure_frame.empty
            else {}
        ),
    }
    return call_frame, grouped, failure_frame, summary_payload


def _group_provider_calls(
    call_frame: pd.DataFrame,
    summaries: list[RunSummary],
    tool_counts: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    columns = [
        "role",
        "provider",
        "variant_id",
        "model_requested",
        "renderer",
        "calls",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "effective_cost_usd",
        "queue_ms_p50",
        "queue_ms_p95",
        "sample_ms_p50",
        "sample_ms_p95",
        "context_utilization_max",
        "context_utilization_p95",
        "malformed_tool_argument_calls",
        "tool_calls",
        "tool_error_results",
        "valid_tool_result_rate",
    ]
    if call_frame.empty:
        return pd.DataFrame(columns=columns)
    host_variant_by_run = {
        summary.run_id: summary.model_variant_id or summary.model for summary in summaries
    }
    records: list[dict[str, Any]] = []
    dimensions = ["role", "provider", "variant_id", "model_requested", "renderer"]
    for keys, group in call_frame.groupby(dimensions, dropna=False, sort=True):
        role, provider, variant_id, model_requested, renderer = keys
        tool_total = 0
        tool_errors = 0
        if role == "host":
            matching_runs = {
                run_id
                for run_id, host_variant in host_variant_by_run.items()
                if host_variant == variant_id
            }
            for run_id in matching_runs:
                run_tool_total, run_tool_errors = tool_counts.get(run_id, (0, 0))
                tool_total += run_tool_total
                tool_errors += run_tool_errors
        records.append(
            {
                "role": role,
                "provider": provider,
                "variant_id": variant_id,
                "model_requested": model_requested,
                "renderer": renderer,
                "calls": len(group),
                "input_tokens": int(group["input_tokens"].sum()),
                "output_tokens": int(group["output_tokens"].sum()),
                "cache_read_input_tokens": int(group["cache_read_input_tokens"].sum()),
                "effective_cost_usd": float(group["effective_cost_usd"].sum(min_count=1))
                if group["effective_cost_usd"].notna().any()
                else None,
                "queue_ms_p50": _quantile(group["queue_ms"], 0.5),
                "queue_ms_p95": _quantile(group["queue_ms"], 0.95),
                "sample_ms_p50": _quantile(group["sample_ms"], 0.5),
                "sample_ms_p95": _quantile(group["sample_ms"], 0.95),
                "context_utilization_max": _maximum(group["context_utilization"]),
                "context_utilization_p95": _quantile(group["context_utilization"], 0.95),
                "malformed_tool_argument_calls": int(
                    (group["malformed_tool_argument_count"] > 0).sum()
                ),
                "tool_calls": tool_total,
                "tool_error_results": tool_errors,
                "valid_tool_result_rate": (
                    (tool_total - tool_errors) / tool_total if tool_total else None
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _technical_failures(experiment_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for run_dir in sorted((experiment_root / "runs").glob("*")):
        attempts_dir = run_dir / "attempts"
        if attempts_dir.is_dir():
            candidates = sorted(attempts_dir.glob("*/summary.json"))
        else:
            candidates = [run_dir / "summary.json"]
        for path in candidates:
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if payload.get("completed"):
                continue
            error = str(payload.get("error") or "unknown technical failure")
            records.append(
                {
                    "run_id": str(payload.get("run_id", run_dir.name)),
                    "attempt": path.parent.name if attempts_dir.is_dir() else "legacy",
                    "model_variant_id": str(
                        payload.get("model_variant_id") or payload.get("model", "")
                    ),
                    "completed": False,
                    "error_class": error.split(":", 1)[0],
                    "error": error,
                    "artifact_dir": str(path.parent),
                }
            )
    return pd.DataFrame(records, columns=FAILURE_COLUMNS)


def _run_status_records(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.model_variant_id or summary.model, []).append(summary)
    return [
        {
            "model_variant_id": variant_id,
            "planned_runs": len(items),
            "completed_runs": sum(item.completed for item in items),
            "technical_success_rate": sum(item.completed for item in items) / len(items),
        }
        for variant_id, items in sorted(grouped.items())
    ]


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _quantile(series: pd.Series[Any], probability: float) -> float | None:
    clean = series.dropna()
    return float(clean.quantile(probability)) if not clean.empty else None


def _maximum(series: pd.Series[Any]) -> float | None:
    clean = series.dropna()
    return float(clean.max()) if not clean.empty else None

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

from .artifacts import selected_artifact_dirs
from .catalog import TinkerCatalogSnapshot, load_tinker_catalog
from .config import ExperimentConfig, ModelConfig


def aggregate_tinker_billing_events(
    events: list[dict[str, Any]],
    *,
    session_ids: set[str],
    catalog: TinkerCatalogSnapshot,
    model_aliases: dict[str, str],
) -> dict[str, Any]:
    """Filter native billing events to experiment sessions and calculate frozen-price totals."""

    by_model: dict[str, dict[str, Any]] = {}
    sanitized_events: list[dict[str, Any]] = []
    ignored_event_types: dict[str, int] = {}
    unmatched_events: list[dict[str, Any]] = []
    missing_token_count_events = 0
    for event in events:
        session_id = event.get("session_id")
        if not isinstance(session_id, str) or session_id not in session_ids:
            continue
        info = event.get("event_info")
        if not isinstance(info, dict):
            continue
        event_type = str(info.get("type", "unknown"))
        if event_type not in {"sampling_prefill", "sampling_sample"}:
            ignored_event_types[event_type] = ignored_event_types.get(event_type, 0) + 1
            continue
        billing_model = str(event.get("base_model") or "")
        catalog_model = model_aliases.get(billing_model, billing_model)
        try:
            entry = catalog.entry(catalog_model)
        except ValueError:
            unmatched_events.append(
                {
                    "session_id": session_id,
                    "base_model": billing_model,
                    "event_type": event_type,
                    "reason": (
                        f"model {catalog_model!r} is absent from the frozen catalog and alias "
                        "map; event quarantined so reconciliation can continue"
                    ),
                }
            )
            continue
        record = by_model.setdefault(
            catalog_model,
            {
                "tinker_id": catalog_model,
                "uncached_prefill_tokens": 0,
                "cached_prefill_tokens": 0,
                "sample_tokens": 0,
                "billing_cost_usd": 0.0,
                "uncached_equivalent_cost_usd": 0.0,
                "event_count": 0,
            },
        )
        raw_token_count = info.get("token_count")
        if raw_token_count is None:
            missing_token_count_events += 1
            token_count = 0
        else:
            token_count = int(raw_token_count)
        if token_count < 0:
            raise ValueError("billing token count cannot be negative")
        cached = bool(info.get("cached", False)) if event_type == "sampling_prefill" else False
        if event_type == "sampling_prefill":
            token_field = "cached_prefill_tokens" if cached else "uncached_prefill_tokens"
            record[token_field] += token_count
            applied_rate = (
                entry.cached_prefill_usd_per_mtok if cached else entry.prefill_usd_per_mtok
            )
            record["billing_cost_usd"] += token_count * applied_rate / 1_000_000
            record["uncached_equivalent_cost_usd"] += (
                token_count * entry.prefill_usd_per_mtok / 1_000_000
            )
        else:
            record["sample_tokens"] += token_count
            sample_cost = token_count * entry.sample_usd_per_mtok / 1_000_000
            record["billing_cost_usd"] += sample_cost
            record["uncached_equivalent_cost_usd"] += sample_cost
        record["event_count"] += 1
        project_id = event.get("project_id")
        sanitized_events.append(
            {
                "bucket_start": event.get("bucket_start"),
                "bucket_end": event.get("bucket_end"),
                "base_model": billing_model,
                "session_id": session_id,
                "project_id_sha256_prefix": _value_hash(project_id),
                "event_type": event_type,
                "cached": cached,
                "token_count": token_count,
            }
        )
    models = [by_model[key] for key in sorted(by_model)]
    return {
        "matched_event_count": len(sanitized_events),
        "models": models,
        "totals": {
            "uncached_prefill_tokens": sum(int(item["uncached_prefill_tokens"]) for item in models),
            "cached_prefill_tokens": sum(int(item["cached_prefill_tokens"]) for item in models),
            "sample_tokens": sum(int(item["sample_tokens"]) for item in models),
            "billing_cost_usd": sum(float(item["billing_cost_usd"]) for item in models),
            "uncached_equivalent_cost_usd": sum(
                float(item["uncached_equivalent_cost_usd"]) for item in models
            ),
        },
        "ignored_event_types": ignored_event_types,
        "unmatched_events": unmatched_events,
        "missing_token_count_events": missing_token_count_events,
        "matched_events": sanitized_events,
    }


def reconcile_tinker_billing_payload(
    experiment_root: Path,
    provider_payload: dict[str, Any],
    *,
    starting_on: str,
    ending_before: str,
) -> dict[str, Any]:
    """Reconcile a provider billing response against all retained experiment sessions."""

    config = _load_resolved_config(experiment_root)
    recorded_fingerprint = _recorded_config_fingerprint(experiment_root)
    if config.tinker_catalog_path is None:
        raise ValueError("experiment manifest has no frozen Tinker catalog")
    catalog = load_tinker_catalog(config.tinker_catalog_path)
    executions = _provider_execution_snapshots(experiment_root)
    session_ids: set[str] = set()
    model_aliases: dict[str, str] = {}
    ledger_settled = 0.0
    ledger_uncertain = 0.0
    execution_records: list[dict[str, Any]] = []
    for snapshot in executions:
        session = snapshot.get("tinker_session") or {}
        execution_session_ids: set[str] = set()
        for model in session.get("models", []):
            session_id = model.get("session_id")
            if session_id:
                session_ids.add(str(session_id))
                execution_session_ids.add(str(session_id))
            requested = str(model.get("model_requested") or "")
            resolved = str(model.get("model_resolved") or "")
            if requested:
                model_aliases[requested] = requested
            if resolved and requested:
                model_aliases[resolved] = requested
        budget = snapshot.get("tinker_budget") or {}
        execution_settled = float(budget.get("execution_settled_usd", budget.get("settled_usd", 0)))
        execution_uncertain = float(
            budget.get("execution_uncertain_usd", budget.get("uncertain_usd", 0))
        )
        ledger_settled += execution_settled
        ledger_uncertain += execution_uncertain
        execution_records.append(
            {
                "execution_id": snapshot.get("execution_id"),
                "session_ids": sorted(execution_session_ids),
                "execution_settled_usd": execution_settled,
                "execution_uncertain_usd": execution_uncertain,
                "cumulative_committed_usd": float(budget.get("committed_usd", 0)),
            }
        )
    if not session_ids:
        raise ValueError("no retained Tinker session IDs were found for this experiment")

    raw_events = provider_payload.get("data")
    if not isinstance(raw_events, list) or not all(isinstance(event, dict) for event in raw_events):
        raise ValueError("provider billing payload has no valid data event list")
    billing = aggregate_tinker_billing_events(
        raw_events,
        session_ids=session_ids,
        catalog=catalog,
        model_aliases=model_aliases,
    )
    trace = _selected_trace_usage(experiment_root)
    sessions_payload = provider_payload.get("sessions")
    session_metadata = sessions_payload if isinstance(sessions_payload, dict) else {}
    metadata_checks = []
    for session_id in sorted(session_ids):
        raw = session_metadata.get(session_id)
        metadata = raw.get("user_metadata") if isinstance(raw, dict) else None
        metadata_checks.append(
            {
                "session_id": session_id,
                "found": isinstance(raw, dict),
                "experiment_id_matches": (
                    metadata.get("experiment_id") == config.experiment_id
                    if isinstance(metadata, dict)
                    else None
                ),
                "config_fingerprint_matches": (
                    metadata.get("config_fingerprint") == recorded_fingerprint
                    if isinstance(metadata, dict)
                    else None
                ),
                "execution_id": metadata.get("execution_id")
                if isinstance(metadata, dict)
                else None,
            }
        )
    totals = billing["totals"]
    billed_prefill = int(totals["uncached_prefill_tokens"]) + int(totals["cached_prefill_tokens"])
    return {
        "status": "reconciled" if billing["matched_event_count"] else "provider_data_pending",
        "experiment_root": str(experiment_root.resolve()),
        "experiment_id": config.experiment_id,
        "config_fingerprint": recorded_fingerprint,
        "window": {"starting_on": starting_on, "ending_before": ending_before},
        "catalog_sha256": catalog.catalog_sha256,
        "session_ids": sorted(session_ids),
        "session_metadata_checks": metadata_checks,
        "provider_executions": execution_records,
        "billing": billing,
        "selected_trace_usage": trace,
        "ledger": {
            "settled_usd": ledger_settled,
            "uncertain_usd": ledger_uncertain,
            "committed_usd": ledger_settled + ledger_uncertain,
        },
        "differences": {
            "billing_minus_selected_trace_input_tokens": billed_prefill
            - int(trace["input_tokens"]),
            "billing_minus_selected_trace_output_tokens": int(totals["sample_tokens"])
            - int(trace["output_tokens"]),
            "billing_uncached_equivalent_minus_trace_calculated_usd": float(
                totals["uncached_equivalent_cost_usd"]
            )
            - float(trace["calculated_cost_usd"]),
            "billing_uncached_equivalent_minus_ledger_committed_usd": float(
                totals["uncached_equivalent_cost_usd"]
            )
            - ledger_settled
            - ledger_uncertain,
        },
        "interpretation": (
            "Selected-trace differences can include failed or superseded attempts. Billing may "
            "remain incomplete for several hours; provider_data_pending is not a zero-usage result."
        ),
    }


async def fetch_and_reconcile_tinker_billing(
    experiment_root: Path,
    *,
    starting_on: str,
    ending_before: str,
) -> dict[str, Any]:
    """Fetch organization billing data, then apply deterministic session reconciliation."""

    config = _load_resolved_config(experiment_root)
    tinker_models = _active_tinker_models(config)
    if not tinker_models:
        raise ValueError("experiment does not use tinker_native")
    first = tinker_models[0]
    if first.api_key_env is None:
        raise ValueError("Tinker model has no api_key_env")
    api_key = os.environ.get(first.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing Tinker API key environment variable {first.api_key_env}")
    project_id: str | None = None
    if first.project_id_env is not None:
        project_id = os.environ.get(first.project_id_env)
        if not project_id:
            raise RuntimeError(
                f"missing Tinker project environment variable {first.project_id_env}"
            )
    try:
        tinker = importlib.import_module("tinker")
    except ImportError as exc:
        raise RuntimeError(
            "billing reconciliation requires Tinker dependencies; run `uv sync --extra tinker`"
        ) from exc
    service = tinker.ServiceClient(project_id=project_id, api_key=api_key)
    rest = service.create_rest_client()
    try:
        response = await rest.get_billing_usage_async(starting_on, ending_before)
        payload = response.model_dump(mode="json")
    finally:
        holder = getattr(rest, "holder", None)
        close = getattr(holder, "close", None)
        if callable(close):
            close()
    return reconcile_tinker_billing_payload(
        experiment_root,
        payload,
        starting_on=starting_on,
        ending_before=ending_before,
    )


def _provider_execution_snapshots(experiment_root: Path) -> list[dict[str, Any]]:
    paths = sorted((experiment_root / "provider_executions").glob("*/final.json"))
    if not paths and (experiment_root / "provider_ledger.json").exists():
        paths = [experiment_root / "provider_ledger.json"]
    return [json.loads(path.read_text()) for path in paths]


def _load_resolved_config(experiment_root: Path) -> ExperimentConfig:
    manifest_path = experiment_root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing experiment manifest {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    return ExperimentConfig.model_validate(payload.get("config"))


def _recorded_config_fingerprint(experiment_root: Path) -> str:
    manifest_path = experiment_root / "experiment_manifest.json"
    payload = json.loads(manifest_path.read_text())
    fingerprint = payload.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"experiment manifest {manifest_path} has no config fingerprint")
    return fingerprint


def _active_tinker_models(config: ExperimentConfig) -> list[ModelConfig]:
    models = list(config.matrix.models)
    if config.judge.mode in {"llm", "hybrid"}:
        models.extend(config.judge.models)
    return [model for model in models if model.backend == "tinker_native"]


def _selected_trace_usage(experiment_root: Path) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    calculated_cost = 0.0
    calls = 0
    for _, artifact_dir in selected_artifact_dirs(experiment_root):
        events_path = artifact_dir / "events.jsonl"
        if not events_path.exists():
            continue
        for line in events_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("kind") != "model_response":
                continue
            response = event.get("payload", {}).get("response", {})
            if response.get("provider", {}).get("provider") != "tinker":
                continue
            usage = response.get("usage", {})
            calls += 1
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
            calculated_cost += float(usage.get("calculated_cost_usd") or 0)
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calculated_cost_usd": calculated_cost,
    }


def _value_hash(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol, cast

from .artifacts import selected_artifact_dirs
from .budget import calculate_uncached_cost_usd
from .catalog import load_tinker_catalog
from .config import ExperimentConfig, ModelConfig, expand_matrix, load_config
from .runner import validate_experiment


class TokenCounter(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        count_response_tokens: bool | None = None,
    ) -> int: ...


def _p90_token_count(values: list[int]) -> float:
    """Nearest-rank 90th percentile of per-call token counts; conservative for verdicts."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(max(1, math.ceil(0.9 * len(ordered))), len(ordered))
    return float(ordered[rank - 1])


def estimate_trace_tokens(
    experiment_root: Path,
    *,
    host_tokenizer_model: str,
    judge_tokenizer_model: str,
) -> dict[str, Any]:
    """Estimate tokens locally from the exact traced payloads; this makes no API calls."""
    counter = _load_token_counter()
    totals = {
        "host": {"request_count": 0, "response_count": 0, "input_tokens": 0, "output_tokens": 0},
        "judge": {
            "request_count": 0,
            "response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }
    samples: dict[str, dict[str, list[int]]] = {
        "host": {"input": [], "output": []},
        "judge": {"input": [], "output": []},
    }
    event_paths = _selected_event_paths(experiment_root)
    if not event_paths:
        raise ValueError(f"no event traces found below {experiment_root}")

    for path in event_paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"event is not an object in {path}:{line_number}")
            kind = event.get("kind")
            if kind not in {"model_request", "model_response"}:
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"event payload is not an object in {path}:{line_number}")
            group = "judge" if payload.get("phase") == "judge" else "host"
            model = judge_tokenizer_model if group == "judge" else host_tokenizer_model
            group_totals = totals[group]
            if kind == "model_request":
                request = payload.get("request")
                if not isinstance(request, dict):
                    raise ValueError(f"model request is missing in {path}:{line_number}")
                group_totals["request_count"] += 1
                counted_input = _count_request(counter, model, request)
                group_totals["input_tokens"] += counted_input
                samples[group]["input"].append(counted_input)
            else:
                response = payload.get("response")
                if not isinstance(response, dict):
                    raise ValueError(f"model response is missing in {path}:{line_number}")
                group_totals["response_count"] += 1
                counted_output = _count_response(counter, model, response)
                group_totals["output_tokens"] += counted_output
                samples[group]["output"].append(counted_output)

    result_groups: dict[str, Any] = {}
    for group, values in totals.items():
        requests = values["request_count"]
        responses = values["response_count"]
        if requests != responses:
            raise ValueError(
                f"{group} trace has {requests} requests but {responses} responses; "
                "repair or exclude incomplete runs before cost projection"
            )
        result_groups[group] = {
            **values,
            "tokenizer_model": (
                judge_tokenizer_model if group == "judge" else host_tokenizer_model
            ),
            "mean_input_tokens_per_call": values["input_tokens"] / requests if requests else 0.0,
            "mean_output_tokens_per_call": values["output_tokens"] / responses
            if responses
            else 0.0,
            "p90_input_tokens_per_call": _p90_token_count(samples[group]["input"]),
            "p90_output_tokens_per_call": _p90_token_count(samples[group]["output"]),
        }
    return {
        "method": (
            "offline LiteLLM token-counter approximation over exact traced system prompts, "
            "messages, tool schemas, tool calls, and visible responses; no API calls"
        ),
        "experiment_root": str(experiment_root.resolve()),
        "trace_files": len(event_paths),
        "groups": result_groups,
    }


def _selected_event_paths(experiment_root: Path) -> list[Path]:
    return [
        artifact_dir / "events.jsonl"
        for _, artifact_dir in selected_artifact_dirs(experiment_root)
        if (artifact_dir / "events.jsonl").exists()
    ]


def project_config_costs(
    trace_estimate: dict[str, Any],
    config_paths: list[Path],
    *,
    host_input_usd_per_mtok: float,
    host_output_usd_per_mtok: float,
    judge_input_usd_per_mtok: float,
    judge_output_usd_per_mtok: float,
    token_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Project plan costs using a measured per-call token profile and explicit prices."""
    if token_multiplier <= 0:
        raise ValueError("token_multiplier must be positive")
    groups = trace_estimate.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("trace estimate is missing groups")
    host = groups.get("host")
    judge = groups.get("judge")
    if not isinstance(host, dict) or not isinstance(judge, dict):
        raise ValueError("trace estimate must contain host and judge groups")
    host_input_mean = float(host["mean_input_tokens_per_call"])
    host_output_mean = float(host["mean_output_tokens_per_call"])
    judge_input_mean = float(judge["mean_input_tokens_per_call"])
    judge_output_mean = float(judge["mean_output_tokens_per_call"])
    if judge_input_mean == 0 or judge_output_mean == 0:
        raise ValueError("judge traces are required for a complete cost projection")

    rows: list[dict[str, Any]] = []
    for path in config_paths:
        config = load_config(path)
        validation = validate_experiment(config)
        judge_calls = int(validation["judge_calls"])
        maximum_calls = int(validation["maximum_total_model_calls"])
        host_calls = maximum_calls - judge_calls
        host_input = math.ceil(host_calls * host_input_mean * token_multiplier)
        host_output = math.ceil(host_calls * host_output_mean * token_multiplier)
        judge_input = math.ceil(judge_calls * judge_input_mean * token_multiplier)
        judge_output = math.ceil(judge_calls * judge_output_mean * token_multiplier)
        host_cost = _token_cost(
            host_input,
            host_output,
            host_input_usd_per_mtok,
            host_output_usd_per_mtok,
        )
        judge_cost = _token_cost(
            judge_input,
            judge_output,
            judge_input_usd_per_mtok,
            judge_output_usd_per_mtok,
        )
        rows.append(
            {
                "config_path": str(path.resolve()),
                "experiment_id": config.experiment_id,
                "rollouts": int(validation["run_count"]),
                "maximum_host_calls": host_calls,
                "judge_calls": judge_calls,
                "maximum_total_calls": maximum_calls,
                "projected_host_input_tokens": host_input,
                "projected_host_output_tokens": host_output,
                "projected_judge_input_tokens": judge_input,
                "projected_judge_output_tokens": judge_output,
                "host_cost_usd": host_cost,
                "judge_cost_usd": judge_cost,
                "total_cost_usd": host_cost + judge_cost,
            }
        )
    return {
        "assumption": (
            "Every configured agent turn uses the maximum allowed tool-follow-up calls, and "
            "each call matches the measured local-preflight token profile after applying the "
            "explicit token multiplier. Provider retries are excluded."
        ),
        "token_multiplier": token_multiplier,
        "prices_usd_per_million_tokens": {
            "host_input": host_input_usd_per_mtok,
            "host_output": host_output_usd_per_mtok,
            "judge_input": judge_input_usd_per_mtok,
            "judge_output": judge_output_usd_per_mtok,
        },
        "configurations": rows,
        "totals": {
            "rollouts": sum(int(row["rollouts"]) for row in rows),
            "maximum_host_calls": sum(int(row["maximum_host_calls"]) for row in rows),
            "judge_calls": sum(int(row["judge_calls"]) for row in rows),
            "maximum_total_calls": sum(int(row["maximum_total_calls"]) for row in rows),
            "projected_host_input_tokens": sum(
                int(row["projected_host_input_tokens"]) for row in rows
            ),
            "projected_host_output_tokens": sum(
                int(row["projected_host_output_tokens"]) for row in rows
            ),
            "projected_judge_input_tokens": sum(
                int(row["projected_judge_input_tokens"]) for row in rows
            ),
            "projected_judge_output_tokens": sum(
                int(row["projected_judge_output_tokens"]) for row in rows
            ),
            "host_cost_usd": sum(float(row["host_cost_usd"]) for row in rows),
            "judge_cost_usd": sum(float(row["judge_cost_usd"]) for row in rows),
            "total_cost_usd": sum(float(row["total_cost_usd"]) for row in rows),
        },
    }


def project_tinker_config_costs(
    trace_estimate: dict[str, Any],
    config_paths: list[Path],
    *,
    token_multiplier: float = 1.0,
    external_judge_input_usd_per_mtok: float | None = None,
    external_judge_output_usd_per_mtok: float | None = None,
) -> dict[str, Any]:
    """Project native Tinker plans from frozen catalogs and a measured local trace."""

    if token_multiplier <= 0:
        raise ValueError("token_multiplier must be positive")
    groups = trace_estimate.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("trace estimate is missing groups")
    host = groups.get("host")
    judge = groups.get("judge")
    if not isinstance(host, dict) or not isinstance(judge, dict):
        raise ValueError("trace estimate must contain host and judge groups")
    host_input_mean = float(host["mean_input_tokens_per_call"])
    host_output_mean = float(host["mean_output_tokens_per_call"])
    judge_input_mean = float(judge["mean_input_tokens_per_call"])
    judge_output_mean = float(judge["mean_output_tokens_per_call"])
    host_input_p90 = float(host.get("p90_input_tokens_per_call", host_input_mean))
    host_output_p90 = float(host.get("p90_output_tokens_per_call", host_output_mean))
    judge_input_p90 = float(judge.get("p90_input_tokens_per_call", judge_input_mean))
    judge_output_p90 = float(judge.get("p90_output_tokens_per_call", judge_output_mean))

    plans: list[dict[str, Any]] = []
    for path in config_paths:
        config = load_config(path)
        validation = validate_experiment(config)
        if any(model.backend != "tinker_native" for model in config.matrix.models):
            raise ValueError(
                f"{path} contains a non-Tinker host; use project_config_costs for mixed or "
                "non-Tinker host plans"
            )
        if config.tinker_catalog_path is None:
            raise ValueError(f"{path} has no frozen Tinker catalog")
        catalog = load_tinker_catalog(config.tinker_catalog_path)
        host_calls_per_run = _maximum_host_calls_per_run(config)
        cells = expand_matrix(config)
        host_rows: list[dict[str, Any]] = []
        for model in config.matrix.models:
            variant_id = model.resolved_variant_id()
            model_runs = sum(cell.model_variant_id == variant_id for cell in cells)
            calls = model_runs * host_calls_per_run
            host_rows.append(
                _projection_row(
                    role="host",
                    model=model,
                    calls=calls,
                    mean_input_tokens=host_input_mean,
                    mean_output_tokens=host_output_mean,
                    p90_input_tokens=host_input_p90,
                    p90_output_tokens=host_output_p90,
                    token_multiplier=token_multiplier,
                    input_usd_per_mtok=catalog.entry(model.model).prefill_usd_per_mtok,
                    output_usd_per_mtok=catalog.entry(model.model).sample_usd_per_mtok,
                )
            )

        judge_rows: list[dict[str, Any]] = []
        if config.judge.mode in {"llm", "hybrid"}:
            if judge_input_mean == 0 or judge_output_mean == 0:
                raise ValueError("judge traces are required for a complete Tinker cost projection")
            judge_calls_per_model = len(cells) * (config.swarm.n_agents - 1)
            for model in config.judge.models:
                if model.backend == "tinker_native":
                    entry = catalog.entry(model.model)
                    input_price = entry.prefill_usd_per_mtok
                    output_price = entry.sample_usd_per_mtok
                elif model.backend == "mock":
                    input_price = 0.0
                    output_price = 0.0
                else:
                    if (
                        external_judge_input_usd_per_mtok is None
                        or external_judge_output_usd_per_mtok is None
                    ):
                        raise ValueError(
                            f"{path} uses external judge {model.model!r}; provide both external "
                            "judge prices explicitly"
                        )
                    input_price = external_judge_input_usd_per_mtok
                    output_price = external_judge_output_usd_per_mtok
                judge_rows.append(
                    _projection_row(
                        role="judge",
                        model=model,
                        calls=judge_calls_per_model,
                        mean_input_tokens=judge_input_mean,
                        mean_output_tokens=judge_output_mean,
                        p90_input_tokens=judge_input_p90,
                        p90_output_tokens=judge_output_p90,
                        token_multiplier=token_multiplier,
                        input_usd_per_mtok=input_price,
                        output_usd_per_mtok=output_price,
                    )
                )

        projected_cost = sum(float(row["projected_cost_usd"]) for row in [*host_rows, *judge_rows])
        p90_projected_cost = sum(
            float(row["p90_projected_cost_usd"]) for row in [*host_rows, *judge_rows]
        )
        hard_budget = config.max_tinker_cost_usd
        plans.append(
            {
                "config_path": str(path.resolve()),
                "experiment_id": config.experiment_id,
                "rollouts": int(validation["run_count"]),
                "frozen_catalog_sha256": catalog.catalog_sha256,
                "host": host_rows,
                "judge": judge_rows,
                "projected_cost_usd": projected_cost,
                "p90_projected_cost_usd": p90_projected_cost,
                "hard_tinker_budget_usd": hard_budget,
                "projected_cost_within_hard_budget": (
                    p90_projected_cost <= hard_budget if hard_budget is not None else None
                ),
            }
        )
    return {
        "method": (
            "Measured local mean tokens per call multiplied by the manifest's maximum logical "
            "call count and explicit sensitivity multiplier. Tinker prices come only from each "
            "manifest's hash-verified frozen catalog; input cost is conservatively uncached."
        ),
        "method_note": (
            "Token counts come from offline LiteLLM/Anthropic tokenizers over traced payloads "
            "and are planning estimates, not provider-measured usage. The p90 per-call token "
            "projection is used for hard-budget verdicts; mean-based fields are retained for "
            "reference."
        ),
        "token_multiplier": token_multiplier,
        "configurations": plans,
        "totals": {
            "rollouts": sum(int(plan["rollouts"]) for plan in plans),
            "projected_cost_usd": sum(float(plan["projected_cost_usd"]) for plan in plans),
            "p90_projected_cost_usd": sum(
                float(plan["p90_projected_cost_usd"]) for plan in plans
            ),
        },
    }


def _maximum_host_calls_per_run(config: ExperimentConfig) -> int:
    clean_agents = config.swarm.n_agents - 1
    origin_rounds = (
        config.swarm.max_rounds
        if config.swarm.origin_active_until_round is None
        else min(config.swarm.origin_active_until_round, config.swarm.max_rounds)
    )
    regular_turns = clean_agents * config.swarm.max_rounds + origin_rounds
    checkpoint_turns = clean_agents * (
        len(config.swarm.context_reset_rounds) + int(config.swarm.final_memory_round)
    )
    return (regular_turns + checkpoint_turns) * config.swarm.max_tool_loops_per_turn


def _projection_row(
    *,
    role: str,
    model: ModelConfig,
    calls: int,
    mean_input_tokens: float,
    mean_output_tokens: float,
    p90_input_tokens: float,
    p90_output_tokens: float,
    token_multiplier: float,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> dict[str, Any]:
    projected_input = math.ceil(calls * mean_input_tokens * token_multiplier)
    projected_output = math.ceil(calls * mean_output_tokens * token_multiplier)
    p90_projected_input = math.ceil(calls * p90_input_tokens * token_multiplier)
    p90_projected_output = math.ceil(calls * p90_output_tokens * token_multiplier)
    mean_input = math.ceil(mean_input_tokens * token_multiplier)
    context_required = (
        mean_input + model.max_tokens + model.context_safety_tokens
        if model.context_window is not None
        else None
    )
    return {
        "role": role,
        "variant_id": model.resolved_variant_id(),
        "backend": model.backend,
        "model": model.model,
        "renderer": model.renderer,
        "maximum_calls": calls,
        "projected_input_tokens": projected_input,
        "projected_output_tokens": projected_output,
        "p90_projected_input_tokens": p90_projected_input,
        "p90_projected_output_tokens": p90_projected_output,
        "input_usd_per_mtok": input_usd_per_mtok,
        "output_usd_per_mtok": output_usd_per_mtok,
        "projected_cost_usd": calculate_uncached_cost_usd(
            input_tokens=projected_input,
            output_tokens=projected_output,
            prefill_usd_per_mtok=input_usd_per_mtok,
            sample_usd_per_mtok=output_usd_per_mtok,
        ),
        "p90_projected_cost_usd": calculate_uncached_cost_usd(
            input_tokens=p90_projected_input,
            output_tokens=p90_projected_output,
            prefill_usd_per_mtok=input_usd_per_mtok,
            sample_usd_per_mtok=output_usd_per_mtok,
        ),
        "mean_call_context_required_tokens": context_required,
        "context_window": model.context_window,
        "mean_call_context_check_passed": (
            context_required <= model.context_window
            if context_required is not None and model.context_window is not None
            else None
        ),
    }


def _load_token_counter() -> TokenCounter:
    try:
        from litellm import token_counter
    except ImportError as exc:
        raise RuntimeError(
            "offline token estimation requires provider extras; run `uv sync --extra providers`"
        ) from exc
    return cast(TokenCounter, token_counter)


def _count_request(counter: TokenCounter, model: str, request: dict[str, Any]) -> int:
    system_prompt = request.get("system_prompt")
    raw_messages = request.get("messages")
    raw_tools = request.get("tools")
    if not isinstance(system_prompt, str) or not isinstance(raw_messages, list):
        raise ValueError("traced model request has an invalid system prompt or messages")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError("traced message is not an object")
        item: dict[str, Any] = {
            "role": str(raw.get("role", "user")),
            "content": str(raw.get("content", "")),
        }
        if raw.get("name"):
            item["name"] = str(raw["name"])
        if raw.get("tool_call_id"):
            item["tool_call_id"] = str(raw["tool_call_id"])
        raw_calls = raw.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls:
            item["tool_calls"] = [_openai_tool_call(call) for call in raw_calls]
        messages.append(item)
    tools: list[dict[str, Any]] = []
    if isinstance(raw_tools, list):
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                raise ValueError("traced tool is not an object")
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(raw_tool.get("name", "")),
                        "description": str(raw_tool.get("description", "")),
                        "parameters": raw_tool.get("parameters", {}),
                    },
                }
            )
    return counter(model=model, messages=messages, tools=tools or None)


def _count_response(counter: TokenCounter, model: str, response: dict[str, Any]) -> int:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": str(response.get("content", "")),
    }
    raw_calls = response.get("tool_calls")
    if isinstance(raw_calls, list) and raw_calls:
        message["tool_calls"] = [_openai_tool_call(call) for call in raw_calls]
    return counter(model=model, messages=[message], count_response_tokens=True)


def _openai_tool_call(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("traced tool call is not an object")
    arguments = raw.get("arguments", {})
    return {
        "id": str(raw.get("id", "")),
        "type": "function",
        "function": {
            "name": str(raw.get("name", "")),
            "arguments": json.dumps(arguments),
        },
    }


def _token_cost(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> float:
    return (input_tokens * input_usd_per_mtok + output_tokens * output_usd_per_mtok) / 1_000_000

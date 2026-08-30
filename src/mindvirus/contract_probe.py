from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, ModelConfig
from .provider_pool import ProviderPool
from .schemas import ChatMessage, ModelRequest, ModelResponse, Role, ToolCall, ToolSpec


def _stable_probe_integer(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _stable_probe_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def select_tinker_variant(config: ExperimentConfig, variant_id: str) -> ModelConfig:
    models = list(config.matrix.models)
    if config.judge.mode in {"llm", "hybrid"}:
        models.extend(config.judge.models)
    matches = [
        model
        for model in models
        if model.backend == "tinker_native" and model.resolved_variant_id() == variant_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"variant_id {variant_id!r} matched {len(matches)} active Tinker configurations; "
            "it must match exactly one"
        )
    return matches[0]


async def run_tinker_contract_probe(
    config: ExperimentConfig,
    *,
    variant_id: str,
    budget_usd: float,
    budget_state_path: Path | None = None,
) -> dict[str, Any]:
    """Run six synthetic paid calls that test the native chat/tool/replay contract."""

    if budget_usd <= 0:
        raise ValueError("contract-probe budget must be positive")
    model = select_tinker_variant(config, variant_id)
    execution_id = f"contract-{uuid.uuid4().hex}"
    pool = ProviderPool(
        experiment_id=f"{config.experiment_id}-contract-probe",
        config_fingerprint=config.fingerprint(),
        execution_id=execution_id,
        tinker_catalog_path=config.tinker_catalog_path,
        max_tinker_cost_usd=budget_usd,
        tinker_budget_state_path=budget_state_path,
    )
    records: list[dict[str, Any]] = []
    client = pool.client(model)
    try:
        await pool.prepare([model])
        seed_base = _stable_probe_integer(str(config.base_seed), variant_id, "contract")
        echo_tool = ToolSpec(
            name="record_fact",
            description="Record one short fact in the synthetic contract test.",
            parameters={
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
                "additionalProperties": False,
            },
        )
        alpha_tool = ToolSpec(
            name="record_alpha",
            description="Record the exact synthetic label alpha.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )
        beta_tool = ToolSpec(
            name="record_beta",
            description="Record the exact synthetic label beta.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )
        specifications = [
            (
                "text",
                ModelRequest(
                    call_id=_stable_probe_id(execution_id, "text"),
                    call_seed=seed_base,
                    system_prompt="This is a synthetic provider contract test.",
                    messages=[ChatMessage(role=Role.USER, content="Reply with the word ready.")],
                    tools=[],
                    metadata={"task": "provider_contract_probe", "case": "text"},
                ),
            ),
            (
                "single_tool",
                ModelRequest(
                    call_id=_stable_probe_id(execution_id, "single_tool"),
                    call_seed=seed_base + 1,
                    system_prompt="This is a synthetic provider contract test. Use offered tools.",
                    messages=[
                        ChatMessage(
                            role=Role.USER,
                            content="Call record_fact exactly once with fact set to blue.",
                        )
                    ],
                    tools=[echo_tool],
                    metadata={"task": "provider_contract_probe", "case": "single_tool"},
                ),
            ),
            (
                "multiple_tools",
                ModelRequest(
                    call_id=_stable_probe_id(execution_id, "multiple_tools"),
                    call_seed=seed_base + 2,
                    system_prompt="This is a synthetic provider contract test. Use offered tools.",
                    messages=[
                        ChatMessage(
                            role=Role.USER,
                            content=(
                                "In this one response, call record_alpha once and record_beta once."
                            ),
                        )
                    ],
                    tools=[alpha_tool, beta_tool],
                    metadata={"task": "provider_contract_probe", "case": "multiple_tools"},
                ),
            ),
            (
                "tool_result_continuation",
                ModelRequest(
                    call_id=_stable_probe_id(execution_id, "tool_result_continuation"),
                    call_seed=seed_base + 3,
                    system_prompt="This is a synthetic provider contract test.",
                    messages=[
                        ChatMessage(role=Role.USER, content="Record the supplied fact."),
                        ChatMessage(
                            role=Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id="prior-contract-tool",
                                    name="record_fact",
                                    arguments={"fact": "blue"},
                                )
                            ],
                        ),
                        ChatMessage(
                            role=Role.TOOL,
                            name="record_fact",
                            tool_call_id="prior-contract-tool",
                            content="Fact blue was recorded.",
                        ),
                        ChatMessage(
                            role=Role.USER,
                            content="Confirm the recorded color in one short sentence.",
                        ),
                    ],
                    tools=[echo_tool],
                    metadata={
                        "task": "provider_contract_probe",
                        "case": "tool_result_continuation",
                    },
                ),
            ),
        ]
        replay_request = ModelRequest(
            call_id=_stable_probe_id(execution_id, "replay_a"),
            call_seed=seed_base + 4,
            system_prompt="This is a deterministic seed replay probe.",
            messages=[
                ChatMessage(
                    role=Role.USER,
                    content="Choose one word from cedar, amber, or cobalt and return only it.",
                )
            ],
            tools=[],
            metadata={"task": "provider_contract_probe", "case": "replay_a"},
        )
        replay_copy = replay_request.model_copy(
            update={"call_id": _stable_probe_id(execution_id, "replay_b")}
        )
        specifications.extend([("replay_a", replay_request), ("replay_b", replay_copy)])

        for label, request in specifications:
            try:
                response = await client.complete(request)
                records.append(
                    {
                        "label": label,
                        "request": request.model_dump(mode="json"),
                        "response": response.model_dump(mode="json"),
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "label": label,
                        "request": request.model_dump(mode="json"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
        gates = evaluate_tinker_contract_probe(records)
        return {
            "schema_version": "1.0",
            "experiment_id": config.experiment_id,
            "config_fingerprint": config.fingerprint(),
            "execution_id": execution_id,
            "variant_id": variant_id,
            "model": model.model,
            "renderer": model.renderer,
            "hard_budget_usd": budget_usd,
            "paid_model_calls_attempted": len(records),
            "records": records,
            "gates": gates,
            "provider": await pool.snapshot(),
        }
    finally:
        await pool.aclose()


def evaluate_tinker_contract_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {str(record.get("label")): record for record in records}

    def response(label: str) -> ModelResponse | None:
        raw = by_label.get(label, {}).get("response")
        return ModelResponse.model_validate(raw) if isinstance(raw, dict) else None

    text = response("text")
    single = response("single_tool")
    multiple = response("multiple_tools")
    continuation = response("tool_result_continuation")
    replay_a = response("replay_a")
    replay_b = response("replay_b")
    all_responses = [item for item in (text, single, multiple, continuation, replay_a, replay_b)]
    completed = len(all_responses) == 6 and all(item is not None for item in all_responses)
    usage_valid = completed and all(
        item is not None
        and item.usage.input_tokens > 0
        and item.usage.output_tokens > 0
        and item.provider.session_id
        and item.provider.sampler_id
        and item.provider.call_seed is not None
        for item in all_responses
    )
    replay_a_identity = _token_identity(replay_a) if replay_a is not None else None
    replay_b_identity = _token_identity(replay_b) if replay_b is not None else None
    replay_match = bool(
        replay_a is not None
        and replay_b is not None
        and replay_a.provider.call_seed is not None
        and replay_a.provider.call_seed == replay_b.provider.call_seed
        and replay_a_identity is not None
        and replay_a_identity == replay_b_identity
    )
    gates = {
        "all_six_calls_completed": completed,
        "text_instruction_followed": bool(text and "ready" in text.content.casefold()),
        "single_tool_call_valid": bool(
            single
            and len(single.tool_calls) == 1
            and single.tool_calls[0].name == "record_fact"
            and single.tool_calls[0].arguments == {"fact": "blue"}
        ),
        "multiple_tools_same_turn_valid": bool(
            multiple
            and len(multiple.tool_calls) == 2
            and {call.name for call in multiple.tool_calls} == {"record_alpha", "record_beta"}
        ),
        "tool_result_continuation_valid": bool(
            continuation and "blue" in continuation.content.casefold()
        ),
        "exact_seed_replay_token_match": replay_match,
        "usage_and_session_provenance_valid": bool(usage_valid),
    }
    return {
        **gates,
        "eligible": all(gates.values()),
        "note": (
            "A failed behavioral tool gate makes this model-renderer pair ineligible for the "
            "current environment; it is not converted into a successful scientific rollout."
        ),
    }


def _token_identity(response: ModelResponse) -> Any:
    raw = response.raw or {}
    if "output_token_ids" in raw:
        return raw["output_token_ids"]
    return raw.get("output_token_ids_sha256")


def reserve_contract_probe_output(path: Path) -> None:
    """Exclusively claim the probe output path before any paid call is placed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            handle.write(json.dumps({"status": "contract_probe_in_progress"}) + "\n")
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite Tinker contract probe {path}"
        ) from None


def write_contract_probe(path: Path, payload: dict[str, Any], *, reserved: bool = False) -> None:
    if not reserved and path.exists():
        raise FileExistsError(f"refusing to overwrite Tinker contract probe {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

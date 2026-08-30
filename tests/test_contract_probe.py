from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mindvirus import contract_probe
from mindvirus.config import ModelConfig, load_config
from mindvirus.contract_probe import (
    evaluate_tinker_contract_probe,
    reserve_contract_probe_output,
    run_tinker_contract_probe,
    write_contract_probe,
)
from mindvirus.schemas import ModelResponse, ProviderMetadata, ToolCall, UsageRecord

ROOT = Path(__file__).resolve().parents[1]


def _response(
    *,
    content: str = "done",
    tools: list[ToolCall] | None = None,
    token_hash: str = "same-token-hash",
) -> dict[str, Any]:
    return ModelResponse(
        content=content,
        tool_calls=tools or [],
        usage=UsageRecord(input_tokens=10, output_tokens=2, total_tokens=12),
        provider=ProviderMetadata(
            provider="tinker",
            session_id="session",
            sampler_id="sampler",
            call_seed=123,
        ),
        raw={"output_token_ids_sha256": token_hash},
    ).model_dump(mode="json")


def test_contract_probe_requires_all_tool_and_replay_gates() -> None:
    records = [
        {"label": "text", "response": _response(content="ready")},
        {
            "label": "single_tool",
            "response": _response(
                content="",
                tools=[ToolCall(id="one", name="record_fact", arguments={"fact": "blue"})],
            ),
        },
        {
            "label": "multiple_tools",
            "response": _response(
                content="",
                tools=[
                    ToolCall(id="alpha", name="record_alpha"),
                    ToolCall(id="beta", name="record_beta"),
                ],
            ),
        },
        {"label": "tool_result_continuation", "response": _response(content="blue")},
        {"label": "replay_a", "response": _response(content="cedar")},
        {"label": "replay_b", "response": _response(content="cedar")},
    ]
    result = evaluate_tinker_contract_probe(records)
    assert result["eligible"]

    records[-1]["response"] = _response(content="amber", token_hash="different")
    mismatch = evaluate_tinker_contract_probe(records)
    assert not mismatch["exact_seed_replay_token_match"]
    assert not mismatch["eligible"]


def test_contract_probe_rejects_missing_token_identity_and_wrong_arguments() -> None:
    records = [
        {"label": "text", "response": _response(content="ready")},
        {
            "label": "single_tool",
            "response": _response(
                content="",
                tools=[ToolCall(id="one", name="record_fact", arguments={"fact": "red"})],
            ),
        },
        {
            "label": "multiple_tools",
            "response": _response(
                content="",
                tools=[
                    ToolCall(id="alpha", name="record_alpha"),
                    ToolCall(id="beta", name="record_beta"),
                ],
            ),
        },
        {"label": "tool_result_continuation", "response": _response(content="blue")},
        {"label": "replay_a", "response": _response(content="cedar")},
        {"label": "replay_b", "response": _response(content="cedar")},
    ]
    for record in records[-2:]:
        record["response"]["raw"] = {}

    result = evaluate_tinker_contract_probe(records)
    assert not result["single_tool_call_valid"]
    assert not result["exact_seed_replay_token_match"]
    assert not result["eligible"]


def test_reserve_contract_probe_output_fails_fast_when_claimed(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    reserve_contract_probe_output(output)
    assert output.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reserve_contract_probe_output(output)


def test_write_contract_probe_overwrites_only_a_reserved_path(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    reserve_contract_probe_output(output)
    write_contract_probe(output, {"status": "done"}, reserved=True)
    assert json.loads(output.read_text()) == {"status": "done"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_contract_probe(output, {"status": "second"})


def _tinker_probe_config(tmp_path: Path) -> Any:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "contract-probe-fixture"
    config.output_dir = tmp_path / "runs"
    config.tinker_catalog_path = ROOT / "frozen/tinker-models-2026-08-28.json"
    config.max_tinker_cost_usd = 0.01
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
    return config


async def test_probe_passes_budget_journal_to_provider_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        async def complete(self, request: Any) -> ModelResponse:
            return ModelResponse(content="ready")

    class FakePool:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def client(self, config: Any) -> FakeClient:
            return FakeClient()

        async def prepare(self, configs: Any) -> None:
            return None

        async def aclose(self) -> None:
            return None

        async def snapshot(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(contract_probe, "ProviderPool", FakePool)
    journal_path = tmp_path / "probe.json.budget.json"
    payload = await run_tinker_contract_probe(
        _tinker_probe_config(tmp_path),
        variant_id="qwen3_8b_fixture",
        budget_usd=0.01,
        budget_state_path=journal_path,
    )
    assert captured["tinker_budget_state_path"] == journal_path
    assert captured["max_tinker_cost_usd"] == 0.01
    assert payload["paid_model_calls_attempted"] == 6

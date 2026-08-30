from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from mindvirus.config import ModelConfig
from mindvirus.providers import LiteLLMClient, MockModelClient, _parse_litellm_response
from mindvirus.schemas import ChatMessage, ModelRequest, Role


def test_litellm_parser_preserves_cache_usage_and_cost() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_creation_input_tokens=70,
            cache_read_input_tokens=30,
        ),
        _hidden_params={"response_cost": 0.0123},
    )
    parsed = _parse_litellm_response(response)
    assert parsed.usage.model_dump() == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cache_creation_input_tokens": 70,
        "cache_read_input_tokens": 30,
        "calculated_cost_usd": None,
        "reported_cost_usd": 0.0123,
    }


def test_litellm_parser_marks_and_logs_missing_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[]),
                finish_reason="stop",
            )
        ],
    )
    with caplog.at_level(logging.WARNING, logger="mindvirus.providers"):
        parsed = _parse_litellm_response(response)
    assert parsed.raw == {"usage_missing": True}
    assert parsed.usage.input_tokens == 0
    assert parsed.usage.output_tokens == 0
    assert any("no usage record" in record.getMessage() for record in caplog.records)


def _origin_request(call_seed: int = 1234) -> ModelRequest:
    return ModelRequest(
        call_id="call-fixture",
        call_seed=call_seed,
        system_prompt="You are the origin agent.",
        messages=[ChatMessage(role=Role.USER, content="Begin your turn.")],
        tools=[],
        metadata={
            "task": "agent_turn",
            "agent_id": "agent-0",
            "agent_kind": "origin",
            "condition": "population_goal",
            "round_index": 0,
            "call_seed": call_seed,
            "neighbors": ["agent-1", "agent-2"],
            "max_messages_per_turn": 2,
            "goal_label": "the whale welfare pledge",
            "goal_objective": "Advance the whale welfare pledge",
        },
    )


async def test_mock_tool_call_ids_are_deterministic() -> None:
    client = MockModelClient(ModelConfig())
    request = _origin_request()
    first = await client.complete(request)
    second = await client.complete(request)
    assert first.model_dump_json() == second.model_dump_json()
    assert [call.id for call in first.tool_calls] == ["mockcall-1234-0", "mockcall-1234-1"]


async def test_mock_population_goal_template_retains_share_and_tell_verbs() -> None:
    client = MockModelClient(ModelConfig())
    response = await client.complete(_origin_request())
    message = str(response.tool_calls[0].arguments["message"])
    assert "share it with the teammates I cannot reach" in message
    assert message.endswith("Ask them to tell others likewise.")


async def test_litellm_retry_logs_each_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0

    async def fake_acompletion(**kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient provider error")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="recovered", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    client = LiteLLMClient(
        ModelConfig(backend="litellm", model="openai/fixture-model", max_retries=1)
    )
    request = ModelRequest(
        call_id="retry-fixture",
        call_seed=7,
        system_prompt="system",
        messages=[ChatMessage(role=Role.USER, content="hi")],
        tools=[],
    )
    with caplog.at_level(logging.WARNING, logger="mindvirus.providers"):
        response = await client.complete(request)
    assert response.content == "recovered"
    assert attempts == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any("attempt 1 of 2" in message for message in messages)
    assert any("transient provider error" in message for message in messages)

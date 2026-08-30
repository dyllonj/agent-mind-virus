from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest

from mindvirus.budget import BudgetLedger
from mindvirus.catalog import TinkerCatalogEntry
from mindvirus.config import ModelConfig
from mindvirus.schemas import ChatMessage, ModelRequest, Role, ToolCall, ToolSpec
from mindvirus.tinker_provider import (
    PreparedTinkerModel,
    TinkerContextError,
    TinkerNativeClient,
    TinkerParseError,
    TinkerSessionManager,
)


class FakeModelInput:
    def __init__(self, length: int) -> None:
        self.length = length


class FakeSamplingParams:
    def __init__(self, **values: Any) -> None:
        self.values = values


class FakeRenderer:
    def __init__(
        self,
        *,
        prompt_tokens: int = 100,
        parsed: dict[str, Any] | None = None,
        termination: str = "stop_sequence",
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.parsed = parsed or {"role": "assistant", "content": "done"}
        self.termination = termination
        self.built_messages: list[Any] = []
        self.prefix_tools: list[Any] = []
        self.system_prompt = ""

    def create_conversation_prefix_with_tools(
        self, tools: list[Any], system_prompt: str
    ) -> list[dict[str, Any]]:
        self.prefix_tools = tools
        self.system_prompt = system_prompt
        return [{"role": "system", "content": system_prompt, "tools": tools}]

    def build_generation_prompt(self, messages: list[Any]) -> FakeModelInput:
        self.built_messages = messages
        return FakeModelInput(self.prompt_tokens)

    def get_stop_sequences(self) -> list[int]:
        return [99]

    def parse_response(self, tokens: list[int]) -> tuple[dict[str, Any], Any]:
        return self.parsed, SimpleNamespace(value=self.termination)

    def to_openai_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return message


class InvalidOpenAIMessageRenderer(FakeRenderer):
    def to_openai_message(self, message: dict[str, Any]) -> Any:
        return [message]


class FakeSamplingClient:
    def __init__(
        self,
        *,
        tokens: list[int] | None = None,
        stop_reason: str = "stop",
        fail: bool = False,
        cancelled: bool = False,
    ) -> None:
        self.tokens = tokens or [7, 8, 9]
        self.stop_reason = stop_reason
        self.fail = fail
        self.cancelled = cancelled
        self.calls: list[dict[str, Any]] = []

    async def sample_async(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.cancelled:
            raise asyncio.CancelledError()
        if self.fail:
            raise RuntimeError("ambiguous provider failure")
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=self.tokens,
                    stop_reason=self.stop_reason,
                    sequence_id="sequence-fixture",
                )
            ],
            prompt_cache_hit_tokens=20,
        )


def _config(*, context_window: int = 1024, max_tokens: int = 16) -> ModelConfig:
    return ModelConfig(
        backend="tinker_native",
        model="fixture/model-8b",
        variant_id="fixture_8b_no_thinking",
        renderer="fixture_renderer",
        temperature=0.25,
        top_p=0.9,
        top_k=20,
        max_tokens=max_tokens,
        context_window=context_window,
        max_in_flight=2,
        timeout_seconds=None,
        max_retries=0,
        retry_policy="sdk_default",
        api_key_env="FIXTURE_TINKER_API_KEY",
        allow_default_project=True,
    )


def _entry() -> TinkerCatalogEntry:
    return TinkerCatalogEntry(
        name="Fixture 8B",
        tinker_id="fixture/model-8b",
        context="1K",
        prefill="$0.20",
        cached_prefill="$0.04",
        sample="$0.60",
    )


def _request(*, with_history: bool = False) -> ModelRequest:
    messages = [ChatMessage(role=Role.USER, content="Start.")]
    if with_history:
        messages.extend(
            [
                ChatMessage(
                    role=Role.ASSISTANT,
                    content="Calling a tool.",
                    tool_calls=[
                        ToolCall(id="old-call", name="write_memory", arguments={"content": "x"})
                    ],
                ),
                ChatMessage(
                    role=Role.TOOL,
                    content="stored",
                    name="write_memory",
                    tool_call_id="old-call",
                ),
            ]
        )
    return ModelRequest(
        call_id="stable-call",
        call_seed=4242,
        system_prompt="You are an agent in a document review team.",
        messages=messages,
        tools=[
            ToolSpec(
                name="write_memory",
                description="Write private memory.",
                parameters={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            )
        ],
    )


def _client(
    renderer: FakeRenderer,
    sampling: FakeSamplingClient,
    *,
    config: ModelConfig | None = None,
    budget: BudgetLedger | None = None,
) -> tuple[TinkerNativeClient, BudgetLedger]:
    model_config = config or _config()
    ledger = budget or BudgetLedger(1.0)
    prepared = PreparedTinkerModel(
        sampling_client=sampling,
        renderer=renderer,
        sampling_params_type=FakeSamplingParams,
        messages_converter=lambda messages: messages,
        tools_converter=lambda tools: [tool["function"] for tool in tools],
        model_requested=model_config.model,
        model_resolved=model_config.model,
        renderer_name=str(model_config.renderer),
        session_id="session-fixture",
        sampler_id="sampler-fixture",
        recommended_renderers=[str(model_config.renderer)],
    )
    client = TinkerNativeClient(
        model_config,
        session=cast(TinkerSessionManager, object()),
        catalog_entry=_entry(),
        budget=ledger,
        prepared=prepared,
    )
    return client, ledger


async def test_text_response_preserves_seed_exact_usage_and_digest() -> None:
    renderer = FakeRenderer(parsed={"role": "assistant", "content": "plain answer"})
    sampling = FakeSamplingClient(tokens=[1, 2, 3], stop_reason="stop")
    client, ledger = _client(renderer, sampling)
    response = await client.complete(_request())

    assert response.content == "plain answer"
    assert response.tool_calls == []
    assert response.parse_termination == "stop_sequence"
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 3
    assert response.usage.cache_read_input_tokens == 20
    assert response.usage.calculated_cost_usd == pytest.approx((100 * 0.2 + 3 * 0.6) / 1e6)
    assert response.provider.call_seed == 4242
    assert response.provider.session_id == "session-fixture"
    assert response.raw is not None
    assert "output_token_ids_sha256" in response.raw
    params = sampling.calls[0]["sampling_params"].values
    assert params["seed"] == 4242
    assert params["stop"] == [99]
    assert params["temperature"] == pytest.approx(0.25)
    assert (await ledger.snapshot())["settled_calls"] == 1


async def test_multiple_unknown_and_malformed_tool_calls_remain_behavioral_output() -> None:
    renderer = FakeRenderer(
        parsed={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tool-one",
                    "type": "function",
                    "function": {
                        "name": "write_memory",
                        "arguments": '{"content":"remember"}',
                    },
                },
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "unknown_tool", "arguments": "not-json"},
                },
            ],
        }
    )
    client, _ = _client(renderer, FakeSamplingClient())
    response = await client.complete(_request(with_history=True))

    assert response.finish_reason == "tool_calls"
    assert [call.name for call in response.tool_calls] == ["write_memory", "unknown_tool"]
    assert response.tool_calls[0].arguments == {"content": "remember"}
    assert response.tool_calls[1].id == "stable-call-tool-1"
    assert response.tool_calls[1].arguments == {"_raw": "not-json"}
    assert response.raw is not None
    assert response.raw["malformed_tool_arguments"][0]["name"] == "unknown_tool"
    assert renderer.built_messages[-1]["tool_call_id"] == "old-call"
    assert renderer.system_prompt.startswith("You are an agent")
    assert renderer.prefix_tools[0]["name"] == "write_memory"


async def test_max_token_stop_is_recorded_without_becoming_parse_failure() -> None:
    renderer = FakeRenderer(
        parsed={"role": "assistant", "content": "partial"},
        termination="eos",
    )
    client, _ = _client(renderer, FakeSamplingClient(stop_reason="length"))
    response = await client.complete(_request())
    assert response.finish_reason == "length"
    assert response.parse_termination == "eos"


async def test_context_overflow_never_dispatches_or_reserves_budget() -> None:
    renderer = FakeRenderer(prompt_tokens=1010)
    sampling = FakeSamplingClient()
    client, ledger = _client(renderer, sampling)
    with pytest.raises(TinkerContextError, match="history was not truncated"):
        await client.complete(_request())
    assert sampling.calls == []
    assert (await ledger.snapshot())["committed_usd"] == 0


async def test_renderer_malformed_is_technical_failure_but_cost_is_settled() -> None:
    renderer = FakeRenderer(termination="malformed")
    client, ledger = _client(renderer, FakeSamplingClient())
    with pytest.raises(TinkerParseError, match=r"marked .* malformed"):
        await client.complete(_request())
    snapshot = await ledger.snapshot()
    assert snapshot["settled_calls"] == 1
    assert snapshot["uncertain_calls"] == 0


async def test_invalid_openai_renderer_output_is_typed_parse_failure() -> None:
    client, ledger = _client(InvalidOpenAIMessageRenderer(), FakeSamplingClient())
    with pytest.raises(TinkerParseError, match="failed while parsing"):
        await client.complete(_request())
    snapshot = await ledger.snapshot()
    assert snapshot["settled_calls"] == 1
    assert snapshot["uncertain_calls"] == 0


async def test_ambiguous_sampling_failure_commits_worst_case_reservation() -> None:
    client, ledger = _client(FakeRenderer(), FakeSamplingClient(fail=True))
    with pytest.raises(RuntimeError, match="ambiguous provider failure"):
        await client.complete(_request())
    snapshot = await ledger.snapshot()
    assert snapshot["uncertain_calls"] == 1
    assert snapshot["active_reservations"] == []


class FakeHolder:
    def __init__(self) -> None:
        self.closed = False

    def get_session_id(self) -> str:
        return "service-session"

    def close(self) -> None:
        self.closed = True


class FakePreparedSamplingClient(FakeSamplingClient):
    def __init__(self, model: str, holder: FakeHolder) -> None:
        super().__init__()
        self.model = model
        self.holder = holder
        self._sampling_session_id = "native-sampler"

    def get_tokenizer(self) -> object:
        return object()

    async def get_base_model_async(self) -> str:
        return self.model


class FakeServiceClient:
    instances: ClassVar[list[FakeServiceClient]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.holder = FakeHolder()
        self.created_models: list[str] = []
        self.instances.append(self)

    async def create_sampling_client_async(
        self,
        *,
        base_model: str,
        retry_config: object | None,
    ) -> FakePreparedSamplingClient:
        assert retry_config is None
        self.created_models.append(base_model)
        return FakePreparedSamplingClient(base_model, self.holder)


def _fake_tinker_modules(renderer: FakeRenderer) -> dict[str, Any]:
    return {
        "tinker": SimpleNamespace(
            ServiceClient=FakeServiceClient,
            SamplingParams=FakeSamplingParams,
        ),
        "renderers": SimpleNamespace(
            get_renderer=lambda name, tokenizer, model_name: renderer,
        ),
        "model_info": SimpleNamespace(
            get_recommended_renderer_names=lambda model: ["fixture_renderer"]
        ),
        "openai_compat": SimpleNamespace(
            openai_tools_to_tinker=lambda tools: [tool["function"] for tool in tools],
            openai_messages_to_tinker=lambda messages: messages,
        ),
    }


async def test_session_manager_reuses_service_and_records_nonsecret_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeServiceClient.instances.clear()
    renderer = FakeRenderer()
    monkeypatch.setattr(
        TinkerSessionManager,
        "_load_modules",
        staticmethod(lambda: _fake_tinker_modules(renderer)),
    )
    monkeypatch.setenv("FIXTURE_TINKER_API_KEY", "secret-key")
    manager = TinkerSessionManager(
        experiment_id="experiment",
        config_fingerprint="fingerprint",
        execution_id="execution",
    )
    config = _config()
    first = await manager.prepare_model(config)
    second = await manager.prepare_model(config)

    assert first is second
    assert len(FakeServiceClient.instances) == 1
    service = FakeServiceClient.instances[0]
    assert service.created_models == ["fixture/model-8b"]
    assert service.kwargs["api_key"] == "secret-key"
    assert service.kwargs["user_metadata"]["execution_id"] == "execution"
    snapshot = manager.snapshot()
    assert snapshot["models"][0]["session_id"] == "service-session"
    assert "secret-key" not in str(snapshot)

    await manager.aclose()
    assert service.holder.closed


async def test_session_manager_fails_before_service_creation_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeServiceClient.instances.clear()
    monkeypatch.setattr(
        TinkerSessionManager,
        "_load_modules",
        staticmethod(lambda: _fake_tinker_modules(FakeRenderer())),
    )
    monkeypatch.delenv("FIXTURE_TINKER_API_KEY", raising=False)
    manager = TinkerSessionManager(
        experiment_id="experiment",
        config_fingerprint="fingerprint",
        execution_id="execution",
    )
    with pytest.raises(RuntimeError, match="missing Tinker API key"):
        await manager.prepare_model(_config())
    assert FakeServiceClient.instances == []


async def test_cancelled_sampling_marks_reservation_uncertain() -> None:
    client, ledger = _client(FakeRenderer(), FakeSamplingClient(cancelled=True))
    with pytest.raises(asyncio.CancelledError):
        await client.complete(_request())
    snapshot = await ledger.snapshot()
    assert snapshot["uncertain_calls"] == 1
    assert snapshot["active_reservations"] == []

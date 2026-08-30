from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .budget import BudgetLedger, calculate_uncached_cost_usd
from .catalog import TinkerCatalogEntry
from .config import HARNESS_PROTOCOL_VERSION, ModelConfig
from .providers import ModelClient
from .schemas import ModelRequest, ModelResponse, ProviderMetadata, ToolCall, UsageRecord


class TinkerSetupError(RuntimeError):
    """Local dependency, credential, renderer, or catalog setup is invalid."""


class TinkerContextError(RuntimeError):
    """A rendered request cannot fit without changing the frozen protocol."""


class TinkerParseError(RuntimeError):
    """The official renderer could not cleanly parse a sampled response."""


MessageConverter = Callable[[list[dict[str, Any]]], list[Any]]
ToolConverter = Callable[[list[dict[str, Any]]], list[Any]]


@dataclass(slots=True)
class PreparedTinkerModel:
    sampling_client: Any
    renderer: Any
    sampling_params_type: Any
    messages_converter: MessageConverter
    tools_converter: ToolConverter
    model_requested: str
    model_resolved: str
    renderer_name: str
    session_id: str | None
    sampler_id: str | None
    recommended_renderers: list[str]


def _as_string(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _project_value_fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()[:16]


class TinkerSessionManager:
    """Own one native Tinker service session and its variant-specific samplers."""

    def __init__(
        self,
        *,
        experiment_id: str,
        config_fingerprint: str,
        execution_id: str,
    ) -> None:
        self.experiment_id = experiment_id
        self.config_fingerprint = config_fingerprint
        self.execution_id = execution_id
        self._service_client: Any | None = None
        self._prepared: dict[str, PreparedTinkerModel] = {}
        self._credential_selection: tuple[str, str | None, bool] | None = None
        self._project_fingerprint: str | None = None
        self._closed = False

    async def prepare_model(self, config: ModelConfig) -> PreparedTinkerModel:
        if self._closed:
            raise RuntimeError("Tinker session manager is closed")
        key = config.model_dump_json(exclude_none=False)
        if key in self._prepared:
            return self._prepared[key]

        modules = self._load_modules()
        tinker = modules["tinker"]
        renderers = modules["renderers"]
        model_info = modules["model_info"]
        openai_compat = modules["openai_compat"]

        if config.api_key_env is None:
            raise TinkerSetupError("tinker_native requires api_key_env")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise TinkerSetupError(
                f"missing Tinker API key environment variable {config.api_key_env}"
            )
        project_id: str | None = None
        if config.project_id_env is not None:
            project_id = os.environ.get(config.project_id_env)
            if not project_id:
                raise TinkerSetupError(
                    f"missing Tinker project environment variable {config.project_id_env}"
                )
        elif not config.allow_default_project:
            raise TinkerSetupError(
                "Tinker project selection must name project_id_env or explicitly allow the "
                "SDK default project"
            )
        selection = (config.api_key_env, config.project_id_env, config.allow_default_project)
        if self._credential_selection is not None and selection != self._credential_selection:
            raise TinkerSetupError(
                "one experiment-owned Tinker session cannot mix credential or project selections"
            )
        self._credential_selection = selection
        self._project_fingerprint = _project_value_fingerprint(project_id)

        if self._service_client is None:
            self._service_client = tinker.ServiceClient(
                user_metadata={
                    "experiment_id": self.experiment_id,
                    "config_fingerprint": self.config_fingerprint,
                    "protocol_version": HARNESS_PROTOCOL_VERSION,
                    "execution_id": self.execution_id,
                },
                project_id=project_id,
                api_key=api_key,
            )

        if config.renderer is None:
            raise TinkerSetupError("tinker_native requires an explicit renderer")
        try:
            recommended = list(model_info.get_recommended_renderer_names(config.model))
        except Exception as exc:
            raise TinkerSetupError(
                f"Tinker Cookbook cannot resolve renderer recommendations for {config.model!r}"
            ) from exc
        if config.renderer not in recommended:
            raise TinkerSetupError(
                f"renderer {config.renderer!r} is not recommended for {config.model!r}; "
                f"the pinned Cookbook recommends {recommended}"
            )

        sampling_client = await self._service_client.create_sampling_client_async(
            base_model=config.model,
            retry_config=None,
        )
        tokenizer = sampling_client.get_tokenizer()
        renderer = renderers.get_renderer(
            config.renderer,
            tokenizer,
            model_name=config.model,
        )
        tools_converter: ToolConverter = openai_compat.openai_tools_to_tinker
        messages_converter: MessageConverter = openai_compat.openai_messages_to_tinker
        probe_tools = tools_converter(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "mindvirus_contract_probe",
                        "description": "Local renderer capability probe.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        try:
            renderer.create_conversation_prefix_with_tools(probe_tools, "")
        except NotImplementedError as exc:
            raise TinkerSetupError(
                f"renderer {config.renderer!r} does not support the required native tool protocol"
            ) from exc

        model_resolved = str(await sampling_client.get_base_model_async())
        holder = getattr(sampling_client, "holder", None)
        get_session_id = getattr(holder, "get_session_id", None)
        session_id = str(get_session_id()) if callable(get_session_id) else None
        # SDK 0.26 exposes the sampler ID only as private state. It is read for provenance
        # and guarded so a later SDK can remove it without breaking sampling.
        raw_sampler_id = getattr(sampling_client, "_sampling_session_id", None)
        sampler_id = str(raw_sampler_id) if raw_sampler_id is not None else None
        prepared = PreparedTinkerModel(
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params_type=tinker.SamplingParams,
            messages_converter=messages_converter,
            tools_converter=tools_converter,
            model_requested=config.model,
            model_resolved=model_resolved,
            renderer_name=config.renderer,
            session_id=session_id,
            sampler_id=sampler_id,
            recommended_renderers=recommended,
        )
        self._prepared[key] = prepared
        return prepared

    @staticmethod
    def _load_modules() -> dict[str, Any]:
        try:
            return {
                "tinker": importlib.import_module("tinker"),
                "renderers": importlib.import_module("tinker_cookbook.renderers"),
                "model_info": importlib.import_module("tinker_cookbook.model_info"),
                "openai_compat": importlib.import_module(
                    "tinker_cookbook.third_party.openai_compat"
                ),
            }
        except ImportError as exc:
            raise TinkerSetupError(
                "Tinker backend requested but optional dependencies are missing; "
                "run `uv sync --extra tinker`"
            ) from exc

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": "tinker",
            "transport": "native_sampling_client",
            "experiment_id": self.experiment_id,
            "config_fingerprint": self.config_fingerprint,
            "execution_id": self.execution_id,
            "credential_env": self._credential_selection[0]
            if self._credential_selection is not None
            else None,
            "project_id_env": self._credential_selection[1]
            if self._credential_selection is not None
            else None,
            "allow_default_project": self._credential_selection[2]
            if self._credential_selection is not None
            else None,
            "project_value_sha256_prefix": self._project_fingerprint,
            "models": [
                {
                    "model_requested": item.model_requested,
                    "model_resolved": item.model_resolved,
                    "renderer": item.renderer_name,
                    "recommended_renderers": item.recommended_renderers,
                    "session_id": item.session_id,
                    "sampler_id": item.sampler_id,
                }
                for item in self._prepared.values()
            ],
        }

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._service_client is None:
            return
        holder = getattr(self._service_client, "holder", None)
        close = getattr(holder, "close", None)
        if callable(close):
            close()


class TinkerNativeClient(ModelClient):
    """Native render/sample/parse adapter with exact tokens and hard budget control."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        session: TinkerSessionManager,
        catalog_entry: TinkerCatalogEntry,
        budget: BudgetLedger,
        prepared: PreparedTinkerModel | None = None,
    ) -> None:
        self.config = config
        self.session = session
        self.catalog_entry = catalog_entry
        self.budget = budget
        self._prepared = prepared

    async def prepare(self) -> None:
        if self.catalog_entry.tinker_id != self.config.model:
            raise TinkerSetupError(
                f"catalog model {self.catalog_entry.tinker_id!r} does not match "
                f"configured model {self.config.model!r}"
            )
        if self.config.context_window != self.catalog_entry.context_tokens:
            raise TinkerSetupError(
                f"configured context {self.config.context_window} does not match frozen catalog "
                f"context {self.catalog_entry.context_tokens}"
            )
        if self._prepared is None:
            self._prepared = await self.session.prepare_model(self.config)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        await self.prepare()
        prepared = self._require_prepared()
        if request.call_seed is None:
            raise TinkerSetupError("native Tinker calls require an explicit call_seed")
        if not request.call_id:
            raise TinkerSetupError("native Tinker calls require a stable call_id")

        messages = prepared.messages_converter(self._openai_messages(request))
        tools = prepared.tools_converter(self._openai_tools(request))
        try:
            prefix = prepared.renderer.create_conversation_prefix_with_tools(
                tools,
                request.system_prompt,
            )
            model_input = prepared.renderer.build_generation_prompt(prefix + messages)
        except Exception as exc:
            raise TinkerParseError(
                f"renderer {prepared.renderer_name!r} could not build the generation prompt"
            ) from exc
        prompt_tokens = int(model_input.length)
        context_window = self.config.context_window
        if context_window is None:
            raise TinkerSetupError("native Tinker model is missing context_window")
        required = prompt_tokens + self.config.max_tokens + self.config.context_safety_tokens
        if required > context_window:
            raise TinkerContextError(
                f"call {request.call_id} requires {prompt_tokens} prompt + "
                f"{self.config.max_tokens} output + {self.config.context_safety_tokens} safety "
                f"tokens, exceeding context window {context_window}; history was not truncated"
            )

        parameters = prepared.sampling_params_type(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stop=prepared.renderer.get_stop_sequences(),
            seed=request.call_seed,
        )
        await self.budget.reserve(
            call_id=request.call_id,
            variant_id=self.config.resolved_variant_id(),
            input_tokens=prompt_tokens,
            maximum_output_tokens=self.config.max_tokens,
            prefill_usd_per_mtok=self.catalog_entry.prefill_usd_per_mtok,
            sample_usd_per_mtok=self.catalog_entry.sample_usd_per_mtok,
        )
        sample_started = time.perf_counter()
        try:
            sampled = await prepared.sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=parameters,
            )
            sequences = list(sampled.sequences)
            if len(sequences) != 1:
                raise TinkerParseError(
                    f"native Tinker call {request.call_id} returned {len(sequences)} sequences; "
                    "exactly one was requested"
                )
            sequence = sequences[0]
            output_tokens = [int(token) for token in sequence.tokens]
        except asyncio.CancelledError:
            await self.budget.mark_uncertain(request.call_id)
            raise
        except Exception:
            await self.budget.mark_uncertain(request.call_id)
            raise
        sample_ms = (time.perf_counter() - sample_started) * 1000
        actual_cost = calculate_uncached_cost_usd(
            input_tokens=prompt_tokens,
            output_tokens=len(output_tokens),
            prefill_usd_per_mtok=self.catalog_entry.prefill_usd_per_mtok,
            sample_usd_per_mtok=self.catalog_entry.sample_usd_per_mtok,
        )
        try:
            await self.budget.settle(request.call_id, actual_cost)
        except Exception:
            await self.budget.mark_uncertain(request.call_id)
            raise

        try:
            parsed_message, termination = prepared.renderer.parse_response(output_tokens)
            termination_value = _as_string(termination)
            openai_message = prepared.renderer.to_openai_message(parsed_message)
            if not isinstance(openai_message, dict):
                raise TypeError(
                    "renderer.to_openai_message returned "
                    f"{type(openai_message).__name__}, not a mapping"
                )
            raw_tool_calls = openai_message.get("tool_calls") or []
            if not isinstance(raw_tool_calls, list):
                raise TypeError(
                    "renderer.to_openai_message returned non-list tool_calls "
                    f"({type(raw_tool_calls).__name__})"
                )
        except Exception as exc:
            raise TinkerParseError(
                f"renderer {prepared.renderer_name!r} failed while parsing call {request.call_id}"
            ) from exc
        if termination_value == "malformed":
            raise TinkerParseError(
                f"renderer {prepared.renderer_name!r} marked call {request.call_id} malformed"
            )

        tool_calls, malformed_arguments = self._normalize_tool_calls(
            raw_tool_calls,
            request.call_id,
        )
        stop_reason = _as_string(getattr(sequence, "stop_reason", ""))
        finish_reason = "tool_calls" if tool_calls else stop_reason
        cache_hit_tokens = int(getattr(sampled, "prompt_cache_hit_tokens", 0) or 0)
        token_record: dict[str, Any]
        if self.config.record_output_token_ids:
            token_record = {"output_token_ids": output_tokens}
        else:
            encoded_tokens = json.dumps(output_tokens, separators=(",", ":")).encode()
            token_record = {
                "output_token_count": len(output_tokens),
                "output_token_ids_sha256": hashlib.sha256(encoded_tokens).hexdigest(),
            }
        return ModelResponse(
            content=str(openai_message.get("content") or ""),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            parse_termination=termination_value,
            usage=UsageRecord(
                input_tokens=prompt_tokens,
                output_tokens=len(output_tokens),
                total_tokens=prompt_tokens + len(output_tokens),
                cache_read_input_tokens=cache_hit_tokens,
                calculated_cost_usd=actual_cost,
            ),
            provider=ProviderMetadata(
                provider="tinker",
                transport="native_sampling_client",
                model_requested=prepared.model_requested,
                model_resolved=prepared.model_resolved,
                variant_id=self.config.resolved_variant_id(),
                renderer=prepared.renderer_name,
                session_id=prepared.session_id,
                sampler_id=prepared.sampler_id,
                call_id=request.call_id,
                call_seed=request.call_seed,
                sample_ms=sample_ms,
            ),
            raw={
                "sequence_id": getattr(sequence, "sequence_id", None),
                "stop_reason": stop_reason,
                "parse_termination": termination_value,
                "prompt_tokens": prompt_tokens,
                "maximum_output_tokens": self.config.max_tokens,
                "prompt_cache_hit_tokens": cache_hit_tokens,
                "context_window": context_window,
                "context_safety_tokens": self.config.context_safety_tokens,
                "context_required_tokens": required,
                "malformed_tool_arguments": malformed_arguments,
                **token_record,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        prepared = self._prepared
        return {
            "backend": self.config.backend,
            "variant_id": self.config.resolved_variant_id(),
            "model_requested": self.config.model,
            "model_resolved": prepared.model_resolved if prepared is not None else None,
            "renderer": self.config.renderer,
            "context_window": self.config.context_window,
            "context_safety_tokens": self.config.context_safety_tokens,
            "max_in_flight": self.config.max_in_flight,
            "sampling": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_tokens": self.config.max_tokens,
            },
            "pricing": self.catalog_entry.pricing_record(),
        }

    def _require_prepared(self) -> PreparedTinkerModel:
        if self._prepared is None:
            raise RuntimeError("Tinker client was not prepared")
        return self._prepared

    @staticmethod
    def _openai_messages(request: ModelRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            item: dict[str, Any] = {"role": message.role.value, "content": message.content}
            if message.name is not None:
                item["name"] = message.name
            if message.tool_call_id is not None:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, sort_keys=True),
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(item)
        return messages

    @staticmethod
    def _openai_tools(request: ModelRequest) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]

    @staticmethod
    def _normalize_tool_calls(
        raw_calls: list[Any],
        call_id: str,
    ) -> tuple[list[ToolCall], list[dict[str, Any]]]:
        calls: list[ToolCall] = []
        malformed: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_calls):
            raw_function = raw.get("function", {}) if isinstance(raw, dict) else {}
            name = str(raw_function.get("name", ""))
            raw_arguments = raw_function.get("arguments", "{}")
            parsed_arguments: Any
            if isinstance(raw_arguments, dict):
                parsed_arguments = raw_arguments
            else:
                try:
                    parsed_arguments = json.loads(str(raw_arguments or "{}"))
                except json.JSONDecodeError:
                    parsed_arguments = {"_raw": str(raw_arguments)}
                    malformed.append(
                        {"index": index, "name": name, "raw_arguments": str(raw_arguments)}
                    )
            if not isinstance(parsed_arguments, dict):
                malformed.append({"index": index, "name": name, "raw_arguments": raw_arguments})
                parsed_arguments = {"_raw": raw_arguments}
            raw_id = raw.get("id") if isinstance(raw, dict) else None
            normalized_id = str(raw_id) if raw_id else f"{call_id}-tool-{index}"
            calls.append(ToolCall(id=normalized_id, name=name, arguments=parsed_arguments))
        return calls, malformed


def offline_tinker_sdk_preflight(configs: list[ModelConfig]) -> dict[str, Any]:
    """Verify the pinned SDK surface and model-renderer registry without credentials or calls."""

    tinker_configs = [config for config in configs if config.backend == "tinker_native"]
    if not tinker_configs:
        raise ValueError("offline Tinker preflight requires at least one tinker_native model")
    modules = TinkerSessionManager._load_modules()
    tinker = modules["tinker"]
    renderers = modules["renderers"]
    model_info = modules["model_info"]

    sampling_fields = set(inspect.signature(tinker.SamplingParams).parameters)
    required_sampling_fields = {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "stop",
        "seed",
    }
    missing_sampling = sorted(required_sampling_fields - sampling_fields)
    if missing_sampling:
        raise TinkerSetupError(
            f"installed Tinker SamplingParams is missing required fields {missing_sampling}"
        )
    service_create = getattr(tinker.ServiceClient, "create_sampling_client_async", None)
    if not callable(service_create):
        raise TinkerSetupError("installed Tinker SDK lacks create_sampling_client_async")
    sampling_complete = getattr(tinker.SamplingClient, "sample_async", None)
    if not callable(sampling_complete):
        raise TinkerSetupError("installed Tinker SDK lacks SamplingClient.sample_async")
    for method in (
        "create_conversation_prefix_with_tools",
        "build_generation_prompt",
        "get_stop_sequences",
        "parse_response",
        "to_openai_message",
    ):
        if not callable(getattr(renderers.Renderer, method, None)):
            raise TinkerSetupError(f"installed Tinker renderer contract lacks {method}")

    models: list[dict[str, Any]] = []
    for config in tinker_configs:
        if config.renderer is None:
            raise TinkerSetupError("tinker_native model lacks an explicit renderer")
        recommended = list(model_info.get_recommended_renderer_names(config.model))
        if config.renderer not in recommended:
            raise TinkerSetupError(
                f"renderer {config.renderer!r} is not registered for {config.model!r}; "
                f"recommended renderers are {recommended}"
            )
        models.append(
            {
                "variant_id": config.resolved_variant_id(),
                "model": config.model,
                "renderer": config.renderer,
                "recommended_renderers": recommended,
                "context_window": config.context_window,
            }
        )
    return {
        "status": "passed",
        "network_calls": 0,
        "paid_model_calls": 0,
        "packages": {
            "tinker": importlib.metadata.version("tinker"),
            "tinker-cookbook": importlib.metadata.version("tinker-cookbook"),
        },
        "sampling_params_fields": sorted(sampling_fields),
        "models": models,
    }

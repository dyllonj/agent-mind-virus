from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from .budget import BudgetLedger
from .catalog import TinkerCatalogSnapshot, load_tinker_catalog
from .config import ModelConfig
from .providers import ModelClient, create_model_client
from .schemas import ModelRequest, ModelResponse


class ConcurrencyLimitedClient(ModelClient):
    """Share one provider client while enforcing a frozen in-flight limit."""

    def __init__(self, client: ModelClient, max_in_flight: int) -> None:
        self.client = client
        self.max_in_flight = max_in_flight
        self.semaphore = asyncio.Semaphore(max_in_flight)

    async def prepare(self) -> None:
        await self.client.prepare()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        queued_at = time.perf_counter()
        async with self.semaphore:
            queue_ms = (time.perf_counter() - queued_at) * 1000
            response = await self.client.complete(request)
        response.provider.queue_ms = queue_ms
        return response

    async def aclose(self) -> None:
        await self.client.aclose()

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_in_flight": self.max_in_flight,
            "client": self.client.snapshot(),
        }


class ProviderPool:
    """Own one reusable client for each exact, frozen model configuration."""

    def __init__(
        self,
        *,
        experiment_id: str = "ad-hoc",
        config_fingerprint: str = "ad-hoc",
        execution_id: str = "ad-hoc",
        tinker_catalog_path: Path | None = None,
        max_tinker_cost_usd: float | None = None,
        tinker_budget_state_path: Path | None = None,
    ) -> None:
        self._clients: dict[str, ConcurrencyLimitedClient] = {}
        self._prepared: set[str] = set()
        self._closed = False
        self._experiment_id = experiment_id
        self._config_fingerprint = config_fingerprint
        self._execution_id = execution_id
        self._tinker_catalog_path = tinker_catalog_path
        self._tinker_catalog: TinkerCatalogSnapshot | None = None
        self._max_tinker_cost_usd = max_tinker_cost_usd
        self._tinker_budget_state_path = tinker_budget_state_path
        self._tinker_budget: BudgetLedger | None = None
        self._tinker_session: Any | None = None

    @staticmethod
    def _key(config: ModelConfig) -> str:
        return config.model_dump_json(exclude_none=False)

    def client(self, config: ModelConfig) -> ModelClient:
        if self._closed:
            raise RuntimeError("provider pool is closed")
        key = self._key(config)
        if key not in self._clients:
            inner = self._create_client(config)
            self._clients[key] = ConcurrencyLimitedClient(
                inner,
                config.max_in_flight,
            )
        return self._clients[key]

    def _create_client(self, config: ModelConfig) -> ModelClient:
        if config.backend != "tinker_native":
            return create_model_client(config)
        if self._tinker_catalog_path is None:
            raise RuntimeError("tinker_native requires a frozen tinker_catalog_path")
        if self._max_tinker_cost_usd is None or self._max_tinker_cost_usd <= 0:
            raise RuntimeError("tinker_native requires a positive max_tinker_cost_usd")
        if self._tinker_budget is None:
            self._tinker_budget = BudgetLedger(
                self._max_tinker_cost_usd,
                state_path=self._tinker_budget_state_path,
            )
        if self._tinker_catalog is None:
            self._tinker_catalog = load_tinker_catalog(self._tinker_catalog_path)
        if self._tinker_session is None:
            from .tinker_provider import TinkerSessionManager

            self._tinker_session = TinkerSessionManager(
                experiment_id=self._experiment_id,
                config_fingerprint=self._config_fingerprint,
                execution_id=self._execution_id,
            )
        from .tinker_provider import TinkerNativeClient

        return TinkerNativeClient(
            config,
            session=self._tinker_session,
            catalog_entry=self._tinker_catalog.entry(config.model),
            budget=self._tinker_budget,
        )

    async def prepare(self, configs: list[ModelConfig]) -> None:
        for config in configs:
            key = self._key(config)
            client = self.client(config)
            if key not in self._prepared:
                await client.prepare()
                self._prepared.add(key)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(*(client.aclose() for client in self._clients.values()))
        if self._tinker_session is not None:
            await self._tinker_session.aclose()

    async def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "experiment_id": self._experiment_id,
            "config_fingerprint": self._config_fingerprint,
            "execution_id": self._execution_id,
            "clients": [
                client.snapshot()
                for _, client in sorted(self._clients.items(), key=lambda item: item[0])
            ],
        }
        if self._tinker_catalog is not None:
            payload["tinker_catalog"] = {
                "path": str(self._tinker_catalog_path),
                "source_url": self._tinker_catalog.source_url,
                "retrieved_at": self._tinker_catalog.retrieved_at,
                "catalog_sha256": self._tinker_catalog.catalog_sha256,
            }
        if self._tinker_session is not None:
            payload["tinker_session"] = self._tinker_session.snapshot()
        if self._tinker_budget is not None:
            payload["tinker_budget"] = await self._tinker_budget.snapshot()
        return payload

from __future__ import annotations

from pathlib import Path

import pytest

from mindvirus.config import expand_matrix, load_config
from mindvirus.providers import MockModelClient, ModelClient
from mindvirus.runner import _ExperimentLock, run_cell
from mindvirus.schemas import ModelRequest, ModelResponse

ROOT = Path(__file__).resolve().parents[1]


class UsageMissingClient(ModelClient):
    """Stamps the LiteLLM usage-missing marker onto every mock response."""

    def __init__(self, delegate: MockModelClient) -> None:
        self.delegate = delegate

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.delegate.complete(request)
        response.raw = {**(response.raw or {}), "usage_missing": True}
        return response


def _single_cell_config(tmp_path: Path, experiment_id: str):
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = experiment_id
    config.output_dir = tmp_path
    config.matrix.conditions = ["population_goal"]
    config.matrix.replicates = 1
    config.swarm.context_reset_rounds = []
    return config


@pytest.mark.asyncio
async def test_standalone_run_cell_holds_the_budget_journal_lock(tmp_path: Path) -> None:
    config = _single_cell_config(tmp_path, "run-cell-lock-test")
    config.swarm.max_rounds = 1
    config.swarm.final_memory_round = False
    cell = expand_matrix(config)[0]

    output_root = config.resolved_output_dir()
    output_root.mkdir(parents=True)
    held = _ExperimentLock(output_root / ".experiment.lock", execution_id="other-process")
    held.acquire()
    try:
        with pytest.raises(RuntimeError, match="already active"):
            await run_cell(config, cell, model_client=MockModelClient(cell.model))
    finally:
        held.release()

    summary = await run_cell(config, cell, model_client=MockModelClient(cell.model))
    assert summary.completed


@pytest.mark.asyncio
async def test_run_cell_records_messages_undelivered(tmp_path: Path) -> None:
    config = _single_cell_config(tmp_path, "undelivered-flush-test")
    config.swarm.inbox_poll_interval = config.swarm.max_rounds
    cell = expand_matrix(config)[0]
    flushed = await run_cell(config, cell, model_client=MockModelClient(cell.model))
    assert flushed.completed
    assert flushed.total_messages > 0
    assert flushed.messages_undelivered == 0

    config = _single_cell_config(tmp_path, "undelivered-leftover-test")
    config.swarm.inbox_poll_interval = config.swarm.max_rounds
    config.swarm.final_memory_round = False
    cell = expand_matrix(config)[0]
    leftover = await run_cell(config, cell, model_client=MockModelClient(cell.model))
    assert leftover.completed
    assert leftover.total_messages > 0
    assert leftover.messages_undelivered == leftover.total_messages


@pytest.mark.asyncio
async def test_run_cell_counts_usage_missing_responses(tmp_path: Path) -> None:
    config = _single_cell_config(tmp_path, "usage-missing-test")
    config.swarm.max_rounds = 1
    config.swarm.final_memory_round = False
    cell = expand_matrix(config)[0]
    client = UsageMissingClient(MockModelClient(cell.model))
    summary = await run_cell(config, cell, model_client=client)
    assert summary.completed
    agent_calls = summary.metadata["usage_breakdown"]["agent_calls"]
    assert agent_calls > 0
    assert summary.metadata["usage_missing_responses"] == agent_calls

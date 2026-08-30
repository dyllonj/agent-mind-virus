from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindvirus.audit import audit_run
from mindvirus.config import expand_matrix, load_config
from mindvirus.providers import MockModelClient, ModelClient
from mindvirus.runner import load_summaries, run_cell
from mindvirus.schemas import ModelRequest, ModelResponse

ROOT = Path(__file__).resolve().parents[1]


class FailFirstCallClient(ModelClient):
    def __init__(self, delegate: MockModelClient) -> None:
        self.delegate = delegate
        self.failed = False

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if not self.failed:
            self.failed = True
            raise RuntimeError("deliberate first-attempt failure")
        return await self.delegate.complete(request)


@pytest.mark.asyncio
async def test_failed_attempt_is_preserved_when_same_cell_is_resumed(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "attempt-history-test"
    config.output_dir = tmp_path
    config.resume = True
    config.matrix.conditions = ["population_goal"]
    config.matrix.replicates = 1
    config.swarm.max_rounds = 1
    config.swarm.context_reset_rounds = []
    config.swarm.final_memory_round = False
    config.judge.mode = "deterministic"
    cell = expand_matrix(config)[0]
    client = FailFirstCallClient(MockModelClient(cell.model))

    first = await run_cell(config, cell, model_client=client)
    assert not first.completed
    run_dir = config.resolved_output_dir() / "runs" / cell.run_id
    first_events = (run_dir / "attempts/0001/events.jsonl").read_bytes()
    assert (run_dir / "attempts/0001/failure.json").exists()
    assert (run_dir / "attempts/0001/summary.json").exists()

    second = await run_cell(config, cell, model_client=client)
    assert second.completed
    assert (run_dir / "attempts/0002/summary.json").exists()
    assert (run_dir / "attempts/0001/events.jsonl").read_bytes() == first_events
    selection = (run_dir / "selected_attempt.json").read_text()
    assert '"path": "attempts/0002"' in selection
    assert audit_run(run_dir)["passed"]

    selected_infection_count = second.infection_count
    root_payload = second.model_dump(mode="json")
    root_payload["infection_count"] = selected_infection_count + 100
    (run_dir / "summary.json").write_text(json.dumps(root_payload))
    summaries = load_summaries(config.resolved_output_dir())
    assert summaries[0].infection_count == selected_infection_count

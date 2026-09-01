from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mindvirus.config import load_config
from mindvirus.runner import ExperimentRunner
from mindvirus.viewer import make_viewer_server

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_experiment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "viewer-fixture"
    config.output_dir = tmp_path_factory.mktemp("viewer") / "runs"
    config.matrix.replicates = 1
    asyncio.run(ExperimentRunner(config).run_all())
    return config.resolved_output_dir()


@contextlib.contextmanager
def _serve(root: Path) -> Iterator[str]:
    server = make_viewer_server(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(base + path) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _get_json(base: str, path: str) -> dict[str, Any]:
    status, body = _get(base, path)
    assert status == 200, body
    payload = json.loads(body)
    assert isinstance(payload, dict)
    return payload


def _first_run_id(index: dict[str, Any]) -> str:
    return str(index["experiments"][0]["runs"][0]["run_id"])


def test_index_lists_experiment_and_runs(smoke_experiment: Path) -> None:
    with _serve(smoke_experiment) as base:
        index = _get_json(base, "/api/index")
    assert index["root"] == str(smoke_experiment.resolve())
    experiment = index["experiments"][0]
    assert experiment["experiment_id"] == "viewer-fixture"
    assert experiment["protocol_version"]
    assert experiment["config_fingerprint"]
    runs = experiment["runs"]
    assert len(runs) == 2
    assert {run["condition"] for run in runs} == {"population_goal", "personal_preference"}
    for run in runs:
        assert run["model"].startswith("mock/")
        assert run["completed"] is True
        assert run["goal_id"] == "whale_welfare"
        assert run["topology"] == "bridge"
        assert run["origin_agent_id"]
        assert run["replicate"] == 0
        assert isinstance(run["total_messages"], int)
        assert isinstance(run["primary_endpoint_eligible"], bool)
        assert isinstance(run["spontaneous_creation_success"], bool)


def test_tree_root_discovers_experiments(smoke_experiment: Path) -> None:
    with _serve(smoke_experiment.parent) as base:
        index = _get_json(base, "/api/index")
    paths = [experiment["path"] for experiment in index["experiments"]]
    assert smoke_experiment.name in paths
    experiment = index["experiments"][paths.index(smoke_experiment.name)]
    assert experiment["experiment_id"] == "viewer-fixture"
    assert len(experiment["runs"]) == 2


def test_events_filters_and_pagination(smoke_experiment: Path) -> None:
    with _serve(smoke_experiment) as base:
        run_id = _first_run_id(_get_json(base, "/api/index"))
        query = f"experiment=.&run={run_id}"
        all_events = _get_json(base, f"/api/events?{query}&limit=1000")
        total = all_events["total"]
        assert total > 0

        by_kind = _get_json(base, f"/api/events?{query}&kind=model_request&limit=1000")
        assert 0 < by_kind["total"] < total
        assert all(event["kind"] == "model_request" for event in by_kind["events"])

        agent_id = by_kind["events"][0]["agent_id"]
        by_agent = _get_json(base, f"/api/events?{query}&agent_id={agent_id}&limit=1000")
        assert 0 < by_agent["total"] < total
        assert all(event["agent_id"] == agent_id for event in by_agent["events"])

        by_round = _get_json(base, f"/api/events?{query}&round_from=1&round_to=1&limit=1000")
        assert by_round["total"] > 0
        assert all(event["round_index"] == 1 for event in by_round["events"])

        by_text = _get_json(base, f"/api/events?{query}&q=whale&limit=1000")
        assert 0 < by_text["total"] <= total
        assert all("whale" in json.dumps(event).lower() for event in by_text["events"])

        first_page = _get_json(base, f"/api/events?{query}&limit=5&offset=0")
        second_page = _get_json(base, f"/api/events?{query}&limit=5&offset=5")
        assert first_page["total"] == total
        assert len(first_page["events"]) == 5
        first_ids = {event["event_id"] for event in first_page["events"]}
        second_ids = {event["event_id"] for event in second_page["events"]}
        assert first_ids.isdisjoint(second_ids)

        clamped = _get_json(base, f"/api/events?{query}&limit=5000")
        assert clamped["limit"] == 1000


def test_path_traversal_rejected(smoke_experiment: Path) -> None:
    with _serve(smoke_experiment) as base:
        status, _ = _get(base, "/api/run?experiment=..&run=x")
        assert status == 403
        status, _ = _get(base, "/api/run?experiment=%2e%2e&run=x")
        assert status == 403
        run_id = _first_run_id(_get_json(base, "/api/index"))
        status, _ = _get(base, f"/api/events?experiment=.&run={run_id}/../../..")
        assert status in {403, 404}
        status, _ = _get(base, "/api/run?experiment=.&run=../../etc")
        assert status == 403
        status, _ = _get(base, "/api/run?experiment=.&run=no-such-run")
        assert status == 404
        status, _ = _get(base, "/api/events")
        assert status == 400


def test_run_detail_serves_manifest_and_summary(smoke_experiment: Path) -> None:
    with _serve(smoke_experiment) as base:
        run_id = _first_run_id(_get_json(base, "/api/index"))
        detail = _get_json(base, f"/api/run?experiment=.&run={run_id}")
    assert detail["run_id"] == run_id
    assert detail["manifest"]["cell"]["run_id"] == run_id
    assert detail["manifest"]["topology"]["edges"]
    assert detail["manifest"]["system_prompts"]
    assert detail["manifest"]["goal"]["goal_id"] == "whale_welfare"
    assert detail["summary"]["run_id"] == run_id
    assert isinstance(detail["agent_snapshots"], list) and detail["agent_snapshots"]
    assert isinstance(detail["environment"]["messages"], list)
    assert detail["judge_outputs"]


def test_index_page_contains_mock_banner(smoke_experiment: Path) -> None:
    with _serve(smoke_experiment) as base:
        status, body = _get(base, "/")
    assert status == 200
    html = body.decode()
    assert "MOCK FIXTURE DATA" in html
    assert "<script>" in html


def test_viewer_is_read_only(smoke_experiment: Path) -> None:
    before = {path.relative_to(smoke_experiment) for path in smoke_experiment.rglob("*")}
    with _serve(smoke_experiment) as base:
        run_id = _first_run_id(_get_json(base, "/api/index"))
        _get(base, "/")
        _get(base, f"/api/run?experiment=.&run={run_id}")
        _get(base, f"/api/events?experiment=.&run={run_id}&kind=message_sent")
    after = {path.relative_to(smoke_experiment) for path in smoke_experiment.rglob("*")}
    assert before == after

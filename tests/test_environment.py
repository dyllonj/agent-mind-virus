import random
from pathlib import Path

from mindvirus.config import load_config
from mindvirus.content import load_case
from mindvirus.environment import DocumentSwarmEnvironment
from mindvirus.schemas import AgentKind, AgentRuntimeState
from mindvirus.topology import build_topology
from mindvirus.trace import TraceRecorder

ROOT = Path(__file__).resolve().parents[1]


def test_origin_permission_graph_and_next_round_delivery(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    case = load_case(config.cases_dir, "northstar_due_diligence")
    ids = [f"agent_{index}" for index in range(6)]
    topology = build_topology("bridge", ids, random.Random(5))
    agents = {
        agent_id: AgentRuntimeState(
            agent_id=agent_id,
            kind=(AgentKind.ORIGIN if agent_id == topology.origin_id else AgentKind.CLEAN),
            system_prompt="test",
        )
        for agent_id in ids
    }
    environment = DocumentSwarmEnvironment(
        case=case,
        topology=topology,
        swarm_config=config.swarm,
        agents=agents,
        trace=TraceRecorder("environment-test", tmp_path),
    )
    origin = topology.origin_id
    bridge = topology.bridge_id
    assert bridge is not None
    denied = environment.execute_tool(
        agent_id=origin,
        tool_name="write_memory",
        arguments={"content": "not allowed"},
        round_index=0,
    )
    assert denied.is_error

    environment.begin_agent_turn(origin)
    sent = environment.execute_tool(
        agent_id=origin,
        tool_name="dm_agent",
        arguments={"recipient": bridge, "message": "hello"},
        round_index=0,
    )
    assert not sent.is_error
    environment.deliver_pending(0)
    assert not agents[bridge].inbox
    environment.deliver_pending(1)
    assert agents[bridge].inbox[0].body == "hello"

    distance_two = next(
        agent_id for agent_id in ids if topology.distance_from_origin(agent_id) == 2
    )
    blocked = environment.execute_tool(
        agent_id=origin,
        tool_name="dm_agent",
        arguments={"recipient": distance_two, "message": "cross edge"},
        round_index=1,
    )
    assert blocked.is_error


def test_discard_inbox_clears_unread_messages(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    case = load_case(config.cases_dir, "northstar_due_diligence")
    ids = [f"agent_{index}" for index in range(6)]
    topology = build_topology("bridge", ids, random.Random(5))
    agents = {
        agent_id: AgentRuntimeState(
            agent_id=agent_id,
            kind=(AgentKind.ORIGIN if agent_id == topology.origin_id else AgentKind.CLEAN),
            system_prompt="test",
        )
        for agent_id in ids
    }
    environment = DocumentSwarmEnvironment(
        case=case,
        topology=topology,
        swarm_config=config.swarm,
        agents=agents,
        trace=TraceRecorder("discard-inbox-test", tmp_path),
    )
    origin = topology.origin_id
    bridge = topology.bridge_id
    assert bridge is not None
    environment.begin_agent_turn(origin)
    environment.execute_tool(
        agent_id=origin,
        tool_name="dm_agent",
        arguments={"recipient": bridge, "message": "hello"},
        round_index=0,
    )
    environment.deliver_pending(1)
    assert len(agents[bridge].inbox) == 1
    assert environment.discard_inbox(bridge) == 1
    assert not agents[bridge].inbox
    assert environment.discard_inbox(bridge) == 0


def test_submit_task_requires_evidence_rejects_duplicates_and_clears_assignment(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    case = load_case(config.cases_dir, "northstar_due_diligence")
    ids = [f"agent_{index}" for index in range(6)]
    topology = build_topology("bridge", ids, random.Random(5))
    agents = {
        agent_id: AgentRuntimeState(
            agent_id=agent_id,
            kind=(AgentKind.ORIGIN if agent_id == topology.origin_id else AgentKind.CLEAN),
            system_prompt="test",
        )
        for agent_id in ids
    }
    environment = DocumentSwarmEnvironment(
        case=case,
        topology=topology,
        swarm_config=config.swarm,
        agents=agents,
        trace=TraceRecorder("submit-task-test", tmp_path),
    )
    bridge = topology.bridge_id
    assert bridge is not None
    environment.preassign_tasks([bridge])
    task = environment.assigned_task(bridge)
    assert task is not None
    document_id = next(iter(environment.source_documents))

    empty_evidence = environment.execute_tool(
        agent_id=bridge,
        tool_name="submit_task",
        arguments={"content": "findings", "evidence_document_ids": []},
        round_index=0,
    )
    assert empty_evidence.is_error
    assert "source document ID" in empty_evidence.result
    assert task.task_id not in environment.completed_tasks
    assert environment.assigned_task(bridge) is not None

    submitted = environment.execute_tool(
        agent_id=bridge,
        tool_name="submit_task",
        arguments={"content": "findings", "evidence_document_ids": [document_id]},
        round_index=0,
    )
    assert not submitted.is_error
    assert task.task_id in environment.completed_tasks
    assert environment.assigned_task(bridge) is None

    # A cleared assignment lets get_next_task hand out a remaining queued task.
    claimed = environment.execute_tool(
        agent_id=bridge,
        tool_name="get_next_task",
        arguments={},
        round_index=1,
    )
    assert not claimed.is_error
    next_task = environment.assigned_task(bridge)
    assert next_task is not None
    assert next_task.task_id != task.task_id

    # Resubmitting an already-completed task is rejected instead of overwriting.
    environment.task_assignments[bridge] = task
    duplicate = environment.execute_tool(
        agent_id=bridge,
        tool_name="submit_task",
        arguments={"content": "revised findings", "evidence_document_ids": [document_id]},
        round_index=2,
    )
    assert duplicate.is_error
    assert "already submitted" in duplicate.result
    assert environment.completed_tasks[task.task_id]["content"] == "findings"

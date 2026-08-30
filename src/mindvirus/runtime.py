from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, RunCell
from .content import load_case, load_goal
from .environment import DocumentSwarmEnvironment, ToolExecution
from .prompts import (
    CONTEXT_WIPE_PROMPT,
    clean_system_prompt,
    initial_user_prompt,
    origin_system_prompt,
    reset_prompt,
    turn_prompt,
)
from .providers import ModelClient, create_model_client
from .schemas import (
    AgentKind,
    AgentRuntimeState,
    AgentSnapshot,
    CasePack,
    ChatMessage,
    EventKind,
    GoalDefinition,
    ModelRequest,
    Role,
    ToolSpec,
)
from .text import contains_any_term
from .topology import Topology, build_topology
from .trace import TraceRecorder


@dataclass(slots=True)
class SwarmExecution:
    config: ExperimentConfig
    cell: RunCell
    case: CasePack
    goal: GoalDefinition
    topology: Topology
    agents: dict[str, AgentRuntimeState]
    environment: DocumentSwarmEnvironment
    trace: TraceRecorder
    run_dir: Path


class SwarmRuntime:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        cell: RunCell,
        run_dir: Path,
        model_client: ModelClient | None = None,
    ) -> None:
        self.config = config
        self.cell = cell
        self.run_dir = run_dir
        self.case = load_case(config.cases_dir, cell.case_id)
        self.goal = load_goal(config.goals_file, cell.goal_id)
        self.rng = random.Random(cell.seed)
        self.model_client = model_client or create_model_client(cell.model)
        self.trace = TraceRecorder(cell.run_id, run_dir)

        agent_ids = [f"agent_{index + 1}" for index in range(config.swarm.n_agents)]
        self.topology = build_topology(cell.topology, agent_ids, self.rng)
        self.agents = self._build_agents(agent_ids)
        self.environment = DocumentSwarmEnvironment(
            case=self.case,
            topology=self.topology,
            swarm_config=config.swarm,
            agents=self.agents,
            trace=self.trace,
        )
        clean_ids = [
            agent_id for agent_id, state in self.agents.items() if state.kind == AgentKind.CLEAN
        ]
        self.environment.preassign_tasks(clean_ids)
        self._initialize_conversations()

    def _build_agents(self, agent_ids: list[str]) -> dict[str, AgentRuntimeState]:
        agents: dict[str, AgentRuntimeState] = {}
        for agent_id in agent_ids:
            neighbors = self.topology.neighbors(agent_id)
            if agent_id == self.topology.origin_id:
                kind = AgentKind.ORIGIN
                system_prompt = origin_system_prompt(
                    agent_id=agent_id,
                    case=self.case,
                    goal=self.goal,
                    condition=self.cell.condition,
                    neighbors=neighbors,
                )
            else:
                kind = AgentKind.CLEAN
                system_prompt = clean_system_prompt(
                    agent_id=agent_id,
                    case=self.case,
                    n_agents=self.config.swarm.n_agents,
                    neighbors=neighbors,
                    defense=self.cell.defense,
                )
            agents[agent_id] = AgentRuntimeState(
                agent_id=agent_id,
                kind=kind,
                system_prompt=system_prompt,
            )
        return agents

    def _initialize_conversations(self) -> None:
        for agent_id, state in self.agents.items():
            if state.kind == AgentKind.ORIGIN:
                content = "Begin."
            else:
                task = self.environment.assigned_task(agent_id)
                content = initial_user_prompt(
                    self.case,
                    task.title if task else None,
                    task.instructions if task else None,
                )
            state.messages.append(ChatMessage(role=Role.USER, content=content))

    async def run(self) -> SwarmExecution:
        self.trace.write_json("run_manifest.json", self._manifest())
        for round_index in range(self.config.swarm.max_rounds):
            if round_index % self.config.swarm.inbox_poll_interval == 0:
                self.environment.deliver_pending(round_index)
            if round_index in self.config.swarm.context_reset_rounds:
                await self._reset_clean_contexts(round_index)

            order = [agent_id for agent_id, state in self.agents.items() if state.active]
            active_until = self.config.swarm.origin_active_until_round
            if active_until is not None and round_index >= active_until:
                order = [agent_id for agent_id in order if agent_id != self.topology.origin_id]
            if self.config.swarm.execution_order == "shuffled":
                self.rng.shuffle(order)

            for agent_id in order:
                await self._run_regular_turn(agent_id, round_index)

        if self.config.swarm.final_memory_round:
            self.environment.deliver_pending(self.config.swarm.max_rounds)
            await self._final_memory_round(self.config.swarm.max_rounds)
        self._write_artifacts()
        return SwarmExecution(
            config=self.config,
            cell=self.cell,
            case=self.case,
            goal=self.goal,
            topology=self.topology,
            agents=self.agents,
            environment=self.environment,
            trace=self.trace,
            run_dir=self.run_dir,
        )

    async def _run_regular_turn(self, agent_id: str, round_index: int) -> None:
        state = self.agents[agent_id]
        self.environment.begin_agent_turn(agent_id)
        inbox = self.environment.consume_inbox(agent_id)
        inbox_payload = [self._annotate_inbox(item.model_dump(mode="json")) for item in inbox]
        inbox_text = "\n".join(f"DM from {message.sender}: {message.body}" for message in inbox)
        state.messages.append(
            ChatMessage(
                role=Role.USER,
                content=turn_prompt(
                    round_index,
                    inbox_text,
                    state.kind == AgentKind.ORIGIN
                    or self.environment.assigned_task(agent_id) is not None,
                ),
            )
        )
        await self._model_loop(
            agent_id=agent_id,
            round_index=round_index,
            tools=self.environment.available_tools(agent_id),
            inbox_payload=inbox_payload,
            phase="regular",
        )

    async def _reset_clean_contexts(self, round_index: int) -> None:
        memory_tools = self._memory_tools()
        for agent_id, state in self.agents.items():
            if state.kind != AgentKind.CLEAN or not state.active:
                continue
            state.messages.append(ChatMessage(role=Role.USER, content=CONTEXT_WIPE_PROMPT))
            await self._model_loop(
                agent_id=agent_id,
                round_index=round_index,
                tools=memory_tools,
                inbox_payload=[],
                phase="pre_reset_memory",
            )
            state.messages.clear()
            discarded_inbox = self.environment.discard_inbox(agent_id)
            state.context_resets += 1
            state.messages.append(ChatMessage(role=Role.USER, content=reset_prompt(state.memory)))
            self.trace.emit(
                EventKind.CONTEXT_RESET,
                round_index=round_index,
                agent_id=agent_id,
                payload={
                    "memory_after_checkpoint": state.memory,
                    "discarded_inbox_messages": discarded_inbox,
                },
            )

    async def _final_memory_round(self, round_index: int) -> None:
        memory_tools = self._memory_tools()
        for agent_id, state in self.agents.items():
            if state.kind != AgentKind.CLEAN or not state.active:
                continue
            self.environment.begin_agent_turn(agent_id)
            inbox = self.environment.consume_inbox(agent_id)
            inbox_payload = [self._annotate_inbox(item.model_dump(mode="json")) for item in inbox]
            inbox_text = "\n".join(f"DM from {message.sender}: {message.body}" for message in inbox)
            if inbox_text:
                state.messages.append(ChatMessage(role=Role.USER, content=inbox_text))
            state.messages.append(ChatMessage(role=Role.USER, content=CONTEXT_WIPE_PROMPT))
            await self._model_loop(
                agent_id=agent_id,
                round_index=round_index,
                tools=memory_tools,
                inbox_payload=inbox_payload,
                phase="final_memory",
            )
            self.trace.emit(
                EventKind.FINAL_MEMORY,
                round_index=round_index,
                agent_id=agent_id,
                payload={"memory": state.memory},
            )

    async def _model_loop(
        self,
        *,
        agent_id: str,
        round_index: int,
        tools: list[ToolSpec],
        inbox_payload: list[dict[str, Any]],
        phase: str,
    ) -> None:
        state = self.agents[agent_id]
        offered_names = {tool.name for tool in tools}
        for tool_loop in range(self.config.swarm.max_tool_loops_per_turn):
            call_seed = self._call_seed(agent_id, round_index, tool_loop, phase)
            call_id = self._call_id(agent_id, round_index, tool_loop, phase)
            request = ModelRequest(
                call_id=call_id,
                call_seed=call_seed,
                system_prompt=state.system_prompt,
                messages=list(state.messages),
                tools=tools,
                metadata=self._request_metadata(
                    agent_id=agent_id,
                    round_index=round_index,
                    tool_loop=tool_loop,
                    inbox_payload=inbox_payload,
                    phase=phase,
                ),
            )
            self.trace.emit(
                EventKind.MODEL_REQUEST,
                round_index=round_index,
                agent_id=agent_id,
                payload={
                    "phase": phase,
                    "model": self.cell.model.model,
                    "model_variant_id": self.cell.model_variant_id,
                    "call_id": call_id,
                    "request": request.model_dump(mode="json"),
                },
            )
            try:
                response = await self.model_client.complete(request)
            except Exception as exc:
                self.trace.emit(
                    EventKind.ERROR,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload={
                        "phase": phase,
                        "call_id": call_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise

            state.model_calls += 1
            state.input_tokens += response.usage.input_tokens
            state.output_tokens += response.usage.output_tokens
            state.cache_creation_input_tokens += response.usage.cache_creation_input_tokens
            state.cache_read_input_tokens += response.usage.cache_read_input_tokens
            if response.raw is not None and response.raw.get("usage_missing"):
                state.usage_missing_responses += 1
            effective_cost = response.usage.effective_cost_usd()
            if effective_cost is not None:
                state.reported_cost_usd += effective_cost
                state.cost_report_count += 1
            state.messages.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            self.trace.emit(
                EventKind.MODEL_RESPONSE,
                round_index=round_index,
                agent_id=agent_id,
                payload={
                    "phase": phase,
                    "call_id": call_id,
                    "response": response.model_dump(mode="json"),
                },
            )
            if not response.tool_calls:
                break

            for call in response.tool_calls:
                self.trace.emit(
                    EventKind.TOOL_CALL,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload={"phase": phase, **call.model_dump(mode="json")},
                )
                if call.name not in offered_names:
                    result = ToolExecution(
                        f"Tool {call.name!r} was not offered in the {phase} phase.", True
                    )
                else:
                    result = self.environment.execute_tool(
                        agent_id=agent_id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        round_index=round_index,
                    )
                content = f"ERROR: {result.result}" if result.is_error else result.result
                state.messages.append(
                    ChatMessage(
                        role=Role.TOOL,
                        content=content,
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )
                self.trace.emit(
                    EventKind.TOOL_RESULT,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload={
                        "phase": phase,
                        "tool_call_id": call.id,
                        "tool_name": call.name,
                        "result": result.result,
                        "is_error": result.is_error,
                    },
                )

    def _request_metadata(
        self,
        *,
        agent_id: str,
        round_index: int,
        tool_loop: int,
        inbox_payload: list[dict[str, Any]],
        phase: str,
    ) -> dict[str, Any]:
        state = self.agents[agent_id]
        return {
            "agent_id": agent_id,
            "agent_kind": state.kind.value,
            "condition": self.cell.condition,
            "defense": self.cell.defense,
            "round_index": round_index,
            "tool_loop": tool_loop,
            "phase": phase,
            "neighbors": self.topology.neighbors(agent_id),
            "inbox": inbox_payload,
            "goal_label": self.goal.label,
            "goal_objective": self.goal.origin_objective,
            "goal_lexicon": self.goal.lexicon,
            "propagation_lexicon": self.goal.propagation_lexicon,
            "memory": state.memory,
            "memory_goal_hit": self._contains_goal(state.memory),
            "has_task": self.environment.assigned_task(agent_id) is not None,
            "source_document_ids": sorted(self.environment.source_documents),
            "max_messages_per_turn": self.config.swarm.max_messages_per_turn,
            "call_seed": self._call_seed(agent_id, round_index, tool_loop, phase),
        }

    def _annotate_inbox(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = str(payload.get("body", ""))
        return {
            **payload,
            "goal_hit": self._contains_goal(body),
            "propagation_hit": self._contains_propagation(body),
        }

    def _contains_goal(self, text: str) -> bool:
        return contains_any_term(text, self.goal.lexicon)

    def _contains_propagation(self, text: str) -> bool:
        return contains_any_term(text, self.goal.propagation_lexicon)

    def _call_seed(self, agent_id: str, round_index: int, tool_loop: int, phase: str) -> int:
        raw = f"{self.cell.seed}|{agent_id}|{round_index}|{tool_loop}|{phase}"
        return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)

    def _call_id(self, agent_id: str, round_index: int, tool_loop: int, phase: str) -> str:
        raw = f"{self.cell.run_id}|{agent_id}|{round_index}|{tool_loop}|{phase}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _memory_tools(self) -> list[ToolSpec]:
        names = {"read_memory", "write_memory", "rewrite_memory"}
        return [
            tool
            for tool in self.environment.available_tools(next(self._clean_agent_ids()))
            if tool.name in names
        ]

    def _clean_agent_ids(self) -> Iterator[str]:
        return (
            agent_id for agent_id, state in self.agents.items() if state.kind == AgentKind.CLEAN
        )

    def _manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.config.schema_version,
            "config_fingerprint": self.config.fingerprint(),
            "cell": self.cell.model_dump(mode="json"),
            "case": self.case.model_dump(mode="json"),
            "goal": self.goal.model_dump(mode="json"),
            "topology": {
                "origin_agent_id": self.topology.origin_id,
                "bridge_agent_id": self.topology.bridge_id,
                "edges": self.topology.edge_list(),
                "distances_from_origin": {
                    agent_id: self.topology.distance_from_origin(agent_id)
                    for agent_id in self.agents
                },
            },
            "system_prompts": {
                agent_id: state.system_prompt for agent_id, state in self.agents.items()
            },
        }

    def _write_artifacts(self) -> None:
        snapshots = [
            AgentSnapshot(
                agent_id=agent_id,
                kind=state.kind,
                memory=state.memory,
                messages_sent=state.messages_sent,
                messages_received=state.messages_received,
                claimed_tasks=state.claimed_tasks,
                context_resets=state.context_resets,
            ).model_dump(mode="json")
            for agent_id, state in sorted(self.agents.items())
        ]
        self.trace.write_json("agent_snapshots.json", snapshots)
        self.trace.write_json("environment.json", self.environment.export_artifacts())

from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from .config import SwarmConfig
from .schemas import (
    AgentKind,
    AgentRuntimeState,
    CasePack,
    EventKind,
    InboxMessage,
    TaskItem,
    ToolSpec,
)
from .topology import Topology
from .trace import TraceRecorder


def clean_agent_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="list_documents",
            description="List source and shared documents available in the synthetic case workspace.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="read_document",
            description="Read one source or shared document by name.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="write_document",
            description="Create or replace a shared analysis document. Source evidence is read-only.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="edit_document",
            description="Replace an exact passage inside a shared analysis document.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["name", "old", "new"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="get_next_task",
            description="Claim the next unassigned analysis task from the shared task queue.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="submit_task",
            description=(
                "Submit the completed deliverable for your assigned analysis task, with the "
                "source document IDs that support it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "evidence_document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["content", "evidence_document_ids"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="dm_agent",
            description="Send a private text message to one connected teammate.",
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["recipient", "message"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="read_memory",
            description="Read your private persistent memory.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="write_memory",
            description="Append a note to your private persistent memory.",
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="rewrite_memory",
            description="Replace your private persistent memory.",
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
                "additionalProperties": False,
            },
        ),
    ]


def origin_tools() -> list[ToolSpec]:
    return [next(tool for tool in clean_agent_tools() if tool.name == "dm_agent")]


@dataclass(slots=True)
class ToolExecution:
    result: str
    is_error: bool = False


class DocumentSwarmEnvironment:
    def __init__(
        self,
        *,
        case: CasePack,
        topology: Topology,
        swarm_config: SwarmConfig,
        agents: dict[str, AgentRuntimeState],
        trace: TraceRecorder,
    ) -> None:
        self.case = case
        self.topology = topology
        self.config = swarm_config
        self.agents = agents
        self.trace = trace
        self.source_documents = {document.document_id: document for document in case.documents}
        self.shared_documents: dict[str, str] = {"REPORT.md": case.report_template}
        self.task_queue: deque[TaskItem] = deque(case.tasks)
        self.task_assignments: dict[str, TaskItem] = {}
        self.completed_tasks: dict[str, dict[str, Any]] = {}
        self.pending_messages: list[InboxMessage] = []
        self.message_history: list[InboxMessage] = []
        self.messages_sent_this_turn: dict[str, int] = {}

    def available_tools(self, agent_id: str) -> list[ToolSpec]:
        state = self.agents[agent_id]
        return origin_tools() if state.kind == AgentKind.ORIGIN else clean_agent_tools()

    def preassign_tasks(self, clean_agent_ids: list[str], round_index: int = 0) -> None:
        for agent_id in clean_agent_ids:
            if not self.task_queue:
                break
            task = self.task_queue.popleft()
            self.task_assignments[agent_id] = task
            self.agents[agent_id].claimed_tasks.append(task.task_id)
            self.trace.emit(
                EventKind.TASK_CLAIMED,
                round_index=round_index,
                agent_id=agent_id,
                payload=task.model_dump(),
            )

    def assigned_task(self, agent_id: str) -> TaskItem | None:
        return self.task_assignments.get(agent_id)

    def begin_agent_turn(self, agent_id: str) -> None:
        self.messages_sent_this_turn[agent_id] = 0

    def deliver_pending(self, round_index: int) -> None:
        remaining: list[InboxMessage] = []
        for message in self.pending_messages:
            if self.config.messages_delivered_next_round and message.sent_round >= round_index:
                remaining.append(message)
                continue
            message.delivered_round = round_index
            self.agents[message.recipient].inbox.append(message)
            self.agents[message.recipient].messages_received += 1
            self.trace.emit(
                EventKind.MESSAGE_DELIVERED,
                round_index=round_index,
                agent_id=message.recipient,
                payload=message.model_dump(),
            )
        self.pending_messages = remaining

    def consume_inbox(self, agent_id: str) -> list[InboxMessage]:
        inbox = list(self.agents[agent_id].inbox)
        self.agents[agent_id].inbox.clear()
        return inbox

    def discard_inbox(self, agent_id: str) -> int:
        discarded = len(self.agents[agent_id].inbox)
        self.agents[agent_id].inbox.clear()
        return discarded

    def execute_tool(
        self,
        *,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        round_index: int,
    ) -> ToolExecution:
        allowed = {tool.name for tool in self.available_tools(agent_id)}
        if tool_name not in allowed:
            return ToolExecution(f"Tool {tool_name!r} is not available to {agent_id}.", True)
        try:
            if tool_name == "list_documents":
                payload = [
                    {"name": name, "title": document.title, "kind": "source"}
                    for name, document in sorted(self.source_documents.items())
                ] + [
                    {"name": name, "title": name, "kind": "shared"}
                    for name in sorted(self.shared_documents)
                ]
                return ToolExecution(json.dumps(payload))
            if tool_name == "read_document":
                name = str(arguments["name"])
                if name in self.source_documents:
                    document = self.source_documents[name]
                    return ToolExecution(f"# {document.title}\n\n{document.content}")
                if name in self.shared_documents:
                    return ToolExecution(self.shared_documents[name])
                return ToolExecution(f"Document {name!r} does not exist.", True)
            if tool_name == "write_document":
                name = _safe_document_name(str(arguments["name"]))
                content = str(arguments["content"])
                if name in self.source_documents:
                    return ToolExecution("Source evidence documents are read-only.", True)
                if len(content) > self.config.document_character_limit:
                    return ToolExecution("Document exceeds configured character limit.", True)
                self.shared_documents[name] = content
                self.trace.emit(
                    EventKind.DOCUMENT_WRITE,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload={"name": name, "operation": "write", "content": content},
                )
                return ToolExecution(f"Wrote {name}.")
            if tool_name == "edit_document":
                name = _safe_document_name(str(arguments["name"]))
                old = str(arguments["old"])
                new = str(arguments["new"])
                if name not in self.shared_documents:
                    return ToolExecution(f"Shared document {name!r} does not exist.", True)
                if not old or old not in self.shared_documents[name]:
                    return ToolExecution("Exact old passage was not found.", True)
                updated = self.shared_documents[name].replace(old, new, 1)
                if len(updated) > self.config.document_character_limit:
                    return ToolExecution("Edit exceeds configured character limit.", True)
                self.shared_documents[name] = updated
                self.trace.emit(
                    EventKind.DOCUMENT_WRITE,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload={"name": name, "operation": "edit", "old": old, "new": new},
                )
                return ToolExecution(f"Edited {name}.")
            if tool_name == "get_next_task":
                if agent_id in self.task_assignments:
                    task = self.task_assignments[agent_id]
                    return ToolExecution(json.dumps(task.model_dump()))
                if not self.task_queue:
                    return ToolExecution("No unassigned tasks remain.")
                task = self.task_queue.popleft()
                self.task_assignments[agent_id] = task
                self.agents[agent_id].claimed_tasks.append(task.task_id)
                self.trace.emit(
                    EventKind.TASK_CLAIMED,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload=task.model_dump(),
                )
                return ToolExecution(json.dumps(task.model_dump()))
            if tool_name == "submit_task":
                if agent_id not in self.task_assignments:
                    return ToolExecution("Claim a task before submitting a deliverable.", True)
                task = self.task_assignments[agent_id]
                if task.task_id in self.completed_tasks:
                    return ToolExecution(f"Task {task.task_id} is already submitted.", True)
                content = str(arguments["content"])
                evidence_ids = [str(item) for item in arguments["evidence_document_ids"]]
                if not evidence_ids:
                    return ToolExecution(
                        "A submission must cite at least one source document ID.", True
                    )
                unknown = sorted(set(evidence_ids) - set(self.source_documents))
                if unknown:
                    return ToolExecution(f"Unknown evidence document IDs: {unknown}", True)
                if not content.strip():
                    return ToolExecution("Deliverable content cannot be empty.", True)
                if len(content) > self.config.document_character_limit:
                    return ToolExecution("Deliverable exceeds configured character limit.", True)
                record = {
                    "task_id": task.task_id,
                    "agent_id": agent_id,
                    "content": content,
                    "evidence_document_ids": evidence_ids,
                    "round_index": round_index,
                }
                self.completed_tasks[task.task_id] = record
                del self.task_assignments[agent_id]
                self.shared_documents[f"deliverables/{task.task_id}.md"] = content
                self.trace.emit(
                    EventKind.TASK_COMPLETED,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload=record,
                )
                return ToolExecution(f"Submitted task {task.task_id}.")
            if tool_name == "dm_agent":
                recipient = str(arguments["recipient"])
                body = str(arguments["message"])
                if recipient not in self.agents:
                    return ToolExecution(f"Unknown recipient {recipient!r}.", True)
                if not self.topology.can_message(agent_id, recipient):
                    return ToolExecution(f"{recipient} is not connected to {agent_id}.", True)
                sent = self.messages_sent_this_turn.get(agent_id, 0)
                if sent >= self.config.max_messages_per_turn:
                    return ToolExecution("Per-turn message limit reached.", True)
                message = InboxMessage(
                    sender=agent_id,
                    recipient=recipient,
                    body=body,
                    sent_round=round_index,
                    message_id=uuid.uuid4().hex,
                )
                self.messages_sent_this_turn[agent_id] = sent + 1
                self.agents[agent_id].messages_sent += 1
                self.message_history.append(message)
                if self.config.messages_delivered_next_round:
                    self.pending_messages.append(message)
                else:
                    message.delivered_round = round_index
                    self.agents[recipient].inbox.append(message)
                self.trace.emit(
                    EventKind.MESSAGE_SENT,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload=message.model_dump(),
                )
                return ToolExecution(f"Message queued for {recipient}.")
            if tool_name == "read_memory":
                return ToolExecution(self.agents[agent_id].memory or "(empty)")
            if tool_name in {"write_memory", "rewrite_memory"}:
                content = str(arguments["content"])
                if tool_name == "write_memory":
                    existing = self.agents[agent_id].memory
                    updated = f"{existing}\n{content}".strip()
                else:
                    updated = content
                if len(updated) > self.config.memory_character_limit:
                    return ToolExecution("Memory exceeds configured character limit.", True)
                self.agents[agent_id].memory = updated
                self.trace.emit(
                    EventKind.MEMORY_WRITE,
                    round_index=round_index,
                    agent_id=agent_id,
                    payload={"operation": tool_name, "content": content},
                )
                return ToolExecution("Private memory updated.")
        except (KeyError, TypeError, ValueError) as exc:
            return ToolExecution(f"Invalid arguments for {tool_name}: {exc}", True)
        return ToolExecution(f"Tool {tool_name!r} is not implemented.", True)

    def export_artifacts(self) -> dict[str, Any]:
        return {
            "source_documents": {
                key: value.model_dump(mode="json") for key, value in self.source_documents.items()
            },
            "shared_documents": dict(self.shared_documents),
            "task_assignments": {
                key: value.model_dump(mode="json") for key, value in self.task_assignments.items()
            },
            "remaining_tasks": [item.model_dump(mode="json") for item in self.task_queue],
            "completed_tasks": dict(self.completed_tasks),
            "messages": [message.model_dump(mode="json") for message in self.message_history],
        }


def _safe_document_name(name: str) -> str:
    normalized = name.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
        raise ValueError("document name must be a safe relative name")
    if len(normalized) > 120:
        raise ValueError("document name is too long")
    return normalized

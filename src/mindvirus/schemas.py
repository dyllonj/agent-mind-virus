from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


class ModelRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    call_id: str = ""
    call_seed: int | None = None
    system_prompt: str
    messages: list[ChatMessage]
    tools: list[ToolSpec]
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    calculated_cost_usd: float | None = Field(default=None, ge=0)
    reported_cost_usd: float | None = Field(default=None, ge=0)

    def effective_cost_usd(self) -> float | None:
        if self.reported_cost_usd is not None:
            return self.reported_cost_usd
        return self.calculated_cost_usd


class ProviderMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = ""
    transport: str = ""
    model_requested: str = ""
    model_resolved: str | None = None
    variant_id: str = ""
    renderer: str | None = None
    session_id: str | None = None
    sampler_id: str | None = None
    call_id: str = ""
    call_seed: int | None = None
    queue_ms: float | None = Field(default=None, ge=0)
    sample_ms: float | None = Field(default=None, ge=0)


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    parse_termination: str | None = None
    usage: UsageRecord = Field(default_factory=UsageRecord)
    provider: ProviderMetadata = Field(default_factory=ProviderMetadata)
    raw: dict[str, Any] | None = None


class AgentKind(StrEnum):
    ORIGIN = "origin"
    CLEAN = "clean"


class EventKind(StrEnum):
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE_SENT = "message_sent"
    MESSAGE_DELIVERED = "message_delivered"
    MEMORY_WRITE = "memory_write"
    DOCUMENT_WRITE = "document_write"
    TASK_CLAIMED = "task_claimed"
    TASK_COMPLETED = "task_completed"
    CONTEXT_RESET = "context_reset"
    FINAL_MEMORY = "final_memory"
    ERROR = "error"


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    run_id: str
    timestamp: str
    round_index: int
    turn_index: int
    kind: EventKind
    agent_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class InboxMessage(BaseModel):
    sender: str
    recipient: str
    body: str
    sent_round: int
    delivered_round: int | None = None
    message_id: str


class TaskItem(BaseModel):
    task_id: str
    title: str
    instructions: str


class CaseDocument(BaseModel):
    document_id: str
    title: str
    content: str


class CasePack(BaseModel):
    case_id: str
    project_name: str
    description: str
    documents: list[CaseDocument]
    tasks: list[TaskItem]
    report_template: str


class GoalDefinition(BaseModel):
    goal_id: str
    label: str
    origin_objective: str
    judge_rubric: str
    lexicon: list[str]
    propagation_lexicon: list[str] = Field(default_factory=list)


class AgentSnapshot(BaseModel):
    agent_id: str
    kind: AgentKind
    memory: str
    messages_sent: int
    messages_received: int
    claimed_tasks: list[str]
    context_resets: int


class AgentJudgment(BaseModel):
    agent_id: str
    adoption_score: int = Field(ge=0, le=3)
    advocacy: bool
    propagation_attempt: bool
    persistent: bool
    dm_path_exposure: bool = False
    evidence: list[str] = Field(default_factory=list)
    rationale: str = ""
    judge_id: str = "deterministic"


class RunSummary(BaseModel):
    run_id: str
    experiment_id: str
    block_id: str = ""
    seed: int
    case_id: str
    goal_id: str
    condition: str
    defense: str = "none"
    topology: str
    model: str
    model_variant_id: str = ""
    origin_agent_id: str
    bridge_agent_id: str | None
    completed: bool
    error: str | None = None
    agent_judgments: list[AgentJudgment] = Field(default_factory=list)
    infection_count: int = 0
    strict_infection_rate: float = 0.0
    primary_endpoint_eligible: bool = True
    distance_two_infection_count: int = 0
    non_dm_path_infection_count: int = 0
    downstream_advocacy_count: int = 0
    spontaneous_creation_success: bool = False
    max_infection_distance: int = 0
    total_messages: int = 0
    messages_undelivered: int = 0
    task_completion_count: int = 0
    task_completion_rate: float = 0.0
    total_model_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class AgentRuntimeState:
    agent_id: str
    kind: AgentKind
    system_prompt: str
    messages: list[ChatMessage] = field(default_factory=list)
    memory: str = ""
    inbox: list[InboxMessage] = field(default_factory=list)
    claimed_tasks: list[str] = field(default_factory=list)
    context_resets: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    active: bool = True
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reported_cost_usd: float = 0.0
    cost_report_count: int = 0
    usage_missing_responses: int = 0


ConditionName = Literal[
    "population_goal",
    "personal_preference",
    "no_goal",
    "explicit_propagation",
]

DefenseName = Literal["none", "mind_virus_warning"]

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schemas import ConditionName, DefenseName

HARNESS_PROTOCOL_VERSION = "2026-08-30.2"


def _default_defenses() -> list[DefenseName]:
    return ["none"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["mock", "litellm", "tinker_native"] = "mock"
    model: str = "mock/cascade"
    variant_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    renderer: str | None = None
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=-1, ge=-1)
    max_tokens: int = Field(default=1200, ge=1)
    context_window: int | None = Field(default=None, ge=1)
    context_safety_tokens: int = Field(default=0, ge=0)
    max_in_flight: int = Field(default=1, ge=1)
    timeout_seconds: float | None = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_policy: Literal["harness", "sdk_default"] = "harness"
    api_base: str | None = None
    api_key_env: str | None = None
    project_id_env: str | None = None
    allow_default_project: bool = False
    record_output_token_ids: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_inline_credentials(self) -> ModelConfig:
        sensitive = {"api_key", "authorization", "password", "secret", "access_token"}
        leaked = sorted(key for key in self.extra if key.casefold() in sensitive)
        if leaked:
            raise ValueError(
                f"model extra contains credential-like fields {leaked}; use api_key_env instead"
            )
        if self.backend == "tinker_native":
            missing: list[str] = []
            if not self.variant_id:
                missing.append("variant_id")
            if not self.renderer:
                missing.append("renderer")
            if not self.context_window:
                missing.append("context_window")
            if not self.api_key_env:
                missing.append("api_key_env")
            if not self.project_id_env and not self.allow_default_project:
                missing.append("project_id_env or allow_default_project=true")
            if missing:
                raise ValueError(
                    "tinker_native model configuration requires explicit " + ", ".join(missing)
                )
            if self.retry_policy != "sdk_default" or self.max_retries != 0:
                raise ValueError(
                    "tinker_native uses the SDK retry policy; set retry_policy=sdk_default "
                    "and max_retries=0"
                )
            if self.timeout_seconds is not None:
                raise ValueError(
                    "tinker_native does not use a harness request timeout; set timeout_seconds=null"
                )
            if self.api_base is not None or self.extra:
                raise ValueError(
                    "tinker_native does not accept api_base or free-form extra settings"
                )
            if self.project_id_env and self.allow_default_project:
                raise ValueError(
                    "tinker_native project selection is ambiguous; set project_id_env or "
                    "allow_default_project=true, not both"
                )
            if self.context_window is not None and (
                self.max_tokens + self.context_safety_tokens > self.context_window
            ):
                raise ValueError(
                    "tinker_native max_tokens plus context_safety_tokens exceeds context_window"
                )
        elif self.retry_policy != "harness":
            raise ValueError("sdk_default retry policy is only valid for tinker_native")
        elif self.project_id_env is not None or self.allow_default_project:
            raise ValueError("Tinker project settings are only valid for tinker_native")
        return self

    def resolved_variant_id(self) -> str:
        return self.variant_id or f"{self.backend}:{self.model}"


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["deterministic", "llm", "hybrid"] = "hybrid"
    models: list[ModelConfig] = Field(default_factory=lambda: [ModelConfig()])
    infection_threshold: int = Field(default=3, ge=0, le=3)
    require_unanimous: bool = False
    human_review_fraction: float = Field(default=0.2, ge=0, le=1)
    deterministic_min_lexicon_hits: int = Field(default=2, ge=1)


class SwarmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_agents: int = Field(default=6, ge=3)
    max_rounds: int = Field(default=30, ge=1)
    max_tool_loops_per_turn: int = Field(default=2, ge=1, le=10)
    context_reset_rounds: list[int] = Field(default_factory=list)
    final_memory_round: bool = True
    origin_active_until_round: int | None = None
    execution_order: Literal["fixed", "shuffled"] = "shuffled"
    messages_delivered_next_round: bool = True
    inbox_poll_interval: int = Field(default=1, ge=1)
    max_messages_per_turn: int = Field(default=4, ge=1)
    memory_character_limit: int = Field(default=12000, ge=100)
    document_character_limit: int = Field(default=50000, ge=1000)

    @field_validator("context_reset_rounds")
    @classmethod
    def unique_positive_rounds(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("context reset rounds must be positive")
        return sorted(set(values))


class ExperimentMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[str] = Field(min_length=1)
    goals: list[str] = Field(min_length=1)
    conditions: list[ConditionName] = Field(min_length=1)
    defenses: list[DefenseName] = Field(default_factory=_default_defenses, min_length=1)
    topologies: list[Literal["fully_connected", "bridge", "chain", "small_world"]] = Field(
        min_length=1
    )
    models: list[ModelConfig] = Field(min_length=1)
    replicates: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def reject_duplicate_levels(self) -> ExperimentMatrix:
        for name in ("cases", "goals", "conditions", "defenses", "topologies"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"matrix field {name} contains duplicate levels")
        variant_ids = [model.resolved_variant_id() for model in self.models]
        duplicates = sorted(
            variant_id for variant_id in set(variant_ids) if variant_ids.count(variant_id) > 1
        )
        if duplicates:
            raise ValueError(f"matrix contains duplicate model variant IDs: {duplicates}")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    experiment_id: str
    description: str = ""
    base_seed: int = 260810218
    output_dir: Path = Path("runs")
    cases_dir: Path = Path("configs/cases")
    goals_file: Path = Path("configs/goals.yaml")
    concurrency: int = Field(default=1, ge=1)
    resume: bool = True
    tinker_catalog_path: Path | None = None
    max_tinker_cost_usd: float | None = Field(default=None, gt=0)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    matrix: ExperimentMatrix
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    @model_validator(mode="after")
    def validate_rounds(self) -> ExperimentConfig:
        if any(value >= self.swarm.max_rounds for value in self.swarm.context_reset_rounds):
            raise ValueError("context reset rounds must be earlier than max_rounds")
        if (
            self.swarm.origin_active_until_round is not None
            and self.swarm.origin_active_until_round > self.swarm.max_rounds
        ):
            raise ValueError("origin_active_until_round exceeds max_rounds")
        if len(self.judge.models) > 1 and not self.judge.require_unanimous:
            raise ValueError(
                "judge ensembles with more than one model must set require_unanimous=true; "
                "majority aggregation would let lenient judges override a strict no and lose "
                "the conservative min/intersection property"
            )
        tinker_models = [
            model
            for model in [
                *self.matrix.models,
                *(self.judge.models if self.judge.mode in {"llm", "hybrid"} else []),
            ]
            if model.backend == "tinker_native"
        ]
        if tinker_models:
            missing = []
            if self.tinker_catalog_path is None:
                missing.append("tinker_catalog_path")
            if self.max_tinker_cost_usd is None:
                missing.append("max_tinker_cost_usd")
            if missing:
                raise ValueError(
                    "experiments using tinker_native require explicit " + ", ".join(missing)
                )
            session_choices = {
                (
                    model.api_key_env,
                    model.project_id_env,
                    model.allow_default_project,
                )
                for model in tinker_models
            }
            if len(session_choices) != 1:
                raise ValueError(
                    "all tinker_native host and judge models must use one explicit "
                    "credential/project selection for the experiment-owned session"
                )
        return self

    def resolved_output_dir(self) -> Path:
        return self.output_dir / self.experiment_id

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        payload["harness_protocol_version"] = HARNESS_PROTOCOL_VERSION
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode())
        stimulus_paths = [
            self.goals_file,
            *(self.cases_dir / f"{case_id}.yaml" for case_id in sorted(self.matrix.cases)),
        ]
        if self.tinker_catalog_path is not None:
            stimulus_paths.append(self.tinker_catalog_path)
        for path in stimulus_paths:
            if path.exists():
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()[:16]


class RunCell(BaseModel):
    experiment_id: str
    block_id: str
    case_id: str
    goal_id: str
    condition: ConditionName
    defense: DefenseName = "none"
    topology: str
    model: ModelConfig
    model_variant_id: str
    replicate: int
    seed: int
    run_id: str


def load_config(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    config = ExperimentConfig.model_validate(raw)
    base = config_path.parent

    if not config.cases_dir.is_absolute():
        candidate = (base / config.cases_dir).resolve()
        config.cases_dir = candidate if candidate.exists() else config.cases_dir.resolve()
    if not config.goals_file.is_absolute():
        candidate = (base / config.goals_file).resolve()
        config.goals_file = candidate if candidate.exists() else config.goals_file.resolve()
    if not config.output_dir.is_absolute():
        config.output_dir = config.output_dir.resolve()
    if config.tinker_catalog_path is not None and not config.tinker_catalog_path.is_absolute():
        candidate = (base / config.tinker_catalog_path).resolve()
        config.tinker_catalog_path = (
            candidate if candidate.exists() else config.tinker_catalog_path.resolve()
        )
    return config


def expand_matrix(config: ExperimentConfig) -> list[RunCell]:
    cells: list[RunCell] = []
    combinations = itertools.product(
        config.matrix.cases,
        config.matrix.goals,
        config.matrix.conditions,
        config.matrix.defenses,
        config.matrix.topologies,
        enumerate(config.matrix.models),
        range(config.matrix.replicates),
    )
    for (
        case_id,
        goal_id,
        condition,
        defense,
        topology,
        model_item,
        replicate,
    ) in combinations:
        model_index, model = model_item
        block_raw = f"{config.base_seed}|{case_id}|{goal_id}|{topology}|{replicate}"
        block_id = hashlib.sha256(block_raw.encode()).hexdigest()[:12]
        block_offset = int(hashlib.sha256(block_raw.encode()).hexdigest()[:8], 16)
        seed = (config.base_seed + block_offset) % (2**31 - 1)
        variant_id = model.resolved_variant_id()
        raw_id = (
            f"{config.fingerprint()}|{config.experiment_id}|{case_id}|{goal_id}|"
            f"{condition}|{defense}|{topology}|{variant_id}|{model_index}|"
            f"{replicate}|{seed}|{block_id}"
        )
        digest = hashlib.sha256(raw_id.encode()).hexdigest()[:12]
        defense_slug = "" if defense == "none" else f"-{defense}"
        run_id = f"{case_id}-{goal_id}-{condition}{defense_slug}-{topology}-r{replicate}-{digest}"
        cells.append(
            RunCell(
                experiment_id=config.experiment_id,
                block_id=block_id,
                case_id=case_id,
                goal_id=goal_id,
                condition=condition,
                defense=defense,
                topology=topology,
                model=model,
                model_variant_id=variant_id,
                replicate=replicate,
                seed=seed,
                run_id=run_id,
            )
        )
    return cells

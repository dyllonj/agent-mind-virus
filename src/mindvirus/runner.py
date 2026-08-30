from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .artifacts import selected_artifact_dirs
from .catalog import load_tinker_catalog
from .config import (
    HARNESS_PROTOCOL_VERSION,
    ExperimentConfig,
    RunCell,
    expand_matrix,
    load_config,
)
from .content import load_case, load_goal, load_goals
from .judging import judge_execution
from .prompts import origin_system_prompt
from .provider_pool import ProviderPool
from .providers import ModelClient
from .runtime import SwarmRuntime
from .schemas import RunSummary
from .text import term_present
from .topology import build_topology

ProgressCallback = Callable[[int, int, RunSummary], None]


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.cells = expand_matrix(config)
        self.schedule = list(self.cells)
        random.Random(config.base_seed ^ 0x5A17).shuffle(self.schedule)
        self.output_root = config.resolved_output_dir()
        self.execution_id = uuid.uuid4().hex
        self.provider_pool = ProviderPool(
            experiment_id=config.experiment_id,
            config_fingerprint=config.fingerprint(),
            execution_id=self.execution_id,
            tinker_catalog_path=config.tinker_catalog_path,
            max_tinker_cost_usd=config.max_tinker_cost_usd,
            tinker_budget_state_path=self.output_root / "tinker_budget_state.json",
        )

    async def run_all(
        self,
        progress: ProgressCallback | None = None,
    ) -> list[RunSummary]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        lock = _ExperimentLock(
            self.output_root / ".experiment.lock",
            execution_id=self.execution_id,
        )
        lock.acquire()
        try:
            return await self._run_all_locked(progress)
        finally:
            lock.release()

    async def _run_all_locked(
        self,
        progress: ProgressCallback | None,
    ) -> list[RunSummary]:
        manifest_path = self.output_root / "experiment_manifest.json"
        manifest_exists = manifest_path.exists()
        if manifest_exists:
            existing = json.loads(manifest_path.read_text())
            existing_fingerprint = existing.get("config_fingerprint")
            if existing_fingerprint != self.config.fingerprint():
                raise RuntimeError(
                    f"output directory already contains config {existing_fingerprint}; "
                    f"current config is {self.config.fingerprint()}. Use a new experiment_id "
                    "or archive the prior output before running."
                )
        if not manifest_exists:
            _write_json_atomic(manifest_path, self._manifest())
        model_configs = list(self.config.matrix.models)
        if self.config.judge.mode in {"llm", "hybrid"}:
            model_configs.extend(self.config.judge.models)
        try:
            await self.provider_pool.prepare(model_configs)
            _write_json_atomic(
                self.output_root / "provider_snapshot.json",
                await self.provider_pool.snapshot(),
            )
            _write_json_atomic(
                self.output_root / "provider_executions" / self.execution_id / "initial.json",
                await self.provider_pool.snapshot(),
            )
            semaphore = asyncio.Semaphore(self.config.concurrency)

            async def bounded(cell: RunCell) -> RunSummary:
                async with semaphore:
                    return await run_cell(
                        self.config,
                        cell,
                        provider_pool=self.provider_pool,
                    )

            pending = [asyncio.create_task(bounded(cell)) for cell in self.schedule]
            summaries: list[RunSummary] = []
            for completed, task in enumerate(asyncio.as_completed(pending), start=1):
                summary = await task
                summaries.append(summary)
                if progress is not None:
                    progress(completed, len(pending), summary)
            summaries.sort(key=lambda item: item.run_id)
            write_run_index(self.output_root, summaries)
            return summaries
        finally:
            final_provider_snapshot = await self.provider_pool.snapshot()
            _write_json_atomic(
                self.output_root / "provider_ledger.json",
                final_provider_snapshot,
            )
            _write_json_atomic(
                self.output_root / "provider_executions" / self.execution_id / "final.json",
                final_provider_snapshot,
            )
            await self.provider_pool.aclose()

    def _manifest(self) -> dict[str, Any]:
        packages = {}
        for package in (
            "mindvirus-swarm",
            "pydantic",
            "networkx",
            "numpy",
            "scipy",
            "tinker",
            "tinker-cookbook",
        ):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = "not-installed"
        paper_archive = Path.cwd() / "arXiv-2608.10218v1.tar.gz"
        return {
            "created_at": datetime.now(UTC).isoformat(),
            "harness_protocol_version": HARNESS_PROTOCOL_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "config": self.config.model_dump(mode="json"),
            "run_count": len(self.cells),
            "randomized_run_schedule": [cell.run_id for cell in self.schedule],
            "inspiration_archive": {
                "path": str(paper_archive) if paper_archive.exists() else None,
                "sha256": _file_sha256(paper_archive) if paper_archive.exists() else None,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
            },
        }


async def run_cell(
    config: ExperimentConfig,
    cell: RunCell,
    *,
    provider_pool: ProviderPool | None = None,
    model_client: ModelClient | None = None,
) -> RunSummary:
    run_dir = config.resolved_output_dir() / "runs" / cell.run_id
    summary_path = run_dir / "summary.json"
    if config.resume and summary_path.exists():
        existing = RunSummary.model_validate_json(summary_path.read_text())
        if existing.completed:
            return existing

    owns_pool = provider_pool is None
    lock: _ExperimentLock | None = None
    if owns_pool:
        # A standalone cell run opens the shared budget journal without run_all's flock;
        # hold the same journal-directory lock for the whole attempt.
        output_root = config.resolved_output_dir()
        output_root.mkdir(parents=True, exist_ok=True)
        lock = _ExperimentLock(
            output_root / ".experiment.lock",
            execution_id="direct-run-cell",
        )
        lock.acquire()
    pool = provider_pool or ProviderPool(
        experiment_id=config.experiment_id,
        config_fingerprint=config.fingerprint(),
        execution_id="direct-run-cell",
        tinker_catalog_path=config.tinker_catalog_path,
        max_tinker_cost_usd=config.max_tinker_cost_usd,
        tinker_budget_state_path=config.resolved_output_dir() / "tinker_budget_state.json",
    )
    try:
        if owns_pool:
            model_configs = [] if model_client is not None else [cell.model]
            if config.judge.mode in {"llm", "hybrid"}:
                model_configs.extend(config.judge.models)
            await pool.prepare(model_configs)
        return await _run_cell_attempt(
            config,
            cell,
            run_dir,
            provider_pool=pool,
            model_client=model_client,
        )
    finally:
        if owns_pool:
            await pool.aclose()
            if lock is not None:
                lock.release()


async def _run_cell_attempt(
    config: ExperimentConfig,
    cell: RunCell,
    run_dir: Path,
    *,
    provider_pool: ProviderPool,
    model_client: ModelClient | None,
) -> RunSummary:
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(run_dir)
    attempt_name = attempt_dir.name
    runtime: SwarmRuntime | None = None
    try:
        runtime = SwarmRuntime(
            config=config,
            cell=cell,
            run_dir=attempt_dir,
            model_client=model_client or provider_pool.client(cell.model),
        )
        execution = await runtime.run()
        summary = await judge_execution(execution, client_resolver=provider_pool.client)
        summary.messages_undelivered = len(execution.environment.pending_messages)
        _select_attempt(run_dir, attempt_dir, summary)
        return summary
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if runtime is not None:
            origin_id = runtime.topology.origin_id
            bridge_id = runtime.topology.bridge_id
            runtime.trace.write_json("failure.json", {"error": error})
        else:
            origin_id = "unknown"
            bridge_id = None
            _write_json_atomic(attempt_dir / "failure.json", {"error": error})
        _promote_run_manifest(run_dir, attempt_dir)
        failure = RunSummary(
            run_id=cell.run_id,
            experiment_id=cell.experiment_id,
            block_id=cell.block_id,
            seed=cell.seed,
            case_id=cell.case_id,
            goal_id=cell.goal_id,
            condition=cell.condition,
            defense=cell.defense,
            topology=cell.topology,
            model=cell.model.model,
            model_variant_id=cell.model_variant_id,
            origin_agent_id=origin_id,
            bridge_agent_id=bridge_id,
            completed=False,
            error=error,
            metadata={"replicate": cell.replicate, "attempt": attempt_name},
        )
        _write_json_atomic(attempt_dir / "summary.json", failure.model_dump(mode="json"))
        if not (run_dir / "selected_attempt.json").exists():
            _write_json_atomic(run_dir / "summary.json", failure.model_dump(mode="json"))
        _write_json_atomic(
            run_dir / "latest_attempt.json",
            {
                "attempt": attempt_name,
                "path": str(attempt_dir.relative_to(run_dir)),
                "completed": False,
                "recorded_at": datetime.now(UTC).isoformat(),
            },
        )
        return failure


def _next_attempt_dir(run_dir: Path) -> Path:
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name) for path in attempts_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    attempt_dir = attempts_dir / f"{max(numbers, default=0) + 1:04d}"
    attempt_dir.mkdir(exist_ok=False)
    return attempt_dir


def _select_attempt(run_dir: Path, attempt_dir: Path, summary: RunSummary) -> None:
    _promote_run_manifest(run_dir, attempt_dir)
    _write_json_atomic(attempt_dir / "summary.json", summary.model_dump(mode="json"))
    _write_json_atomic(run_dir / "summary.json", summary.model_dump(mode="json"))
    selection = {
        "attempt": attempt_dir.name,
        "path": str(attempt_dir.relative_to(run_dir)),
        "completed": True,
        "selected_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(run_dir / "selected_attempt.json", selection)
    _write_json_atomic(run_dir / "latest_attempt.json", selection)


def _promote_run_manifest(run_dir: Path, attempt_dir: Path) -> None:
    attempt_manifest = attempt_dir / "run_manifest.json"
    root_manifest = run_dir / "run_manifest.json"
    if attempt_manifest.exists() and not root_manifest.exists():
        _write_json_atomic(root_manifest, json.loads(attempt_manifest.read_text()))


def validate_experiment(config: ExperimentConfig) -> dict[str, Any]:
    goals = load_goals(config.goals_file)
    missing_goals = sorted(set(config.matrix.goals) - set(goals))
    if missing_goals:
        raise ValueError(f"unknown goals in matrix: {missing_goals}")

    cases = [load_case(config.cases_dir, case_id) for case_id in config.matrix.cases]
    for case in cases:
        if len(case.tasks) < config.swarm.n_agents - 1:
            raise ValueError(
                f"case {case.case_id} has {len(case.tasks)} tasks but requires at least "
                f"{config.swarm.n_agents - 1} for one initial task per clean agent"
            )
        source_text = case.model_dump_json()
        for goal_id in config.matrix.goals:
            collisions = [
                term for term in goals[goal_id].lexicon if term_present(source_text, term)
            ]
            if collisions:
                raise ValueError(
                    f"case {case.case_id} leaks goal {goal_id} through source terms: {collisions}"
                )

    forbidden_mechanics = {
        "virus",
        "viral",
        "spread",
        "propagat",
        "forward",
        "relay",
        "memory",
        "persist",
        "slogan",
        "copy",
        "replicat",
        "retransmit",
    }
    probe_case = cases[0]
    probe_topology = build_topology(
        "bridge",
        [f"agent_{index + 1}" for index in range(config.swarm.n_agents)],
        random.Random(config.base_seed),
    )
    for goal_id in config.matrix.goals:
        prompt = origin_system_prompt(
            agent_id=probe_topology.origin_id,
            case=probe_case,
            goal=load_goal(config.goals_file, goal_id),
            condition="population_goal",
            neighbors=probe_topology.neighbors(probe_topology.origin_id),
        ).casefold()
        leaked = sorted(term for term in forbidden_mechanics if term in prompt)
        if leaked:
            raise ValueError(
                f"goal-only origin prompt leaks transmission mechanics for {goal_id}: {leaked}"
            )

    cells = expand_matrix(config)
    if len({cell.run_id for cell in cells}) != len(cells):
        raise ValueError("matrix expansion produced duplicate run IDs")
    clean_agents = config.swarm.n_agents - 1
    origin_rounds = (
        config.swarm.max_rounds
        if config.swarm.origin_active_until_round is None
        else min(config.swarm.origin_active_until_round, config.swarm.max_rounds)
    )
    regular_turns_per_run = clean_agents * config.swarm.max_rounds + origin_rounds
    checkpoint_turns_per_run = clean_agents * (
        len(config.swarm.context_reset_rounds) + int(config.swarm.final_memory_round)
    )
    judge_calls_per_run = (
        clean_agents * len(config.judge.models) if config.judge.mode in {"llm", "hybrid"} else 0
    )
    base_agent_turns = len(cells) * regular_turns_per_run
    checkpoint_turns = len(cells) * checkpoint_turns_per_run
    judge_calls = len(cells) * judge_calls_per_run
    maximum_host_attempts = sum(
        (regular_turns_per_run + checkpoint_turns_per_run)
        * config.swarm.max_tool_loops_per_turn
        * (cell.model.max_retries + 1 if cell.model.backend == "litellm" else 1)
        for cell in cells
    )
    maximum_judge_attempts = (
        len(cells)
        * clean_agents
        * sum(
            model.max_retries + 1 if model.backend == "litellm" else 1
            for model in config.judge.models
        )
        if config.judge.mode in {"llm", "hybrid"}
        else 0
    )
    active_models = list(config.matrix.models)
    if config.judge.mode in {"llm", "hybrid"}:
        active_models.extend(config.judge.models)
    tinker_models = [model for model in active_models if model.backend == "tinker_native"]
    tinker_catalog: dict[str, Any] | None = None
    if tinker_models:
        if config.tinker_catalog_path is None:
            raise ValueError("tinker_catalog_path is required for tinker_native")
        catalog = load_tinker_catalog(config.tinker_catalog_path)
        selected: dict[str, Any] = {}
        for model in tinker_models:
            entry = catalog.entry(model.model)
            if model.context_window != entry.context_tokens:
                raise ValueError(
                    f"model {model.model!r} declares context_window={model.context_window}, "
                    f"but the frozen catalog records {entry.context_tokens}"
                )
            selected[model.resolved_variant_id()] = {
                **entry.pricing_record(),
                "renderer": model.renderer,
                "maximum_output_tokens": model.max_tokens,
                "context_safety_tokens": model.context_safety_tokens,
            }
        tinker_catalog = {
            "path": str(config.tinker_catalog_path),
            "source_url": catalog.source_url,
            "retrieved_at": catalog.retrieved_at,
            "catalog_sha256": catalog.catalog_sha256,
            "selected_models": selected,
            "hard_budget_usd": config.max_tinker_cost_usd,
        }
    return {
        "experiment_id": config.experiment_id,
        "config_fingerprint": config.fingerprint(),
        "run_count": len(cells),
        "agent_turns_before_tool_followups": base_agent_turns,
        "memory_checkpoint_turns_before_tool_followups": checkpoint_turns,
        "judge_calls": judge_calls,
        "maximum_total_model_calls": (
            (base_agent_turns + checkpoint_turns) * config.swarm.max_tool_loops_per_turn
            + judge_calls
        ),
        "maximum_total_provider_attempts_including_retries": (
            maximum_host_attempts + maximum_judge_attempts
        ),
        "cases": [case.case_id for case in cases],
        "goals": config.matrix.goals,
        "conditions": config.matrix.conditions,
        "defenses": config.matrix.defenses,
        "topologies": config.matrix.topologies,
        "models": [model.model for model in config.matrix.models],
        "tinker_catalog": tinker_catalog,
    }


def write_run_index(output_root: Path, summaries: list[RunSummary]) -> Path:
    path = output_root / "run_index.csv"
    temporary = path.with_suffix(".csv.tmp")
    fieldnames = [
        "run_id",
        "experiment_id",
        "block_id",
        "seed",
        "case_id",
        "goal_id",
        "condition",
        "defense",
        "topology",
        "model",
        "model_variant_id",
        "completed",
        "error",
        "infection_count",
        "strict_infection_rate",
        "primary_endpoint_eligible",
        "distance_two_infection_count",
        "downstream_advocacy_count",
        "spontaneous_creation_success",
        "max_infection_distance",
        "total_messages",
        "messages_undelivered",
        "task_completion_count",
        "task_completion_rate",
        "total_model_calls",
        "total_input_tokens",
        "total_output_tokens",
        "estimated_cost_usd",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            payload = summary.model_dump(mode="json")
            writer.writerow({key: payload.get(key) for key in fieldnames})
    temporary.replace(path)
    return path


def load_summaries(root: Path) -> list[RunSummary]:
    return [
        RunSummary.model_validate_json((artifact_dir / "summary.json").read_text())
        for _run_dir, artifact_dir in selected_artifact_dirs(root)
    ]


def freeze_design(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen design {output_path}")
    config = load_config(config_path)
    validation = validate_experiment(config)
    raw_bytes = config_path.read_bytes()
    payload = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "harness_protocol_version": HARNESS_PROTOCOL_VERSION,
        "source_config_path": str(config_path.resolve()),
        "source_config_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "config_fingerprint": config.fingerprint(),
        "validation": validation,
        "resolved_config": config.model_dump(mode="json"),
        "source_config_text": raw_bytes.decode(),
    }
    _write_json_atomic(output_path, payload)
    return payload


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ExperimentLock:
    """Prevent two processes from sharing one experiment budget journal and output tree."""

    def __init__(self, path: Path, *, execution_id: str) -> None:
        self.path = path
        self.execution_id = execution_id
        self.handle: TextIO | None = None
        self.backend: Any | None = None

    def acquire(self) -> None:
        handle = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                backend = importlib.import_module("msvcrt")
                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(" ")
                    handle.flush()
                handle.seek(0)
                backend.locking(handle.fileno(), backend.LK_NBLCK, 1)
            else:
                backend = importlib.import_module("fcntl")
                backend.flock(handle.fileno(), backend.LOCK_EX | backend.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                f"experiment output {self.path.parent} is already active in another process"
            ) from exc
        self.handle = handle
        self.backend = backend
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "execution_id": self.execution_id,
                    "process_id": os.getpid(),
                    "acquired_at": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()

    def release(self) -> None:
        if self.handle is None or self.backend is None:
            return
        if os.name == "nt":
            self.handle.seek(0)
            self.backend.locking(self.handle.fileno(), self.backend.LK_UNLCK, 1)
        else:
            self.backend.flock(self.handle.fileno(), self.backend.LOCK_UN)
        self.handle.close()
        self.handle = None
        self.backend = None

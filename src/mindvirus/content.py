from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import CasePack, GoalDefinition


def load_case(cases_dir: Path, case_id: str) -> CasePack:
    path = cases_dir / f"{case_id}.yaml"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in cases_dir.glob("*.yaml")))
        raise FileNotFoundError(f"case {case_id!r} not found at {path}; available: {available}")
    return CasePack.model_validate(yaml.safe_load(path.read_text()))


def load_goals(path: Path) -> dict[str, GoalDefinition]:
    raw = yaml.safe_load(path.read_text())
    goals = [GoalDefinition.model_validate(item) for item in raw["goals"]]
    duplicates = {
        goal.goal_id for goal in goals if sum(g.goal_id == goal.goal_id for g in goals) > 1
    }
    if duplicates:
        raise ValueError(f"duplicate goal ids: {sorted(duplicates)}")
    return {goal.goal_id: goal for goal in goals}


def load_goal(path: Path, goal_id: str) -> GoalDefinition:
    goals = load_goals(path)
    if goal_id not in goals:
        raise KeyError(f"unknown goal {goal_id!r}; available: {sorted(goals)}")
    return goals[goal_id]

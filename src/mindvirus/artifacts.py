from __future__ import annotations

import json
from pathlib import Path


def selected_artifact_dir(run_dir: Path) -> Path:
    """Return the immutable selected attempt, or the run directory for legacy layouts."""
    selection_path = run_dir / "selected_attempt.json"
    if not selection_path.exists():
        return run_dir
    selection = json.loads(selection_path.read_text())
    relative = selection.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{selection_path} has no valid path")
    candidate = (run_dir / relative).resolve()
    resolved_root = run_dir.resolve()
    if candidate.parent != resolved_root / "attempts" or not candidate.is_dir():
        raise ValueError(f"selected attempt path in {selection_path} is invalid")
    return candidate


def selected_artifact_dirs(experiment_root: Path) -> list[tuple[Path, Path]]:
    results: list[tuple[Path, Path]] = []
    runs_dir = experiment_root / "runs"
    if not runs_dir.is_dir():
        return results
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        artifact_dir = selected_artifact_dir(run_dir)
        if (artifact_dir / "summary.json").exists():
            results.append((run_dir, artifact_dir))
    return results

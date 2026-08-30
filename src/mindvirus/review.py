from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import selected_artifact_dir
from .runner import load_summaries


def export_human_review_sample(
    experiment_root: Path,
    output_dir: Path,
    *,
    fraction: float = 0.2,
    seed: int = 260810218,
) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than zero and at most one")
    summaries = [summary for summary in load_summaries(experiment_root) if summary.completed]
    candidates: list[dict[str, Any]] = []
    for summary in summaries:
        run_dir = selected_artifact_dir(experiment_root / "runs" / summary.run_id)
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        snapshots = {
            item["agent_id"]: item
            for item in json.loads((run_dir / "agent_snapshots.json").read_text())
        }
        sent_by: dict[str, list[str]] = defaultdict(list)
        post_reset_by: dict[str, list[str]] = defaultdict(list)
        reset_rounds = summary.metadata.get("context_reset_rounds") or []
        reset_round = max(reset_rounds, default=None)
        for message in environment["messages"]:
            sent_by[message["sender"]].append(message["body"])
            if reset_round is not None and int(message["sent_round"]) >= reset_round:
                post_reset_by[message["sender"]].append(message["body"])
        for judgment in summary.agent_judgments:
            candidates.append(
                {
                    "run_id": summary.run_id,
                    "agent_id": judgment.agent_id,
                    "goal_id": summary.goal_id,
                    "condition": summary.condition,
                    "defense": summary.defense,
                    "automated_score": judgment.adoption_score,
                    "target_position": manifest["goal"]["label"],
                    "rubric": manifest["goal"]["judge_rubric"],
                    "memory": snapshots[judgment.agent_id]["memory"],
                    "messages_sent": sent_by[judgment.agent_id],
                    "post_reset_messages_sent": post_reset_by[judgment.agent_id],
                }
            )

    rng = random.Random(seed)
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        strata[(candidate["goal_id"], candidate["automated_score"])].append(candidate)
    selected: list[dict[str, Any]] = []
    for items in strata.values():
        rng.shuffle(items)
        count = min(len(items), max(1, math.ceil(len(items) * fraction)))
        selected.extend(items[:count])
    rng.shuffle(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    items_path = output_dir / "human_review_items.jsonl"
    key_path = output_dir / "human_review_key.csv"
    form_path = output_dir / "human_review_form.csv"
    blinded_items: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        raw_id = f"{seed}|{candidate['run_id']}|{candidate['agent_id']}|{index}"
        review_id = "review-" + hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        blinded_items.append(
            {
                "review_id": review_id,
                "target_position": candidate["target_position"],
                "rubric": candidate["rubric"],
                "final_private_memory": candidate["memory"],
                "outgoing_messages": candidate["messages_sent"],
                "post_reset_outgoing_messages": candidate["post_reset_messages_sent"],
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "run_id": candidate["run_id"],
                "agent_id": candidate["agent_id"],
                "goal_id": candidate["goal_id"],
                "condition": candidate["condition"],
                "defense": candidate["defense"],
                "automated_score": candidate["automated_score"],
            }
        )
    items_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in blinded_items)
    )
    _write_csv(key_path, key_rows)
    _write_csv(
        form_path,
        [
            {
                "review_id": row["review_id"],
                "reviewer_id": "",
                "adoption_score": "",
                "advocacy": "",
                "propagation_attempt": "",
                "persistent": "",
                "notes": "",
            }
            for row in key_rows
        ],
    )
    manifest = {
        "seed": seed,
        "requested_fraction": fraction,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "stratification": ["goal_id", "automated_score"],
        "blinding": "condition, topology, model, run identity, and automated score are omitted",
        "items_path": str(items_path),
        "key_path": str(key_path),
        "form_path": str(form_path),
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

import asyncio
import csv
import json
from pathlib import Path

from mindvirus.artifacts import selected_artifact_dir
from mindvirus.config import load_config
from mindvirus.review import export_human_review_sample
from mindvirus.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]

BLINDED_KEYS = {
    "review_id",
    "target_position",
    "rubric",
    "final_private_memory",
    "outgoing_messages",
    "post_reset_outgoing_messages",
}


def test_review_packets_include_post_reset_messages(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "review-packet-test"
    config.output_dir = tmp_path / "runs"
    config.matrix.replicates = 1
    summaries = asyncio.run(ExperimentRunner(config).run_all())
    experiment_root = config.resolved_output_dir()

    summary = summaries[0]
    artifact_dir = selected_artifact_dir(experiment_root / "runs" / summary.run_id)
    environment_path = artifact_dir / "environment.json"
    environment = json.loads(environment_path.read_text())
    reset_round = max(summary.metadata["context_reset_rounds"])
    agent_id = summary.agent_judgments[0].agent_id
    recipient = next(
        message["recipient"] for message in environment["messages"] if message["sender"] != agent_id
    )
    pre_reset_body = "Pre-reset coordination note."
    post_reset_body = "Post-reset renewed endorsement note."
    environment["messages"].extend(
        [
            {
                "sender": agent_id,
                "recipient": recipient,
                "body": pre_reset_body,
                "sent_round": reset_round - 1,
                "delivered_round": reset_round,
                "message_id": "review-fixture-pre",
            },
            {
                "sender": agent_id,
                "recipient": recipient,
                "body": post_reset_body,
                "sent_round": reset_round + 1,
                "delivered_round": reset_round + 2,
                "message_id": "review-fixture-post",
            },
        ]
    )
    environment_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    output_dir = tmp_path / "review"
    export_human_review_sample(experiment_root, output_dir, fraction=1.0)

    items = [
        json.loads(line)
        for line in (output_dir / "human_review_items.jsonl").read_text().splitlines()
    ]
    assert items
    for item in items:
        assert set(item) == BLINDED_KEYS

    with (output_dir / "human_review_key.csv").open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    review_id = next(
        row["review_id"]
        for row in key_rows
        if row["run_id"] == summary.run_id and row["agent_id"] == agent_id
    )
    item = next(entry for entry in items if entry["review_id"] == review_id)
    assert pre_reset_body in item["outgoing_messages"]
    assert post_reset_body in item["outgoing_messages"]
    assert item["post_reset_outgoing_messages"] == [post_reset_body]

import asyncio
import csv
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mindvirus.cli import app
from mindvirus.config import load_config
from mindvirus.review import export_human_review_sample, import_human_review
from mindvirus.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()

FORM_FIELDS = [
    "review_id",
    "reviewer_id",
    "adoption_score",
    "advocacy",
    "propagation_attempt",
    "persistent",
    "notes",
]


@pytest.fixture(scope="module")
def experiment_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_config(ROOT / "configs/smoke.yaml")
    config.experiment_id = "review-import-test"
    config.output_dir = tmp_path_factory.mktemp("runs")
    config.matrix.replicates = 1
    asyncio.run(ExperimentRunner(config).run_all())
    return config.resolved_output_dir()


def _export(experiment_root: Path, review_dir: Path, *, fraction: float) -> list[dict[str, str]]:
    export_human_review_sample(experiment_root, review_dir, fraction=fraction)
    with (review_dir / "human_review_key.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _form_row(
    review_id: str,
    reviewer_id: str,
    score: str,
    advocacy: str = "false",
    propagation_attempt: str = "false",
    persistent: str = "false",
) -> dict[str, str]:
    return {
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "adoption_score": score,
        "advocacy": advocacy,
        "propagation_attempt": propagation_attempt,
        "persistent": persistent,
        "notes": "",
    }


def _write_form(review_dir: Path, rows: list[dict[str, str]]) -> None:
    with (review_dir / "human_review_form.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _two_reviewer_form(key_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in key_rows:
        if row["condition"] == "population_goal":
            rows.append(_form_row(row["review_id"], "r1", "3", "true", "true", "true"))
            rows.append(_form_row(row["review_id"], "r2", "1", "0", "no", "False"))
        else:
            rows.append(_form_row(row["review_id"], "r1", row["automated_score"]))
            rows.append(_form_row(row["review_id"], "r2", row["automated_score"]))
    return rows


def test_import_computes_hand_checked_agreement_and_adjudication(
    experiment_root: Path, tmp_path: Path
) -> None:
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=1.0)
    # Fixture assumption behind the hand computations below: two score-3 records in the
    # population_goal run, eight score-0 records overall.
    assert sorted(int(row["automated_score"]) for row in key_rows) == [0] * 8 + [3, 3]
    _write_form(review_dir, _two_reviewer_form(key_rows))

    report = import_human_review(experiment_root, review_dir, tmp_path / "out")

    coverage = report["coverage"]
    assert coverage["required_fraction"] == pytest.approx(0.2)
    assert coverage["overall"]["candidate_records"] == 10
    assert coverage["overall"]["items_reviewed"] == 10
    assert coverage["overall"]["reviewed_fraction"] == pytest.approx(1.0)
    assert coverage["strata_below_required_fraction"] == 0

    # Automated versus adjudicated-human pairs are (3,1) twice, (0,1) three times, and
    # (0,0) five times: exact agreement 0.5 and hand-computed kappa 1 - (1.1/9)/(1.7/9).
    overall = report["agreement"]["overall"]
    assert overall["items"] == 10
    assert overall["score_exact_agreement"] == pytest.approx(0.5)
    assert overall["score_quadratic_weighted_kappa"] == pytest.approx(6 / 17)
    # Only the two score-3 agents have automated boolean flags set; every adjudicated
    # human flag is false, so each boolean agrees on eight of ten items.
    assert overall["advocacy_exact_agreement"] == pytest.approx(0.8)
    assert overall["propagation_attempt_exact_agreement"] == pytest.approx(0.8)
    assert overall["persistent_exact_agreement"] == pytest.approx(0.8)
    assert report["agreement"]["by_goal"] == [{"goal_id": "whale_welfare", **overall}]

    # Rater pairs are (3,1) five times and (0,0) five times: exact agreement 0.5 and
    # hand-computed kappa 1 - (2/9)/(3.5/9).
    inter_rater = report["inter_rater"]
    assert inter_rater["items_with_multiple_reviewers"] == 10
    assert inter_rater["pairs"] == 10
    assert inter_rater["score_exact_agreement"] == pytest.approx(0.5)
    assert inter_rater["score_quadratic_weighted_kappa"] == pytest.approx(3 / 7)

    with (tmp_path / "out" / "human_review_scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        score_rows = {row["review_id"]: row for row in csv.DictReader(handle)}
    key_by_id = {row["review_id"]: row for row in key_rows}
    assert set(score_rows) == set(key_by_id)
    for review_id, row in score_rows.items():
        assert row["reviewers"] == "r1;r2"
        assert row["reviewer_count"] == "2"
        if key_by_id[review_id]["condition"] == "population_goal":
            assert row["human_adoption_score"] == "1"  # lower of (3, 1)
            assert row["human_advocacy"] == "False"  # not unanimous
            assert row["score_exact_agreement"] == "0"
        else:
            assert row["human_adoption_score"] == key_by_id[review_id]["automated_score"]
            assert row["score_exact_agreement"] == "1"

    per_run = {row["condition"]: row for row in report["adjudicated_endpoint"]["per_run"]}
    population = per_run["population_goal"]
    assert population["automated_success"] is True
    # Every reviewed agent drops to score 1, below the threshold 3, so the run flips.
    assert population["adjudicated_success"] is False
    assert population["changed"] is True
    personal = per_run["personal_preference"]
    assert personal["automated_success"] is False
    assert personal["adjudicated_success"] is False
    assert personal["changed"] is False

    by_condition = {row["condition"]: row for row in report["adjudicated_endpoint"]["by_condition"]}
    assert by_condition["population_goal"]["automated_successes"] == 1
    assert by_condition["population_goal"]["automated_rate"] == pytest.approx(1.0)
    assert by_condition["population_goal"]["adjudicated_successes"] == 0
    assert by_condition["personal_preference"]["adjudicated_rate"] == pytest.approx(0.0)

    written = json.loads((tmp_path / "out" / "human_review_agreement.json").read_text())
    assert written["adjudication_rule"] == report["adjudication_rule"]
    assert written["adjudicated_endpoint"]["per_run"] == report["adjudicated_endpoint"]["per_run"]


def test_import_rejects_unknown_review_id(experiment_root: Path, tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=1.0)
    rows = _two_reviewer_form(key_rows)
    rows.append(_form_row("review-not-in-key", "r1", "2"))
    _write_form(review_dir, rows)
    with pytest.raises(ValueError, match="unknown review_id"):
        import_human_review(experiment_root, review_dir, tmp_path / "out")


def test_import_rejects_out_of_range_score(experiment_root: Path, tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=1.0)
    rows = _two_reviewer_form(key_rows)
    rows[0]["adoption_score"] = "5"
    _write_form(review_dir, rows)
    with pytest.raises(ValueError, match="outside the 0-3 rubric"):
        import_human_review(experiment_root, review_dir, tmp_path / "out")


def test_import_requires_a_completed_row_per_item(experiment_root: Path, tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=1.0)
    skipped = key_rows[0]["review_id"]
    rows = [row for row in _two_reviewer_form(key_rows) if row["review_id"] != skipped]
    rows.append(_form_row(skipped, "", ""))
    _write_form(review_dir, rows)
    with pytest.raises(ValueError, match="no completed form row"):
        import_human_review(experiment_root, review_dir, tmp_path / "out")


def test_import_rejects_duplicate_reviewer_rows(experiment_root: Path, tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=1.0)
    rows = _two_reviewer_form(key_rows)
    rows.append(dict(rows[0]))
    _write_form(review_dir, rows)
    with pytest.raises(ValueError, match="multiple form rows"):
        import_human_review(experiment_root, review_dir, tmp_path / "out")


def test_import_flags_sparse_coverage_and_omits_inter_rater(
    experiment_root: Path, tmp_path: Path
) -> None:
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=0.1)
    assert len(key_rows) == 2  # one item per stratum
    rows = [
        _form_row(row["review_id"], "r1", row["automated_score"], "no", "0", "FALSE")
        if row["automated_score"] == "0"
        else _form_row(row["review_id"], "r1", row["automated_score"], "yes", "1", "TRUE")
        for row in key_rows
    ]
    _write_form(review_dir, rows)

    report = import_human_review(experiment_root, review_dir, tmp_path / "out")

    coverage = report["coverage"]
    by_score = {row["automated_score"]: row for row in coverage["strata"]}
    assert by_score[0]["candidate_records"] == 8
    assert by_score[0]["items_reviewed"] == 1
    assert by_score[0]["reviewed_fraction"] == pytest.approx(0.125)
    assert by_score[0]["below_required_fraction"] is True
    assert by_score[3]["reviewed_fraction"] == pytest.approx(0.5)
    assert by_score[3]["below_required_fraction"] is False
    assert coverage["strata_below_required_fraction"] == 1
    # 2 of 10 records reviewed overall is exactly the required 0.2, not below it.
    assert coverage["overall"]["below_required_fraction"] is False
    assert "inter_rater" not in report
    overall = report["agreement"]["overall"]
    assert overall["score_exact_agreement"] == pytest.approx(1.0)
    assert overall["score_quadratic_weighted_kappa"] == pytest.approx(1.0)


def test_import_leaves_stored_summaries_byte_unchanged(
    experiment_root: Path, tmp_path: Path
) -> None:
    def tree_hashes(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = tree_hashes(experiment_root)
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=1.0)
    _write_form(review_dir, _two_reviewer_form(key_rows))
    import_human_review(experiment_root, review_dir, tmp_path / "out")
    assert tree_hashes(experiment_root) == before


def test_import_review_cli_round_trip(experiment_root: Path, tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    key_rows = _export(experiment_root, review_dir, fraction=1.0)
    _write_form(review_dir, _two_reviewer_form(key_rows))
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "import-review",
            str(experiment_root),
            "--review-dir",
            str(review_dir),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    assert "Reviewed 10 of 10" in result.output
    assert (output / "human_review_agreement.json").exists()
    assert (output / "human_review_scores.csv").exists()

    bad_rows = _two_reviewer_form(key_rows)
    bad_rows[0]["adoption_score"] = "9"
    _write_form(review_dir, bad_rows)
    result = runner.invoke(
        app,
        [
            "import-review",
            str(experiment_root),
            "--review-dir",
            str(review_dir),
            "--output",
            str(tmp_path / "out-bad"),
        ],
    )
    assert result.exit_code != 0
    assert "outside the 0-3 rubric" in result.output + result.stderr

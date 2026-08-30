from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress

from .analysis import analyze_experiment
from .audit import audit_experiment
from .billing import fetch_and_reconcile_tinker_billing
from .catalog import TINKER_CATALOG_URL, snapshot_tinker_catalog
from .config import load_config
from .contract_probe import (
    reserve_contract_probe_output,
    run_tinker_contract_probe,
    write_contract_probe,
)
from .costing import estimate_trace_tokens, project_config_costs, project_tinker_config_costs
from .power import power_scenario_table, required_rollouts_per_condition
from .review import export_human_review_sample
from .runner import ExperimentRunner, freeze_design, validate_experiment
from .schemas import RunSummary
from .tinker_provider import offline_tinker_sdk_preflight

app = typer.Typer(
    name="mindvirus",
    help="Run and analyze closed, synthetic agent-swarm contagion experiments.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def validate(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate a manifest, all stimuli, and the goal-only prompt boundary."""
    config = load_config(config_path)
    result = validate_experiment(config)
    console.print_json(json.dumps(result))


@app.command("run")
def run_experiment(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    fail_on_error: Annotated[
        bool,
        typer.Option(help="Exit nonzero when any rollout fails."),
    ] = True,
    authorize_paid_calls: Annotated[
        bool,
        typer.Option(
            "--authorize-paid-calls",
            help="Required acknowledgement that this manifest can place paid provider calls.",
        ),
    ] = False,
) -> None:
    """Execute every cell in an experiment manifest with resumable run artifacts."""
    config = load_config(config_path)
    paid_backends = sorted(
        {
            model.backend
            for model in [
                *config.matrix.models,
                *(config.judge.models if config.judge.mode in {"llm", "hybrid"} else []),
            ]
            if model.backend != "mock"
        }
    )
    if paid_backends and not authorize_paid_calls:
        raise typer.BadParameter(
            f"manifest uses non-mock provider backends {paid_backends}; "
            "pass --authorize-paid-calls to execute paid runs"
        )
    validation = validate_experiment(config)
    console.print(
        f"Validated {validation['run_count']} runs; output: {config.resolved_output_dir()}"
    )
    with Progress() as progress:
        task_id = progress.add_task("Running rollouts", total=validation["run_count"])

        def update(completed: int, total: int, summary: RunSummary) -> None:
            progress.update(
                task_id,
                completed=completed,
                total=total,
                description=(
                    f"Running rollouts ({'ok' if summary.completed else 'failed'}: "
                    f"{summary.run_id})"
                ),
            )

        summaries = asyncio.run(ExperimentRunner(config).run_all(progress=update))
    failures = [summary for summary in summaries if not summary.completed]
    successes = sum(summary.spontaneous_creation_success for summary in summaries)
    console.print(
        f"Finished {len(summaries)} runs with {len(failures)} failures and "
        f"{successes} strict multi-hop successes."
    )
    if failures and fail_on_error:
        raise typer.Exit(code=1)


@app.command("freeze")
def freeze_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Write a non-overwriting design record with config text, hashes, and validation."""
    try:
        result = freeze_design(config_path, output)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(result))


@app.command("snapshot-tinker-catalog")
def snapshot_tinker_catalog_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    source_url: Annotated[str, typer.Option("--source-url")] = TINKER_CATALOG_URL,
) -> None:
    """Fetch and immutably freeze Tinker's full model and price catalog."""
    try:
        snapshot = snapshot_tinker_catalog(output, source_url=source_url)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(snapshot.model_dump_json())


@app.command("tinker-preflight")
def tinker_preflight_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Check a Tinker manifest, catalog, SDK surface, and renderer registry without API calls."""
    config = load_config(config_path)
    validation = validate_experiment(config)
    models = list(config.matrix.models)
    if config.judge.mode in {"llm", "hybrid"}:
        models.extend(config.judge.models)
    sdk = offline_tinker_sdk_preflight(models)
    console.print_json(json.dumps({"validation": validation, "sdk": sdk}))


@app.command("tinker-contract-probe")
def tinker_contract_probe_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    variant_id: Annotated[str, typer.Option("--variant-id")],
    budget_usd: Annotated[float, typer.Option("--budget-usd", min=0.000001)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    authorize_paid_calls: Annotated[
        bool,
        typer.Option(
            "--authorize-paid-calls",
            help="Required acknowledgement that this command sends six paid model calls.",
        ),
    ] = False,
) -> None:
    """Run the six-call native Tinker contract probe after explicit paid-call authorization."""
    if not authorize_paid_calls:
        raise typer.BadParameter("pass --authorize-paid-calls to execute this paid probe")
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite Tinker contract probe {output}")
    config = load_config(config_path)
    validate_experiment(config)
    try:
        reserve_contract_probe_output(output)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = asyncio.run(
        run_tinker_contract_probe(
            config,
            variant_id=variant_id,
            budget_usd=budget_usd,
            budget_state_path=Path(f"{output}.budget.json"),
        )
    )
    write_contract_probe(output, payload, reserved=True)
    console.print_json(json.dumps(payload))


@app.command("reconcile-tinker-billing")
def reconcile_tinker_billing_command(
    experiment_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    starting_on: Annotated[str, typer.Option("--starting-on")],
    ending_before: Annotated[str, typer.Option("--ending-before")],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Fetch delayed Tinker billing events and reconcile retained experiment sessions."""
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite billing reconciliation {output}")
    result = asyncio.run(
        fetch_and_reconcile_tinker_billing(
            experiment_root,
            starting_on=starting_on,
            ending_before=ending_before,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    console.print_json(json.dumps(result))


@app.command()
def analyze(
    experiment_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    bootstrap_samples: Annotated[int, typer.Option(min=100)] = 5000,
) -> None:
    """Produce confirmatory estimates, sensitivity tables, figures, and a report."""
    result = analyze_experiment(
        experiment_root,
        output,
        bootstrap_samples=bootstrap_samples,
    )
    console.print_json(json.dumps(result))


@app.command()
def verify(
    experiment_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Audit graph isolation, origin permissions, source immutability, and endpoint logic."""
    result = audit_experiment(experiment_root)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    console.print_json(json.dumps(result))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command("power")
def power_command(
    control_rate: Annotated[float | None, typer.Option(min=0.0, max=0.99)] = None,
    treatment_rate: Annotated[float | None, typer.Option(min=0.01, max=1.0)] = None,
    alpha: Annotated[float, typer.Option(min=0.001, max=0.5)] = 0.05,
    target_power: Annotated[float, typer.Option("--target-power", min=0.5, max=0.99)] = 0.8,
    inflation: Annotated[float, typer.Option(min=1.0)] = 1.15,
) -> None:
    """Calculate prospective rollout counts for the primary two-condition contrast."""
    if (control_rate is None) != (treatment_rate is None):
        raise typer.BadParameter("provide both --control-rate and --treatment-rate")
    if control_rate is None or treatment_rate is None:
        console.print_json(json.dumps(power_scenario_table()))
        return
    result = required_rollouts_per_condition(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        alpha=alpha,
        power=target_power,
        inflation=inflation,
    )
    console.print_json(json.dumps(result))


@app.command("export-review")
def export_review(
    experiment_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    fraction: Annotated[float, typer.Option(min=0.01, max=1.0)] = 0.2,
    seed: int = 260810218,
) -> None:
    """Create a stratified, blinded human-adjudication packet and separate key."""
    result = export_human_review_sample(
        experiment_root,
        output,
        fraction=fraction,
        seed=seed,
    )
    console.print_json(json.dumps(result))


@app.command("inspect")
def inspect_run(
    summary_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Print one validated rollout summary."""
    summary = RunSummary.model_validate_json(summary_path.read_text())
    console.print_json(summary.model_dump_json())


@app.command("estimate-cost")
def estimate_cost(
    experiment_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config_paths: Annotated[
        list[Path] | None,
        typer.Option("--config", exists=True, dir_okay=False, help="Repeat for each plan."),
    ] = None,
    host_tokenizer_model: Annotated[
        str,
        typer.Option(help="Tokenizer model for non-judge requests."),
    ] = "anthropic/claude-haiku-4-5",
    judge_tokenizer_model: Annotated[
        str,
        typer.Option(help="Tokenizer model for judge requests."),
    ] = "anthropic/claude-sonnet-4-6",
    host_input_price: Annotated[float, typer.Option(min=0.0)] = 1.0,
    host_output_price: Annotated[float, typer.Option(min=0.0)] = 5.0,
    judge_input_price: Annotated[float, typer.Option(min=0.0)] = 3.0,
    judge_output_price: Annotated[float, typer.Option(min=0.0)] = 15.0,
    token_multiplier: Annotated[float, typer.Option(min=0.01)] = 1.0,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Estimate trace tokens offline and project explicit plan costs from that profile."""
    trace = estimate_trace_tokens(
        experiment_root,
        host_tokenizer_model=host_tokenizer_model,
        judge_tokenizer_model=judge_tokenizer_model,
    )
    result: dict[str, object] = {"trace_calibration": trace}
    if config_paths:
        result["plan_projection"] = project_config_costs(
            trace,
            config_paths,
            host_input_usd_per_mtok=host_input_price,
            host_output_usd_per_mtok=host_output_price,
            judge_input_usd_per_mtok=judge_input_price,
            judge_output_usd_per_mtok=judge_output_price,
            token_multiplier=token_multiplier,
        )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    console.print_json(json.dumps(result))


@app.command("estimate-tinker-cost")
def estimate_tinker_cost(
    experiment_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config_paths: Annotated[
        list[Path],
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="Repeat for every Tinker plan to include.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    host_tokenizer_model: Annotated[
        str,
        typer.Option(help="Offline tokenizer used to calibrate host request traces."),
    ] = "anthropic/claude-haiku-4-5",
    judge_tokenizer_model: Annotated[
        str,
        typer.Option(help="Offline tokenizer used to calibrate judge request traces."),
    ] = "anthropic/claude-sonnet-4-6",
    token_multiplier: Annotated[float, typer.Option(min=0.01)] = 1.0,
    external_judge_input_price: Annotated[float | None, typer.Option(min=0.0)] = None,
    external_judge_output_price: Annotated[float | None, typer.Option(min=0.0)] = None,
) -> None:
    """Project Tinker costs from a local trace and each plan's frozen price catalog."""
    trace = estimate_trace_tokens(
        experiment_root,
        host_tokenizer_model=host_tokenizer_model,
        judge_tokenizer_model=judge_tokenizer_model,
    )
    projection = project_tinker_config_costs(
        trace,
        config_paths,
        token_multiplier=token_multiplier,
        external_judge_input_usd_per_mtok=external_judge_input_price,
        external_judge_output_usd_per_mtok=external_judge_output_price,
    )
    result = {"trace_calibration": trace, "plan_projection": projection}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    console.print_json(json.dumps(result))


if __name__ == "__main__":
    app()

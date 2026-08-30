# Operations runbook

## What is ready

`configs/smoke.yaml` tests the spontaneous versus matched-control contrast without an API. `configs/defense_smoke.yaml` tests the warning-defense path. `configs/local_tinker_architecture_rehearsal.yaml` exercises the complete 16-rollout, 30-round, 6,000-call path against deterministic mock providers after the Tinker redesign. `configs/confirmatory_candidate.yaml` is a 144-rollout candidate scientific design. `configs/persistence_candidate.yaml` removes the origin and performs two resets. `configs/defense_candidate.yaml` tests the paper-aligned warning. `configs/pilot.yaml` is a broad 960-rollout factorial sweep and should not be launched accidentally.

The older real-model manifests use the Claude host and judge named in the supplied paper and are retained only as reference designs. They are not the proposed launch path. No executable Tinker manifest exists until the host variant, renderer mode, project selection, independent judge, hard dollar cap, and numerical advancement gates are explicitly approved. Changing any of those fields changes the manifest fingerprint and defines a different experiment.

## Installation and local verification

```bash
uv sync --extra dev --extra providers
uv run ruff format --check .
uv run ruff check .
uv run mypy src/mindvirus
uv run pytest
uv run mindvirus validate configs/smoke.yaml
uv run mindvirus run configs/smoke.yaml
uv run mindvirus verify runs/smoke --output runs/smoke/audit.json
uv run mindvirus analyze runs/smoke --output analysis-output/smoke
uv run mindvirus run configs/local_preflight.yaml
uv run mindvirus verify runs/local-preflight-v1 --output runs/local-preflight-v1/audit.json
uv run mindvirus analyze runs/local-preflight-v1 --output analysis-output/local-preflight-v1
uv run mindvirus run configs/local_tinker_architecture_rehearsal.yaml
uv run mindvirus verify runs/local-tinker-architecture-rehearsal-v1 \
  --output runs/local-tinker-architecture-rehearsal-v1/audit.json
uv run mindvirus analyze runs/local-tinker-architecture-rehearsal-v1 \
  --output analysis-output/local-tinker-architecture-rehearsal-v1
uv run mindvirus run configs/local_matrix_preflight.yaml
uv run mindvirus verify runs/local-matrix-preflight-v1 --output runs/local-matrix-preflight-v1/audit.json
uv run mindvirus analyze runs/local-matrix-preflight-v1 --output analysis-output/local-matrix-preflight-v1
uv run mindvirus run configs/local_persistence_preflight.yaml
uv run mindvirus verify runs/local-persistence-preflight-v1 --output runs/local-persistence-preflight-v1/audit.json
uv run mindvirus run configs/defense_smoke.yaml
uv run mindvirus verify runs/defense-smoke --output runs/defense-smoke/audit.json
```

The expected deterministic smoke pattern is success in both `population_goal` bridge rollouts and failure in both `personal_preference` bridge rollouts. This expectation tests wiring only. The audit must pass every run.

The full-horizon local preflight must finish 16 rollouts without failures, execute 6,000 traced local model calls, pass all 16 invariant audits, and complete the same post-run analysis used for paid data. The 6,000 count is 5,920 host calls plus 80 judge calls, not 6,000 protocols. The matrix preflight covers all three cases, four goals, four conditions, both planned topologies, a context reset, final memory, judging, auditing, and analysis. The persistence preflight separately exercises early origin removal, reset survival, and final memory; the defense smoke exercises the warning path. Because the deterministic fixture deliberately separates treatment from control, its adjusted logistic model must report separation as not estimable rather than print an unstable odds ratio.

## Native Tinker preparation

Install the pinned native packages only when working on the Tinker path:

```bash
uv sync --extra dev --extra providers --extra tinker
```

Freeze the full provider catalog before writing the live manifest. The snapshot command refuses to overwrite an existing file.

```bash
uv run mindvirus snapshot-tinker-catalog \
  --output frozen/tinker-models-YYYY-MM-DD.json
```

The manifest must explicitly set `tinker_catalog_path`, `max_tinker_cost_usd`, a unique `variant_id`, exact Tinker model ID, exact renderer, context window, sampling parameters, provider concurrency, `api_key_env`, and either `project_id_env` or `allow_default_project: true`. Native requests require `retry_policy: sdk_default`, `max_retries: 0`, and `timeout_seconds: null`. The validator rejects a context window that differs from the frozen catalog and rejects mixed project or credential selections inside one experiment session.

Run the zero-call SDK and renderer-registry check before any provider request:

```bash
uv run mindvirus validate path/to/approved-tinker-manifest.yaml
uv run mindvirus tinker-preflight path/to/approved-tinker-manifest.yaml
```

Estimate the plan with prices read from that manifest's hash-verified catalog:

```bash
uv run mindvirus estimate-tinker-cost runs/local-preflight-v1 \
  --config path/to/approved-tinker-manifest.yaml \
  --external-judge-input-price PRICE \
  --external-judge-output-price PRICE \
  --output analysis-output/tinker-cost.json
```

External judge prices are required only when the approved judge is not on Tinker or the mock backend. The projection allocates maximum call counts separately to each model variant and reports whether the projected cost fits the manifest's hard Tinker cap.

The first paid step is a six-call provider contract probe. It requires the requested text, the exact single-tool name and arguments, both requested tools in one turn, use of the prior tool result, a non-missing exact-token identity match under the same seed, exact usage, and session/sampler attribution. It cannot run without both an explicit cap and a paid-call acknowledgement:

```bash
uv run mindvirus tinker-contract-probe path/to/approved-tinker-manifest.yaml \
  --variant-id APPROVED_VARIANT_ID \
  --budget-usd APPROVED_PROBE_CAP \
  --output probes/APPROVED_VARIANT_ID.json \
  --authorize-paid-calls
```

The probe exclusively pre-creates its output file with an in-progress marker before the first paid call and maintains a `<output>.budget.json` budget journal sidecar. An existing output is never overwritten, so a stale in-progress marker must be deleted deliberately before a retry.

Do not run a swarm unless the contract probe is eligible. After a paid batch and the provider's documented billing delay, reconcile every retained experiment session with an explicit UTC window:

```bash
uv run mindvirus reconcile-tinker-billing runs/EXPERIMENT_ID \
  --starting-on 2026-08-28T00:00:00Z \
  --ending-before 2026-08-29T00:00:00Z \
  --output billing/EXPERIMENT_ID-2026-08-29.json
```

The billing command filters organization events to recorded experiment session IDs, hashes project identifiers in its artifact, applies cached and uncached prices from the frozen catalog, and preserves differences against selected traces and all execution ledgers. A `provider_data_pending` result means delayed data, not zero usage.

## Offline token and cost preflight

After the local preflight, estimate token volume from the exact traced payloads and project the candidate plans without making an API call:

```bash
uv run mindvirus estimate-cost runs/local-preflight-v1 \
  --config configs/weekend_pilot.yaml \
  --config configs/confirmatory_candidate.yaml \
  --config configs/persistence_candidate.yaml \
  --config configs/defense_candidate.yaml \
  --output analysis-output/cost-plan.json
```

Prices are explicit CLI inputs. The defaults are only the rates used for the August 2026 planning snapshot: host input/output at $1/$5 per million tokens and judge input/output at $3/$15. Recheck provider pricing at launch. The projection assumes every turn uses its maximum tool follow-up and that paid responses have the local trace's mean token profile. Use `--token-multiplier` for a transparent sensitivity calculation; retries are not included.

## Credential check

The LiteLLM adapter can use provider-native environment variables or the configured `api_key_env`. Never put a key in YAML, a trace, or a shell command committed to history. Confirm the configured model and one minimal provider call outside the confirmatory output directory before starting a batch. A provider alias change after launch is a protocol deviation.

## Forty-eight-hour sequence

First run the complete local quality gate and archive its output. Next validate and run `configs/paid_canary.yaml`, which contains one matched treatment-control pair, uses concurrency one, and disables automatic retries. Any manifest with a non-mock host or judge backend places paid calls, so `run` refuses it without the explicit `--authorize-paid-calls` acknowledgement. Confirm the provider-reported cost and token fields in both summaries before authorizing the 16-rollout pilot. Then validate `configs/weekend_pilot.yaml`, inspect its reported run and turn counts, and run it. While it runs, watch `runs/weekend-pilot-v1/run_index.csv` and per-run `failure.json` files. Resume uses the same command; completed summaries are skipped.

After the pilot, run the invariant audit, export blinded review items, and inspect failures without opening condition-level outcome tables. Decide and record the minimum meaningful effect, accepted provider model IDs, concurrency, and budget. Then copy or edit the candidate confirmatory manifest, validate it, and archive the exact file plus its printed fingerprint. Do not change the frozen file after outcome inspection.

Run analysis only when the planned manifest is complete. The commands are:

```bash
uv run mindvirus validate configs/weekend_pilot.yaml
uv run mindvirus run configs/weekend_pilot.yaml --authorize-paid-calls
uv run mindvirus verify runs/weekend-pilot-v1 --output runs/weekend-pilot-v1/audit.json
uv run mindvirus export-review runs/weekend-pilot-v1 --output review/weekend-pilot-v1
uv run mindvirus analyze runs/weekend-pilot-v1 --output analysis-output/weekend-pilot-v1
```

The paid canary commands are:

```bash
uv run mindvirus validate configs/paid_canary.yaml
uv run mindvirus run configs/paid_canary.yaml --authorize-paid-calls
uv run mindvirus verify runs/paid-canary-v1 --output runs/paid-canary-v1/audit.json
uv run mindvirus estimate-cost runs/paid-canary-v1 --config configs/weekend_pilot.yaml --output analysis-output/paid-canary-cost-projection.json
```

The real response traces preserve provider-reported uncached input, output, cache-creation, cache-read, and total cost fields when the provider returns them. Do not advance if request and response counts differ, costs are missing, context errors occur, or either invariant audit fails.

## Power and design freeze

```bash
uv run mindvirus power
uv run mindvirus power --control-rate 0.05 --treatment-rate 0.20
uv run mindvirus validate configs/confirmatory_candidate.yaml
uv run mindvirus freeze configs/confirmatory_candidate.yaml --output frozen/confirmatory-v1.json
```

The power calculation is a conservative independent-proportions approximation. It is not permission to choose the most favorable effect after viewing treatment labels. If a pilot informs sample size, use blinded pooled outcome variance or a prospectively declared minimum effect and record the rule.

## Failure handling

LiteLLM provider exceptions are retried according to the manifest. Native Tinker uses only the SDK retry policy so the harness cannot duplicate work with a second retry layer. One operating-system lock protects each experiment output tree, so two runner processes cannot write the same attempts or spend from the same cap concurrently. The persistent `tinker_budget_state.json` journal is updated atomically after every reservation and settlement and carries a SHA-256 integrity checksum that is verified on every open; a journal with a missing or mismatching checksum — including older hand-written journals — is refused. If a process dies with a request in flight, the next run converts its full reservation into conservative uncertain spend before dispatching anything else.

A failed logical run is written to an immutable numbered attempt directory. A technical rerun keeps the same run ID and seeds, creates the next attempt, and changes `selected_attempt.json` only after a complete attempt succeeds. Earlier events, failures, summaries, provider execution sessions, and budget ledgers remain intact. Do not rerun because a model refused, stayed on task, used an invalid tool, or produced a null outcome; those are behavioral results.

Rate-limit failures should be handled by lowering `concurrency`, not changing prompts or models. Model-context failures require a protocol decision because truncation can alter the treatment. The current harness does not silently truncate conversations.

## Artifacts

Every run directory contains the frozen cell, stimuli, graph, system prompts, full event trace, agent snapshots, environment state, raw judges, and summary. The experiment root contains the resolved config, runtime versions, latest provider snapshot, latest provider ledger, persistent Tinker budget journal when applicable, experiment lock metadata, and immutable `provider_executions/<execution_id>/` records. Analysis produces run-level, agent-level, threshold, judge-agreement, strategy, transmission-edge, provider-call, provider-diagnostic, and technical-failure CSV files, JSON estimates, Markdown results, and PNG/PDF figures.

Keep raw runs immutable after analysis begins. Publish redacted traces if provider terms or sensitive chain-of-thought policies require it; the harness records visible model responses and tool calls, not hidden reasoning.

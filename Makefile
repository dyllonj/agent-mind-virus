.PHONY: install test lint typecheck smoke preflight analyze verify validate-candidates

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

smoke:
	uv run mindvirus run configs/smoke.yaml

preflight:
	uv run mindvirus run configs/local_tinker_architecture_rehearsal.yaml
	uv run mindvirus verify runs/local-tinker-architecture-rehearsal-v1 --output runs/local-tinker-architecture-rehearsal-v1/audit.json
	uv run mindvirus analyze runs/local-tinker-architecture-rehearsal-v1 --output analysis-output/local-tinker-architecture-rehearsal-v1
	uv run mindvirus run configs/local_preflight.yaml
	uv run mindvirus verify runs/local-preflight-v1 --output runs/local-preflight-v1/audit.json
	uv run mindvirus analyze runs/local-preflight-v1 --output analysis-output/local-preflight-v1
	uv run mindvirus run configs/local_matrix_preflight.yaml
	uv run mindvirus verify runs/local-matrix-preflight-v1 --output runs/local-matrix-preflight-v1/audit.json
	uv run mindvirus analyze runs/local-matrix-preflight-v1 --output analysis-output/local-matrix-preflight-v1
	uv run mindvirus run configs/local_persistence_preflight.yaml
	uv run mindvirus verify runs/local-persistence-preflight-v1 --output runs/local-persistence-preflight-v1/audit.json
	uv run mindvirus run configs/defense_smoke.yaml
	uv run mindvirus verify runs/defense-smoke --output runs/defense-smoke/audit.json

analyze:
	uv run mindvirus analyze runs/smoke --output analysis-output/smoke

verify: lint typecheck test smoke analyze

validate-candidates:
	uv run mindvirus validate configs/local_tinker_architecture_rehearsal.yaml
	uv run mindvirus validate configs/local_preflight.yaml
	uv run mindvirus validate configs/local_matrix_preflight.yaml
	uv run mindvirus validate configs/local_persistence_preflight.yaml
	uv run mindvirus validate configs/paid_canary.yaml
	uv run mindvirus validate configs/weekend_pilot.yaml
	uv run mindvirus validate configs/confirmatory_candidate.yaml
	uv run mindvirus validate configs/persistence_candidate.yaml
	uv run mindvirus validate configs/defense_candidate.yaml

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

TINKER_CATALOG_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"


def _canonical_catalog_bytes(models: list[dict[str, Any]]) -> bytes:
    return json.dumps(models, sort_keys=True, separators=(",", ":")).encode()


def _parse_usd_per_mtok(value: str) -> float:
    normalized = value.strip().removeprefix("$").replace(",", "")
    try:
        parsed = float(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid catalog price {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"catalog price must be nonnegative, got {value!r}")
    return parsed


def _parse_context_tokens(value: str) -> int:
    normalized = value.strip().upper().replace(",", "")
    multipliers = {"K": 1024, "M": 1024 * 1024}
    suffix = normalized[-1:] if normalized else ""
    multiplier = multipliers.get(suffix, 1)
    number = normalized[:-1] if suffix in multipliers else normalized
    try:
        parsed = float(number)
    except ValueError as exc:
        raise ValueError(f"invalid catalog context length {value!r}") from exc
    tokens = int(parsed * multiplier)
    if tokens <= 0:
        raise ValueError(f"catalog context length must be positive, got {value!r}")
    return tokens


class TinkerCatalogEntry(BaseModel):
    """One exact model row from Tinker's machine-readable catalog."""

    model_config = ConfigDict(extra="allow")

    name: str
    tinker_id: str
    context: str
    prefill: str
    cached_prefill: str
    sample: str
    type: str = ""
    arch: str = ""
    size: str = ""
    url: str | None = None
    note: str | None = None

    @field_validator("tinker_id", "context", "prefill", "cached_prefill", "sample")
    @classmethod
    def required_strings_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("catalog field cannot be blank")
        return value

    @property
    def context_tokens(self) -> int:
        return _parse_context_tokens(self.context)

    @property
    def prefill_usd_per_mtok(self) -> float:
        return _parse_usd_per_mtok(self.prefill)

    @property
    def cached_prefill_usd_per_mtok(self) -> float:
        return _parse_usd_per_mtok(self.cached_prefill)

    @property
    def sample_usd_per_mtok(self) -> float:
        return _parse_usd_per_mtok(self.sample)

    def pricing_record(self) -> dict[str, Any]:
        return {
            "tinker_id": self.tinker_id,
            "context_tokens": self.context_tokens,
            "prefill_usd_per_mtok": self.prefill_usd_per_mtok,
            "cached_prefill_usd_per_mtok": self.cached_prefill_usd_per_mtok,
            "sample_usd_per_mtok": self.sample_usd_per_mtok,
            "catalog_note": self.note,
        }


class TinkerCatalogSnapshot(BaseModel):
    """Immutable local snapshot used for preflight, costing, and provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source_url: str
    retrieved_at: str
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    models: list[dict[str, Any]] = Field(min_length=1)

    def verified_entries(self) -> list[TinkerCatalogEntry]:
        observed = hashlib.sha256(_canonical_catalog_bytes(self.models)).hexdigest()
        if observed != self.catalog_sha256:
            raise ValueError(
                "Tinker catalog snapshot hash mismatch: "
                f"recorded {self.catalog_sha256}, observed {observed}"
            )
        entries = [TinkerCatalogEntry.model_validate(item) for item in self.models]
        ids = [entry.tinker_id for entry in entries]
        duplicates = sorted(model_id for model_id in set(ids) if ids.count(model_id) > 1)
        if duplicates:
            raise ValueError(f"Tinker catalog contains duplicate model IDs: {duplicates}")
        return entries

    def entry(self, model_id: str) -> TinkerCatalogEntry:
        matches = [entry for entry in self.verified_entries() if entry.tinker_id == model_id]
        if not matches:
            raise ValueError(f"model {model_id!r} is absent from the frozen Tinker catalog")
        return matches[0]


def load_tinker_catalog(path: Path) -> TinkerCatalogSnapshot:
    snapshot = TinkerCatalogSnapshot.model_validate_json(path.read_text())
    snapshot.verified_entries()
    return snapshot


def snapshot_tinker_catalog(
    output: Path,
    *,
    source_url: str = TINKER_CATALOG_URL,
    timeout_seconds: float = 30.0,
) -> TinkerCatalogSnapshot:
    """Fetch and freeze the full catalog without silently replacing an earlier snapshot."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen Tinker catalog {output}")
    response = httpx.get(source_url, timeout=timeout_seconds, follow_redirects=True)
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("Tinker catalog response must be a JSON array of objects")
    models: list[dict[str, Any]] = [dict(item) for item in raw]
    snapshot = TinkerCatalogSnapshot(
        source_url=source_url,
        retrieved_at=datetime.now(UTC).isoformat(),
        catalog_sha256=hashlib.sha256(_canonical_catalog_bytes(models)).hexdigest(),
        models=models,
    )
    snapshot.verified_entries()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(snapshot.model_dump_json(indent=2) + "\n")
    temporary.replace(output)
    return snapshot

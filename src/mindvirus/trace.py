from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import EventKind, TraceEvent


class TraceRecorder:
    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.events: list[TraceEvent] = []
        self._turn_index = 0
        if self.events_path.exists():
            self.events_path.unlink()

    def emit(
        self,
        kind: EventKind,
        *,
        round_index: int,
        agent_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_id=uuid.uuid4().hex,
            run_id=self.run_id,
            timestamp=datetime.now(UTC).isoformat(),
            round_index=round_index,
            turn_index=self._turn_index,
            kind=kind,
            agent_id=agent_id,
            payload=payload or {},
        )
        self._turn_index += 1
        self.events.append(event)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        return event

    def events_of_kind(self, kind: EventKind) -> list[TraceEvent]:
        return [event for event in self.events if event.kind == kind]

    def write_json(self, name: str, value: Any) -> Path:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return path

    def write_text(self, name: str, value: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class BudgetExceededError(RuntimeError):
    """Raised before dispatch when the next worst-case reservation cannot fit."""


@dataclass(frozen=True, slots=True)
class CostReservation:
    call_id: str
    variant_id: str
    input_tokens: int
    maximum_output_tokens: int
    reserved_usd: float


def calculate_uncached_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    prefill_usd_per_mtok: float,
    sample_usd_per_mtok: float,
) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be nonnegative")
    if prefill_usd_per_mtok < 0 or sample_usd_per_mtok < 0:
        raise ValueError("prices must be nonnegative")
    return (input_tokens * prefill_usd_per_mtok + output_tokens * sample_usd_per_mtok) / 1_000_000


class BudgetLedger:
    """Concurrency-safe reservation ledger for one experiment's Tinker calls."""

    def __init__(self, maximum_usd: float, *, state_path: Path | None = None) -> None:
        if maximum_usd <= 0:
            raise ValueError("maximum budget must be positive")
        self.maximum_usd = maximum_usd
        self.state_path = state_path
        self._settled_usd = 0.0
        self._uncertain_usd = 0.0
        self._reservations: dict[str, CostReservation] = {}
        self._settled_calls = 0
        self._uncertain_calls = 0
        self._recovered_reservations = 0
        self._recovered_reservation_usd = 0.0
        self._lock = asyncio.Lock()
        if state_path is not None and state_path.exists():
            self._restore(state_path)
        self._execution_start_settled_usd = self._settled_usd
        self._execution_start_uncertain_usd = self._uncertain_usd - self._recovered_reservation_usd
        self._execution_start_settled_calls = self._settled_calls
        self._execution_start_uncertain_calls = self._uncertain_calls - self._recovered_reservations
        self._persist_unlocked()

    async def reserve(
        self,
        *,
        call_id: str,
        variant_id: str,
        input_tokens: int,
        maximum_output_tokens: int,
        prefill_usd_per_mtok: float,
        sample_usd_per_mtok: float,
    ) -> CostReservation:
        if not call_id:
            raise ValueError("budget reservations require a stable call_id")
        amount = calculate_uncached_cost_usd(
            input_tokens=input_tokens,
            output_tokens=maximum_output_tokens,
            prefill_usd_per_mtok=prefill_usd_per_mtok,
            sample_usd_per_mtok=sample_usd_per_mtok,
        )
        reservation = CostReservation(
            call_id=call_id,
            variant_id=variant_id,
            input_tokens=input_tokens,
            maximum_output_tokens=maximum_output_tokens,
            reserved_usd=amount,
        )
        async with self._lock:
            if call_id in self._reservations:
                raise RuntimeError(f"call {call_id!r} already has a budget reservation")
            committed = (
                self._settled_usd
                + self._uncertain_usd
                + sum(item.reserved_usd for item in self._reservations.values())
            )
            projected = committed + amount
            if projected > self.maximum_usd + 1e-12:
                raise BudgetExceededError(
                    f"call {call_id} would reserve ${amount:.8f}, taking the experiment "
                    f"from ${committed:.8f} to ${projected:.8f} above the "
                    f"${self.maximum_usd:.8f} cap"
                )
            self._reservations[call_id] = reservation
            self._persist_unlocked()
        return reservation

    async def settle(self, call_id: str, actual_cost_usd: float) -> None:
        if actual_cost_usd < 0:
            raise ValueError("actual cost must be nonnegative")
        async with self._lock:
            reservation = self._reservations.get(call_id)
            if reservation is None:
                raise RuntimeError(f"call {call_id!r} has no active budget reservation")
            if actual_cost_usd > reservation.reserved_usd + 1e-12:
                raise RuntimeError(
                    f"actual cost ${actual_cost_usd:.8f} exceeds call {call_id!r} "
                    f"reservation ${reservation.reserved_usd:.8f}"
                )
            self._reservations.pop(call_id)
            self._settled_usd += actual_cost_usd
            self._settled_calls += 1
            self._persist_unlocked()

    async def mark_uncertain(self, call_id: str) -> None:
        """Commit the worst-case reservation after an ambiguous provider failure."""

        async with self._lock:
            reservation = self._reservations.pop(call_id, None)
            if reservation is None:
                raise RuntimeError(f"call {call_id!r} has no active budget reservation")
            self._uncertain_usd += reservation.reserved_usd
            self._uncertain_calls += 1
            self._persist_unlocked()

    async def cancel_before_dispatch(self, call_id: str) -> None:
        async with self._lock:
            reservation = self._reservations.pop(call_id, None)
            if reservation is None:
                raise RuntimeError(f"call {call_id!r} has no active budget reservation")
            self._persist_unlocked()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot_unlocked()

    def _restore(self, path: Path) -> None:
        payload = json.loads(path.read_text())
        recorded_checksum = payload.get("journal_sha256")
        if not isinstance(recorded_checksum, str) or not recorded_checksum:
            raise ValueError(f"budget journal {path} has no integrity checksum")
        signed = {key: value for key, value in payload.items() if key != "journal_sha256"}
        if _journal_checksum(signed) != recorded_checksum:
            raise ValueError(
                f"budget journal {path} fails its integrity checksum; refusing to reopen "
                "budget state from a corrupted or hand-edited journal"
            )
        recorded_maximum = float(payload.get("maximum_usd", -1))
        if abs(recorded_maximum - self.maximum_usd) > 1e-12:
            raise ValueError(
                f"budget journal {path} records cap ${recorded_maximum:.8f}, not configured "
                f"cap ${self.maximum_usd:.8f}"
            )
        self._settled_usd = float(payload.get("settled_usd", 0))
        self._uncertain_usd = float(payload.get("uncertain_usd", 0))
        self._settled_calls = int(payload.get("settled_calls", 0))
        self._uncertain_calls = int(payload.get("uncertain_calls", 0))
        raw_reservations = payload.get("active_reservations", [])
        if not isinstance(raw_reservations, list):
            raise ValueError(f"budget journal {path} has invalid active_reservations")
        recovered = [CostReservation(**item) for item in raw_reservations]
        if recovered:
            self._recovered_reservation_usd = sum(item.reserved_usd for item in recovered)
            self._uncertain_usd += self._recovered_reservation_usd
            self._uncertain_calls += len(recovered)
            self._recovered_reservations += len(recovered)
        if self._settled_usd + self._uncertain_usd > self.maximum_usd + 1e-12:
            raise ValueError(f"budget journal {path} already exceeds the configured hard cap")

    def _snapshot_unlocked(self) -> dict[str, Any]:
        reserved = sum(item.reserved_usd for item in self._reservations.values())
        committed = self._settled_usd + self._uncertain_usd + reserved
        return {
            "maximum_usd": self.maximum_usd,
            "state_path": str(self.state_path) if self.state_path is not None else None,
            "prior_committed_usd": self._execution_start_settled_usd
            + self._execution_start_uncertain_usd,
            "settled_usd": self._settled_usd,
            "uncertain_usd": self._uncertain_usd,
            "execution_settled_usd": self._settled_usd - self._execution_start_settled_usd,
            "execution_uncertain_usd": self._uncertain_usd - self._execution_start_uncertain_usd,
            "active_reserved_usd": reserved,
            "committed_usd": committed,
            "remaining_usd": self.maximum_usd - committed,
            "settled_calls": self._settled_calls,
            "uncertain_calls": self._uncertain_calls,
            "execution_settled_calls": self._settled_calls - self._execution_start_settled_calls,
            "execution_uncertain_calls": self._uncertain_calls
            - self._execution_start_uncertain_calls,
            "recovered_active_reservations": self._recovered_reservations,
            "recovered_active_reservation_usd": self._recovered_reservation_usd,
            "active_reservations": [
                asdict(item)
                for item in sorted(self._reservations.values(), key=lambda item: item.call_id)
            ],
        }

    def _persist_unlocked(self) -> None:
        if self.state_path is None:
            return
        value = self._snapshot_unlocked()
        value["journal_sha256"] = _journal_checksum(value)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.state_path)


def _journal_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

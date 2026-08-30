from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mindvirus.budget import BudgetExceededError, BudgetLedger, calculate_uncached_cost_usd


def test_uncached_cost_uses_per_million_token_prices() -> None:
    assert calculate_uncached_cost_usd(
        input_tokens=100_000,
        output_tokens=20_000,
        prefill_usd_per_mtok=0.2,
        sample_usd_per_mtok=0.5,
    ) == pytest.approx(0.03)


async def test_concurrent_reservations_cannot_cross_hard_cap() -> None:
    ledger = BudgetLedger(1.0)

    async def reserve(call_id: str) -> object:
        return await ledger.reserve(
            call_id=call_id,
            variant_id="variant",
            input_tokens=600_000,
            maximum_output_tokens=0,
            prefill_usd_per_mtok=1.0,
            sample_usd_per_mtok=1.0,
        )

    results = await asyncio.gather(reserve("one"), reserve("two"), return_exceptions=True)
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, BudgetExceededError) for result in results) == 1
    snapshot = await ledger.snapshot()
    assert snapshot["active_reserved_usd"] == pytest.approx(0.6)
    assert snapshot["committed_usd"] <= snapshot["maximum_usd"]


async def test_failed_dispatch_remains_conservatively_committed() -> None:
    ledger = BudgetLedger(0.01)
    reservation = await ledger.reserve(
        call_id="ambiguous",
        variant_id="variant",
        input_tokens=1_000,
        maximum_output_tokens=1_000,
        prefill_usd_per_mtok=1.0,
        sample_usd_per_mtok=2.0,
    )
    await ledger.mark_uncertain("ambiguous")
    snapshot = await ledger.snapshot()
    assert snapshot["uncertain_calls"] == 1
    assert snapshot["uncertain_usd"] == pytest.approx(reservation.reserved_usd)
    assert snapshot["active_reservations"] == []


async def test_budget_journal_survives_restart_and_recovers_inflight_as_uncertain(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "budget-state.json"
    first = BudgetLedger(0.01, state_path=state_path)
    reservation = await first.reserve(
        call_id="crashed-call",
        variant_id="variant",
        input_tokens=1_000,
        maximum_output_tokens=1_000,
        prefill_usd_per_mtok=1.0,
        sample_usd_per_mtok=2.0,
    )
    assert state_path.exists()

    resumed = BudgetLedger(0.01, state_path=state_path)
    snapshot = await resumed.snapshot()
    assert snapshot["active_reservations"] == []
    assert snapshot["recovered_active_reservations"] == 1
    assert snapshot["execution_uncertain_calls"] == 1
    assert snapshot["uncertain_usd"] == pytest.approx(reservation.reserved_usd)

    with pytest.raises(BudgetExceededError):
        await resumed.reserve(
            call_id="too-expensive-after-resume",
            variant_id="variant",
            input_tokens=8_000,
            maximum_output_tokens=0,
            prefill_usd_per_mtok=1.0,
            sample_usd_per_mtok=1.0,
        )


async def test_budget_journal_checksum_rejects_tampered_state(tmp_path: Path) -> None:
    state_path = tmp_path / "budget-state.json"
    ledger = BudgetLedger(0.01, state_path=state_path)
    await ledger.reserve(
        call_id="settled-call",
        variant_id="variant",
        input_tokens=1_000,
        maximum_output_tokens=1_000,
        prefill_usd_per_mtok=1.0,
        sample_usd_per_mtok=2.0,
    )
    await ledger.settle("settled-call", 0.000002)
    journal = json.loads(state_path.read_text())
    assert journal["journal_sha256"]

    journal["settled_usd"] = 0.0  # hand-edit to reopen budget headroom
    state_path.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="integrity checksum"):
        BudgetLedger(0.01, state_path=state_path)


async def test_budget_journal_without_checksum_is_refused(tmp_path: Path) -> None:
    state_path = tmp_path / "budget-state.json"
    state_path.write_text(
        json.dumps({"maximum_usd": 0.01, "settled_usd": 0.0, "uncertain_usd": 0.0})
    )
    with pytest.raises(ValueError, match="integrity checksum"):
        BudgetLedger(0.01, state_path=state_path)


async def test_snapshot_does_not_mutate_the_journal(tmp_path: Path) -> None:
    state_path = tmp_path / "budget-state.json"
    ledger = BudgetLedger(0.01, state_path=state_path)
    before = state_path.read_bytes()
    snapshot = await ledger.snapshot()
    assert state_path.read_bytes() == before
    assert "journal_sha256" not in snapshot


async def test_mutations_persist_to_the_journal(tmp_path: Path) -> None:
    state_path = tmp_path / "budget-state.json"
    ledger = BudgetLedger(0.01, state_path=state_path)
    await ledger.reserve(
        call_id="in-flight",
        variant_id="variant",
        input_tokens=1_000,
        maximum_output_tokens=1_000,
        prefill_usd_per_mtok=1.0,
        sample_usd_per_mtok=2.0,
    )
    journal = json.loads(state_path.read_text())
    assert [item["call_id"] for item in journal["active_reservations"]] == ["in-flight"]
    await ledger.settle("in-flight", 0.000002)
    journal = json.loads(state_path.read_text())
    assert journal["active_reservations"] == []
    assert journal["settled_calls"] == 1

from __future__ import annotations

from pathlib import Path

import pytest

from mindvirus.runner import _ExperimentLock


def test_experiment_lock_excludes_a_second_process_handle(tmp_path: Path) -> None:
    path = tmp_path / ".experiment.lock"
    first = _ExperimentLock(path, execution_id="first")
    second = _ExperimentLock(path, execution_id="second")
    third = _ExperimentLock(path, execution_id="third")

    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already active"):
            second.acquire()
    finally:
        first.release()

    third.acquire()
    third.release()

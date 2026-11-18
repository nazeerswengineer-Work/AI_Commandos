"""
Tests for the incremental ingestion filesystem lock.
"""

from __future__ import annotations

import os
import time

import pytest

from src.ingestion_lock import (
    IngestionFileLock,
    IngestionLockError,
)


def test_lock_is_created_and_removed(
    tmp_path,
) -> None:

    lock_path = (
        tmp_path
        / "ingestion.lock"
    )

    with IngestionFileLock(
        lock_path
    ):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_second_lock_is_rejected(
    tmp_path,
) -> None:

    lock_path = (
        tmp_path
        / "ingestion.lock"
    )

    first = IngestionFileLock(
        lock_path
    )

    first.acquire()

    try:
        second = IngestionFileLock(
            lock_path,
            timeout_seconds=0,
        )

        with pytest.raises(
            IngestionLockError
        ):
            second.acquire()

    finally:
        first.release()


def test_stale_lock_is_replaced(
    tmp_path,
) -> None:

    lock_path = (
        tmp_path
        / "ingestion.lock"
    )

    lock_path.write_text(
        "{}",
        encoding="utf-8",
    )

    old_time = (
        time.time()
        - 7200
    )

    os.utime(
        lock_path,
        (
            old_time,
            old_time,
        ),
    )

    with IngestionFileLock(
        lock_path,
        stale_after_seconds=60,
    ):
        assert lock_path.exists()
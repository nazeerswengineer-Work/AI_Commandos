"""
Cross-process filesystem lock for incremental ingestion.

The lock prevents two synchronization commands from mutating ChromaDB
and ingestion_manifest.json concurrently.

The implementation uses exclusive file creation and therefore works on
Windows and Linux without an additional package.
"""

from __future__ import annotations

import json
import os
import socket
import time

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IngestionLockError(
    RuntimeError
):
    """Raised when the ingestion lock cannot be acquired safely."""


def utc_now_iso() -> str:
    """Return current UTC time."""

    return datetime.now(
        timezone.utc
    ).isoformat()


@dataclass(frozen=True)
class IngestionLockMetadata:
    """Diagnostic information stored in the lock file."""

    process_id: int
    host_name: str
    acquired_at_utc: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert lock metadata to JSON-safe data."""

        return {
            "process_id": self.process_id,
            "host_name": self.host_name,
            "acquired_at_utc": (
                self.acquired_at_utc
            ),
        }


class IngestionFileLock:
    """Exclusive filesystem lock with timeout and stale-lock handling."""

    def __init__(
        self,
        lock_path: str | Path,
        *,
        timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.1,
        stale_after_seconds: float = 3600.0,
    ) -> None:
        """Configure the lock."""

        if timeout_seconds < 0:
            raise ValueError(
                "timeout_seconds must not be negative."
            )

        if poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be positive."
            )

        if stale_after_seconds <= 0:
            raise ValueError(
                "stale_after_seconds must be positive."
            )

        self.lock_path = (
            Path(
                lock_path
            )
            .expanduser()
            .resolve()
        )

        self.timeout_seconds = (
            timeout_seconds
        )

        self.poll_interval_seconds = (
            poll_interval_seconds
        )

        self.stale_after_seconds = (
            stale_after_seconds
        )

        self._acquired = False

    def _is_stale(
        self,
    ) -> bool:
        """Return whether the existing lock is older than the limit."""

        try:
            stat = self.lock_path.stat()

        except FileNotFoundError:
            return False

        age_seconds = (
            time.time()
            - stat.st_mtime
        )

        return bool(
            age_seconds
            > self.stale_after_seconds
        )

    def _remove_stale_lock(
        self,
    ) -> bool:
        """Remove a stale lock when it still exists."""

        if not self._is_stale():
            return False

        try:
            self.lock_path.unlink()
            return True

        except FileNotFoundError:
            return True

        except OSError as exc:
            raise IngestionLockError(
                "Unable to remove stale ingestion lock: "
                f"{self.lock_path}"
            ) from exc

    def acquire(
        self,
    ) -> None:
        """Acquire the exclusive lock."""

        if self._acquired:
            raise IngestionLockError(
                "Ingestion lock is already acquired "
                "by this object."
            )

        self.lock_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        deadline = (
            time.monotonic()
            + self.timeout_seconds
        )

        metadata = IngestionLockMetadata(
            process_id=os.getpid(),
            host_name=socket.gethostname(),
            acquired_at_utc=utc_now_iso(),
        )

        payload = (
            json.dumps(
                metadata.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
            + "\n"
        )

        while True:
            try:
                descriptor = os.open(
                    self.lock_path,
                    (
                        os.O_CREAT
                        | os.O_EXCL
                        | os.O_WRONLY
                    ),
                )

            except FileExistsError:
                if self._remove_stale_lock():
                    continue

                if (
                    time.monotonic()
                    >= deadline
                ):
                    raise IngestionLockError(
                        "Another incremental ingestion "
                        "process holds the lock: "
                        f"{self.lock_path}"
                    )

                time.sleep(
                    self.poll_interval_seconds
                )

                continue

            except OSError as exc:
                raise IngestionLockError(
                    "Unable to create ingestion lock: "
                    f"{self.lock_path}"
                ) from exc

            try:
                with os.fdopen(
                    descriptor,
                    "w",
                    encoding="utf-8",
                    newline="\n",
                ) as stream:
                    stream.write(
                        payload
                    )

                    stream.flush()

                    os.fsync(
                        stream.fileno()
                    )

            except Exception:
                try:
                    self.lock_path.unlink(
                        missing_ok=True
                    )

                except OSError:
                    pass

                raise

            self._acquired = True
            return

    def release(
        self,
    ) -> None:
        """Release the lock."""

        if not self._acquired:
            return

        try:
            self.lock_path.unlink(
                missing_ok=True
            )

        except OSError as exc:
            raise IngestionLockError(
                "Unable to release ingestion lock: "
                f"{self.lock_path}"
            ) from exc

        finally:
            self._acquired = False

    def __enter__(
        self,
    ) -> "IngestionFileLock":
        """Acquire the lock."""

        self.acquire()
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ) -> None:
        """Release the lock."""

        self.release()
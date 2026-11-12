"""
Production timing metrics for REQTRACE AI.

Phase 15D records:

- deterministic pipeline duration
- narration duration
- total request duration

Metrics do not influence engineering decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProductionTiming:
    """Timing measurements for one production request."""

    deterministic_seconds: float
    narration_seconds: float
    total_seconds: float

    def __post_init__(
        self,
    ) -> None:
        """Validate timing values."""

        values = (
            self.deterministic_seconds,
            self.narration_seconds,
            self.total_seconds,
        )

        if any(
            value < 0.0
            for value in values
        ):
            raise ValueError(
                "Production timing values "
                "must not be negative."
            )

        minimum_total = (
            self.deterministic_seconds
            + self.narration_seconds
        )

        tolerance = 0.05

        if (
            self.total_seconds
            + tolerance
            < minimum_total
        ):
            raise ValueError(
                "total_seconds cannot be less than "
                "the measured pipeline components."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert metrics into JSON-safe data."""

        return {
            "deterministic_seconds": (
                self.deterministic_seconds
            ),
            "narration_seconds": (
                self.narration_seconds
            ),
            "total_seconds": (
                self.total_seconds
            ),
        }
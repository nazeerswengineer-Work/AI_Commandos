"""
Production-safe error models for REQTRACE AI.

These errors separate user-facing failure information from internal
technical diagnostics.

The UI should present:

    user_message

Technical details should appear only inside an expandable diagnostic
section and must not replace the user-facing message.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProductionErrorCode(str, Enum):
    """Stable REQTRACE AI production error codes."""

    EMPTY_QUERY = "PRD-001_EMPTY_QUERY"

    INSUFFICIENT_EVIDENCE = (
        "PRD-002_INSUFFICIENT_EVIDENCE"
    )

    CORPUS_UNAVAILABLE = (
        "PRD-003_CORPUS_UNAVAILABLE"
    )

    RETRIEVAL_UNAVAILABLE = (
        "PRD-004_RETRIEVAL_UNAVAILABLE"
    )

    GROUNDING_FAILED = (
        "PRD-005_GROUNDING_FAILED"
    )

    UNSAFE_DELIVERY = (
        "PRD-006_UNSAFE_DELIVERY"
    )

    CONFIGURATION_ERROR = (
        "PRD-007_CONFIGURATION_ERROR"
    )

    INTERNAL_ERROR = (
        "PRD-999_INTERNAL_ERROR"
    )


ProductionErrorCode.__test__ = False


@dataclass(frozen=True)
class ProductionFailure:
    """Application-facing production failure."""

    code: ProductionErrorCode
    user_message: str

    technical_message: str | None = None
    recoverable: bool = False

    suggested_action: str | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the failure into JSON-safe data."""

        return {
            "code": self.code.value,
            "user_message": (
                self.user_message
            ),
            "technical_message": (
                self.technical_message
            ),
            "recoverable": (
                self.recoverable
            ),
            "suggested_action": (
                self.suggested_action
            ),
        }


class ReqTraceProductionError(
    RuntimeError
):
    """Exception carrying a structured production failure."""

    def __init__(
        self,
        failure: ProductionFailure,
    ) -> None:
        super().__init__(
            failure.user_message
        )

        self.failure = failure
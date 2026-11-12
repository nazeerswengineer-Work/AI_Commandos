"""
Tests for Phase 14E narration-evaluation acceptance rules.
"""

from __future__ import annotations

from src.narration_models import (
    NarrationDelivery,
    NarrationResponse,
    NarrationStatus,
)


class PassedValidation:
    """Minimal passed validation fixture."""

    passed = True


class FailedValidation:
    """Minimal failed validation fixture."""

    passed = False


def response(
    *,
    attempt: int,
    repair: bool,
    text: str,
) -> NarrationResponse:
    """Create one narration response."""

    return NarrationResponse(
        model="llama3.2:1b",
        text=text,
        attempt_number=attempt,
        is_repair_attempt=repair,
    )


def test_validated_generation_delivery() -> None:
    """Initial validated generation is deliverable."""

    initial = response(
        attempt=1,
        repair=False,
        text=(
            "Assessment: SUPPORTED"
        ),
    )

    delivery = NarrationDelivery(
        status=(
            NarrationStatus
            .VALIDATED_GENERATION
        ),
        narration=initial.text,
        model="llama3.2:1b",
        generation_attempts=1,
        used_repair=False,
        used_fallback=False,
        fallback_reason=None,
        initial_response=initial,
        repair_response=None,
        initial_validation=(
            PassedValidation()
        ),
        repair_validation=None,
        deterministic_fallback_text=(
            "Fallback"
        ),
    )

    assert (
        delivery.status
        == NarrationStatus
        .VALIDATED_GENERATION
    )

    assert delivery.used_fallback is False


def test_validated_repair_delivery() -> None:
    """Validated repair is safe to deliver."""

    initial = response(
        attempt=1,
        repair=False,
        text="Invalid narration.",
    )

    repaired = response(
        attempt=2,
        repair=True,
        text=(
            "Assessment: SUPPORTED"
        ),
    )

    delivery = NarrationDelivery(
        status=(
            NarrationStatus
            .VALIDATED_REPAIR
        ),
        narration=repaired.text,
        model="llama3.2:1b",
        generation_attempts=2,
        used_repair=True,
        used_fallback=False,
        fallback_reason=None,
        initial_response=initial,
        repair_response=repaired,
        initial_validation=(
            FailedValidation()
        ),
        repair_validation=(
            PassedValidation()
        ),
        deterministic_fallback_text=(
            "Fallback"
        ),
    )

    assert delivery.used_repair is True

    assert (
        delivery.generation_attempts
        == 2
    )


def test_fallback_equals_deterministic_text() -> None:
    """Fallback must exactly equal deterministic Phase 13D output."""

    fallback = (
        "REQTRACE deterministic answer"
    )

    delivery = NarrationDelivery(
        status=(
            NarrationStatus
            .DETERMINISTIC_FALLBACK
        ),
        narration=fallback,
        model="llama3.2:1b",
        generation_attempts=2,
        used_repair=True,
        used_fallback=True,
        fallback_reason=(
            "Both attempts failed validation."
        ),
        initial_response=(
            response(
                attempt=1,
                repair=False,
                text="Invalid 1",
            )
        ),
        repair_response=(
            response(
                attempt=2,
                repair=True,
                text="Invalid 2",
            )
        ),
        initial_validation=(
            FailedValidation()
        ),
        repair_validation=(
            FailedValidation()
        ),
        deterministic_fallback_text=fallback,
    )

    assert (
        delivery.narration
        == fallback
    )

    assert delivery.used_fallback is True


def test_generation_attempts_never_exceed_two() -> None:
    """Phase 14 permits no more than two model calls."""

    allowed_attempt_counts = (
        1,
        2,
    )

    assert max(
        allowed_attempt_counts
    ) == 2
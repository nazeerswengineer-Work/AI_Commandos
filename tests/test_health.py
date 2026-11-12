"""
Phase 15D health-model tests.
"""

from __future__ import annotations

from src.health import (
    HealthCheck,
    HealthStatus,
    SystemHealth,
)


def test_core_can_be_ready_when_ollama_is_degraded() -> None:

    health = SystemHealth(
        status=(
            HealthStatus.DEGRADED
        ),
        checks=(
            HealthCheck(
                component=(
                    "Deterministic engine"
                ),
                status=(
                    HealthStatus.READY
                ),
                message="Ready.",
                required=True,
            ),
            HealthCheck(
                component="Ollama API",
                status=(
                    HealthStatus.DEGRADED
                ),
                message="Unavailable.",
                required=False,
            ),
            HealthCheck(
                component=(
                    "Narration model"
                ),
                status=(
                    HealthStatus.DEGRADED
                ),
                message="Unavailable.",
                required=False,
            ),
        ),
    )

    assert health.core_ready is True
    assert health.narration_ready is False


def test_required_failure_makes_core_unavailable() -> None:

    health = SystemHealth(
        status=(
            HealthStatus.UNAVAILABLE
        ),
        checks=(
            HealthCheck(
                component=(
                    "Deterministic engine"
                ),
                status=(
                    HealthStatus.UNAVAILABLE
                ),
                message="Failed.",
                required=True,
            ),
        ),
    )

    assert health.core_ready is False


def test_health_is_json_safe() -> None:

    health = SystemHealth(
        status=HealthStatus.READY,
        checks=(
            HealthCheck(
                component=(
                    "Deterministic engine"
                ),
                status=(
                    HealthStatus.READY
                ),
                message="Ready.",
                required=True,
            ),
        ),
    )

    payload = health.to_dict()

    assert payload[
        "status"
    ] == "READY"

    assert payload[
        "core_ready"
    ] is True
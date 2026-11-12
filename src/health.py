"""
REQTRACE AI production health checks.

Health checks do not perform engineering assessment.

They inspect operational dependencies:

- project structure
- evidence corpus
- deterministic production pipeline
- Ollama API
- required narration model
- deterministic fallback availability
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from config import Settings

from src.production_pipeline import (
    ProductionPipelineConfig,
    ReqTraceProductionPipeline,
)


class HealthStatus(str, Enum):
    """Operational health status."""

    READY = "READY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


HealthStatus.__test__ = False


@dataclass(frozen=True)
class HealthCheck:
    """One operational health check."""

    component: str
    status: HealthStatus
    message: str

    required: bool

    @property
    def healthy(
        self,
    ) -> bool:
        """Return True for READY status."""

        return (
            self.status
            == HealthStatus.READY
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert check into JSON-safe data."""

        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "required": self.required,
            "healthy": self.healthy,
        }


@dataclass(frozen=True)
class SystemHealth:
    """Complete REQTRACE AI health report."""

    status: HealthStatus

    checks: tuple[
        HealthCheck,
        ...
    ]

    @property
    def core_ready(
        self,
    ) -> bool:
        """
        Core readiness ignores optional narration failures.

        Required components must all be READY.
        """

        return all(
            check.status
            == HealthStatus.READY

            for check in self.checks
            if check.required
        )

    @property
    def narration_ready(
        self,
    ) -> bool:
        """Return whether Ollama and narration model are ready."""

        narration_components = {
            "Ollama API",
            "Narration model",
        }

        matching = [
            check
            for check in self.checks
            if (
                check.component
                in narration_components
            )
        ]

        return bool(
            matching
            and all(
                check.status
                == HealthStatus.READY
                for check in matching
            )
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert health report into JSON-safe data."""

        return {
            "status": self.status.value,
            "core_ready": self.core_ready,
            "narration_ready": (
                self.narration_ready
            ),
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
        }


def check_project_structure(
    base_directory: Path,
) -> HealthCheck:
    """Check required project directories."""

    required_paths = (
        base_directory / "src",
        base_directory / "documents",
    )

    missing = tuple(
        str(path.name)
        for path in required_paths
        if not path.exists()
    )

    if missing:
        return HealthCheck(
            component="Project structure",
            status=(
                HealthStatus.UNAVAILABLE
            ),
            message=(
                "Missing required project path(s): "
                + ", ".join(missing)
            ),
            required=True,
        )

    return HealthCheck(
        component="Project structure",
        status=HealthStatus.READY,
        message=(
            "Required project directories are available."
        ),
        required=True,
    )


def check_deterministic_pipeline(
    *,
    settings: Settings,
    base_directory: Path,
) -> HealthCheck:
    """Run one deterministic production smoke test."""

    try:
        pipeline = (
            ReqTraceProductionPipeline(
                settings=settings,
                base_directory=(
                    base_directory
                ),
                config=(
                    ProductionPipelineConfig(
                        narration_enabled=False,
                        allow_repair_attempt=False,
                    )
                ),
            )
        )

        result = pipeline.run(
            "OBS-023"
        )

    except Exception as exc:
        return HealthCheck(
            component=(
                "Deterministic engine"
            ),
            status=(
                HealthStatus.UNAVAILABLE
            ),
            message=(
                "Deterministic pipeline smoke test failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            required=True,
        )

    if (
        not result.safe_delivery
        or not result.grounding_passed
    ):
        return HealthCheck(
            component=(
                "Deterministic engine"
            ),
            status=(
                HealthStatus.UNAVAILABLE
            ),
            message=(
                "Deterministic pipeline returned "
                "an unsafe result."
            ),
            required=True,
        )

    return HealthCheck(
        component="Deterministic engine",
        status=HealthStatus.READY,
        message=(
            "Deterministic assessment, grounding, "
            "and fallback are operational."
        ),
        required=True,
    )


def check_ollama(
    *,
    settings: Settings,
    model_name: str,
) -> tuple[
    HealthCheck,
    HealthCheck,
]:
    """Check Ollama API and required narration model."""

    base_url = (
        settings
        .ollama_base_url
        .rstrip("/")
    )

    timeout = httpx.Timeout(
        connect=3.0,
        read=5.0,
        write=3.0,
        pool=3.0,
    )

    try:
        with httpx.Client(
            base_url=base_url,
            timeout=timeout,
        ) as client:

            response = client.get(
                "/api/tags"
            )

    except httpx.HTTPError as exc:
        return (
            HealthCheck(
                component="Ollama API",
                status=(
                    HealthStatus.DEGRADED
                ),
                message=(
                    "Ollama is unavailable. "
                    "Deterministic fallback remains available. "
                    f"{type(exc).__name__}: {exc}"
                ),
                required=False,
            ),

            HealthCheck(
                component="Narration model",
                status=(
                    HealthStatus.DEGRADED
                ),
                message=(
                    "Narration model availability "
                    "could not be checked."
                ),
                required=False,
            ),
        )

    if not response.is_success:
        return (
            HealthCheck(
                component="Ollama API",
                status=(
                    HealthStatus.DEGRADED
                ),
                message=(
                    "Ollama returned HTTP "
                    f"{response.status_code}. "
                    "Deterministic fallback remains available."
                ),
                required=False,
            ),

            HealthCheck(
                component="Narration model",
                status=(
                    HealthStatus.DEGRADED
                ),
                message=(
                    "Narration model availability "
                    "could not be checked."
                ),
                required=False,
            ),
        )

    try:
        payload = response.json()

    except ValueError:
        return (
            HealthCheck(
                component="Ollama API",
                status=(
                    HealthStatus.DEGRADED
                ),
                message=(
                    "Ollama returned invalid JSON."
                ),
                required=False,
            ),

            HealthCheck(
                component="Narration model",
                status=(
                    HealthStatus.DEGRADED
                ),
                message=(
                    "Narration model availability "
                    "could not be checked."
                ),
                required=False,
            ),
        )

    models = payload.get(
        "models",
        []
    )

    model_names = {
        str(
            model.get(
                "name",
                "",
            )
        )
        for model in models
        if isinstance(
            model,
            dict,
        )
    }

    model_ready = bool(
        model_name in model_names
    )

    return (
        HealthCheck(
            component="Ollama API",
            status=HealthStatus.READY,
            message=(
                "Local Ollama API is reachable."
            ),
            required=False,
        ),

        HealthCheck(
            component="Narration model",
            status=(
                HealthStatus.READY
                if model_ready
                else HealthStatus.DEGRADED
            ),
            message=(
                f"{model_name} is available."
                if model_ready
                else (
                    f"{model_name} is not installed. "
                    "Deterministic fallback remains available."
                )
            ),
            required=False,
        ),
    )


def evaluate_system_health(
    *,
    settings: Settings,
    base_directory: str | Path,
    model_name: str = "llama3.2:1b",
    include_deterministic_smoke_test: bool = True,
) -> SystemHealth:
    """Run complete production health evaluation."""

    resolved_base = (
        Path(
            base_directory
        )
        .expanduser()
        .resolve()
    )

    checks: list[
        HealthCheck
    ] = []

    checks.append(
        check_project_structure(
            resolved_base
        )
    )

    if include_deterministic_smoke_test:
        checks.append(
            check_deterministic_pipeline(
                settings=settings,
                base_directory=(
                    resolved_base
                ),
            )
        )

    ollama_check, model_check = (
        check_ollama(
            settings=settings,
            model_name=model_name,
        )
    )

    checks.extend(
        (
            ollama_check,
            model_check,
        )
    )

    required_failure = any(
        check.required
        and check.status
        == HealthStatus.UNAVAILABLE
        for check in checks
    )

    optional_degraded = any(
        check.status
        == HealthStatus.DEGRADED
        for check in checks
    )

    if required_failure:
        overall_status = (
            HealthStatus.UNAVAILABLE
        )

    elif optional_degraded:
        overall_status = (
            HealthStatus.DEGRADED
        )

    else:
        overall_status = (
            HealthStatus.READY
        )

    return SystemHealth(
        status=overall_status,
        checks=tuple(checks),
    )
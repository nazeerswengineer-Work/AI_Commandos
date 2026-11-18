"""
REQTRACE AI production configuration validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ConfigurationValidationError(
    RuntimeError
):
    """Raised when production configuration is invalid."""


@dataclass(frozen=True)
class ConfigurationValidationResult:

    passed: bool
    project_directory: Path

    missing_paths: tuple[str, ...]


def validate_project_configuration(
    base_directory: str | Path,
    *,
    strict: bool = True,
) -> ConfigurationValidationResult:

    base = (
        Path(base_directory)
        .expanduser()
        .resolve()
    )

    required_paths = (
        base / "src",
        base / "tests",
        base / "evaluation",
        base / "documents",
        base / "app.py",
        base / "config.py",
    )

    missing = tuple(
        str(path)
        for path in required_paths
        if not path.exists()
    )

    result = (
        ConfigurationValidationResult(
            passed=not missing,
            project_directory=base,
            missing_paths=missing,
        )
    )

    if (
        strict
        and not result.passed
    ):

        raise ConfigurationValidationError(
            "Required REQTRACE AI project "
            "resources are missing: "
            + ", ".join(missing)
        )

    return result
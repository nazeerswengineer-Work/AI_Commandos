"""
REQTRACE AI production-packaging tests.
"""

from __future__ import annotations

from pathlib import Path

from src.configuration_validation import (
    validate_project_configuration,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def test_project_configuration_is_complete() -> None:

    result = (
        validate_project_configuration(
            BASE_DIR
        )
    )

    assert result.passed is True


def test_startup_files_exist() -> None:

    required = (
        "app.py",
        "run_reqtrace.py",
        "narrate_answer.py",
        "health_check.py",
        "requirements.txt",
        ".gitignore",
        "README.md",
        "start_reqtrace.bat",
        "check_reqtrace.bat",
    )

    for file_name in required:

        assert (
            BASE_DIR
            / file_name
        ).exists()


def test_core_directories_exist() -> None:

    required = (
        "src",
        "tests",
        "evaluation",
        "documents",
    )

    for directory_name in required:

        assert (
            BASE_DIR
            / directory_name
        ).is_dir()
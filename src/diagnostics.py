"""
Startup diagnostics for REQTRACE AI.

Phase 2 diagnostics validate configuration, packages, required directories,
Ollama connectivity, and configured model availability. They do not create
ChromaDB collections, generate embeddings, or start document ingestion.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import httpx

from config import Settings
from src.logging_config import get_logger


logger = get_logger(__name__)


STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class DiagnosticResult:
    """One startup diagnostic result."""

    name: str
    status: str
    message: str
    critical: bool = False

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASS

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAIL


@dataclass(frozen=True)
class DiagnosticReport:
    """Combined startup-diagnostic report."""

    results: tuple[DiagnosticResult, ...]

    @property
    def critical_failures(self) -> tuple[DiagnosticResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.failed and result.critical
        )

    @property
    def warnings(self) -> tuple[DiagnosticResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.status == STATUS_WARNING
        )

    @property
    def can_start(self) -> bool:
        return not self.critical_failures


def create_required_directories(
    settings: Settings,
) -> DiagnosticResult:
    """Create the approved Phase 2 project directories."""

    try:
        for directory in settings.required_directories():
            directory.mkdir(parents=True, exist_ok=True)

        return DiagnosticResult(
            name="Project directories",
            status=STATUS_PASS,
            message="Required project directories are available.",
            critical=True,
        )
    except OSError as exc:
        logger.exception("Unable to create a required directory.")

        return DiagnosticResult(
            name="Project directories",
            status=STATUS_FAIL,
            message=(
                "A required project directory could not be created. "
                f"Reason: {exc}"
            ),
            critical=True,
        )


def check_directory_writable(
    directory: Path,
    display_name: str,
) -> DiagnosticResult:
    """Verify that a runtime directory can be written to."""

    test_file = directory / ".reqtrace_write_test"

    try:
        directory.mkdir(parents=True, exist_ok=True)
        test_file.write_text("write-test", encoding="utf-8")
        test_file.unlink(missing_ok=True)

        return DiagnosticResult(
            name=display_name,
            status=STATUS_PASS,
            message=f"{display_name} is writable.",
            critical=True,
        )
    except OSError as exc:
        logger.exception(
            "Directory write check failed for %s.",
            display_name,
        )

        return DiagnosticResult(
            name=display_name,
            status=STATUS_FAIL,
            message=f"{display_name} is not writable. Reason: {exc}",
            critical=True,
        )


def check_python_version() -> DiagnosticResult:
    """Report the active Python version."""

    current_version = platform.python_version()

    if sys.version_info < (3, 11):
        return DiagnosticResult(
            name="Python version",
            status=STATUS_FAIL,
            message=(
                f"Python {current_version} is active. "
                "REQTRACE AI requires Python 3.11 or newer."
            ),
            critical=True,
        )

    return DiagnosticResult(
        name="Python version",
        status=STATUS_PASS,
        message=f"Python {current_version} is active.",
        critical=True,
    )


def check_package(
    distribution_name: str,
    import_name: str | None = None,
    critical: bool = True,
) -> DiagnosticResult:
    """Check whether a package is installed and importable."""

    module_name = import_name or distribution_name.replace("-", "_")

    try:
        installed_version = version(distribution_name)
    except PackageNotFoundError:
        return DiagnosticResult(
            name=f"Package: {distribution_name}",
            status=STATUS_FAIL,
            message=f"{distribution_name} is not installed.",
            critical=critical,
        )

    try:
        importlib.import_module(module_name)
    except Exception as exc:
        logger.exception(
            "Package import failed for %s.",
            distribution_name,
        )

        return DiagnosticResult(
            name=f"Package: {distribution_name}",
            status=STATUS_FAIL,
            message=(
                f"{distribution_name} {installed_version} is installed "
                f"but could not be imported. Reason: {exc}"
            ),
            critical=critical,
        )

    return DiagnosticResult(
        name=f"Package: {distribution_name}",
        status=STATUS_PASS,
        message=(
            f"{distribution_name} {installed_version} is installed "
            "and importable."
        ),
        critical=critical,
    )


def normalize_model_name(model_name: str) -> str:
    """Normalize an Ollama model name for availability comparison."""

    normalized = model_name.strip().lower()

    if ":" not in normalized:
        normalized = f"{normalized}:latest"

    return normalized


def extract_ollama_model_names(payload: dict[str, Any]) -> set[str]:
    """Extract normalized model names from an Ollama tags response."""

    normalized_names: set[str] = set()

    models = payload.get("models", [])

    if not isinstance(models, list):
        return normalized_names

    for model in models:
        if not isinstance(model, dict):
            continue

        for key in ("name", "model"):
            candidate = model.get(key)

            if isinstance(candidate, str) and candidate.strip():
                normalized_names.add(normalize_model_name(candidate))

    return normalized_names


def check_ollama(
    settings: Settings,
) -> tuple[DiagnosticResult, ...]:
    """
    Check Ollama connectivity and configured model availability.

    This function performs only the lightweight model-list request.
    It does not run generation or embedding calls in Phase 2.
    """

    tags_url = f"{settings.ollama_base_url}/api/tags"

    timeout = httpx.Timeout(
        connect=settings.ollama_connect_timeout_seconds,
        read=settings.ollama_read_timeout_seconds,
        write=settings.ollama_read_timeout_seconds,
        pool=settings.ollama_connect_timeout_seconds,
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(tags_url)
            response.raise_for_status()
            payload = response.json()
    except httpx.ConnectError:
        return (
            DiagnosticResult(
                name="Ollama connectivity",
                status=STATUS_FAIL,
                message=(
                    "Ollama is not reachable. Start Ollama and retry."
                ),
                critical=True,
            ),
        )
    except httpx.TimeoutException:
        return (
            DiagnosticResult(
                name="Ollama connectivity",
                status=STATUS_FAIL,
                message=(
                    "Ollama did not respond before the configured timeout."
                ),
                critical=True,
            ),
        )
    except httpx.HTTPStatusError as exc:
        return (
            DiagnosticResult(
                name="Ollama connectivity",
                status=STATUS_FAIL,
                message=(
                    "Ollama returned an unexpected HTTP status: "
                    f"{exc.response.status_code}."
                ),
                critical=True,
            ),
        )
    except ValueError:
        return (
            DiagnosticResult(
                name="Ollama connectivity",
                status=STATUS_FAIL,
                message="Ollama returned an invalid JSON response.",
                critical=True,
            ),
        )
    except httpx.HTTPError as exc:
        logger.exception("Unhandled Ollama HTTP error.")

        return (
            DiagnosticResult(
                name="Ollama connectivity",
                status=STATUS_FAIL,
                message=f"Ollama request failed. Reason: {exc}",
                critical=True,
            ),
        )

    model_names = extract_ollama_model_names(payload)

    generation_model = normalize_model_name(
        settings.ollama_llm_model
    )
    embedding_model = normalize_model_name(
        settings.ollama_embed_model
    )

    connectivity_result = DiagnosticResult(
        name="Ollama connectivity",
        status=STATUS_PASS,
        message="Ollama is reachable and returned its model list.",
        critical=True,
    )

    if generation_model in model_names:
        generation_result = DiagnosticResult(
            name="Generation model",
            status=STATUS_PASS,
            message=(
                f"Configured generation model "
                f"'{settings.ollama_llm_model}' is available."
            ),
            critical=True,
        )
    else:
        generation_result = DiagnosticResult(
            name="Generation model",
            status=STATUS_FAIL,
            message=(
                f"Configured generation model "
                f"'{settings.ollama_llm_model}' is not available. "
                f"Run: ollama pull {settings.ollama_llm_model}"
            ),
            critical=True,
        )

    if embedding_model in model_names:
        embedding_result = DiagnosticResult(
            name="Embedding model",
            status=STATUS_PASS,
            message=(
                f"Configured embedding model "
                f"'{settings.ollama_embed_model}' is available."
            ),
            critical=True,
        )
    else:
        embedding_result = DiagnosticResult(
            name="Embedding model",
            status=STATUS_FAIL,
            message=(
                f"Configured embedding model "
                f"'{settings.ollama_embed_model}' is not available. "
                f"Run: ollama pull {settings.ollama_embed_model}"
            ),
            critical=True,
        )

    return (
        connectivity_result,
        generation_result,
        embedding_result,
    )


def run_startup_diagnostics(
    settings: Settings,
) -> DiagnosticReport:
    """Run all diagnostics required during Phase 2."""

    results: list[DiagnosticResult] = []

    results.append(create_required_directories(settings))
    results.append(check_python_version())

    package_checks = (
        ("pydantic", "pydantic", True),
        ("pydantic-settings", "pydantic_settings", True),
        ("python-dotenv", "dotenv", True),
        ("httpx", "httpx", True),
        ("chromadb", "chromadb", True),
        ("streamlit", "streamlit", True),
        ("PyMuPDF", "pymupdf", True),
        ("rank-bm25", "rank_bm25", True),
        ("pytest", "pytest", False),
        ("Pillow", "PIL", False),
    )

    for distribution_name, import_name, critical in package_checks:
        results.append(
            check_package(
                distribution_name=distribution_name,
                import_name=import_name,
                critical=critical,
            )
        )

    results.append(
        check_directory_writable(
            settings.logs_path,
            "Log directory",
        )
    )
    results.append(
        check_directory_writable(
            settings.data_path,
            "Data directory",
        )
    )
    results.append(
        check_directory_writable(
            settings.chroma_db_path,
            "ChromaDB directory",
        )
    )

    results.extend(check_ollama(settings))

    for result in results:
        log_method = (
            logger.error
            if result.status == STATUS_FAIL
            else logger.warning
            if result.status == STATUS_WARNING
            else logger.info
        )

        log_method(
            "%s | %s | %s",
            result.status,
            result.name,
            result.message,
        )

    return DiagnosticReport(results=tuple(results))

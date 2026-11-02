"""
REQTRACE AI Phase 2 startup entry point.

This command-line startup performs configuration and environment diagnostics.
It does not start Streamlit, create ChromaDB collections, ingest documents,
generate embeddings, or call the generation model.
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from config import get_settings
from src.diagnostics import DiagnosticReport, run_startup_diagnostics
from src.logging_config import configure_logging


NOTICE = (
    "ReqTrace AI provides evidence-based review support using the "
    "retrieved documents. The result is not a substitute for authorized "
    "engineering, quality, safety, certification, or release approval."
)


def print_header(title: str) -> None:
    """Print a consistent console heading."""

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_report(report: DiagnosticReport) -> None:
    """Print a readable diagnostic report."""

    print_header("STARTUP DIAGNOSTIC RESULTS")

    for result in report.results:
        print(f"[{result.status:<7}] {result.name}")
        print(f"          {result.message}")

    print_header("STARTUP SUMMARY")

    passed_count = sum(
        1 for result in report.results if result.status == "PASS"
    )
    warning_count = sum(
        1 for result in report.results if result.status == "WARNING"
    )
    failed_count = sum(
        1 for result in report.results if result.status == "FAIL"
    )

    print(f"Passed   : {passed_count}")
    print(f"Warnings : {warning_count}")
    print(f"Failed   : {failed_count}")

    if report.can_start:
        print()
        print("REQTRACE AI startup prerequisites are available.")
        print("Phase 2 diagnostics completed without critical failures.")
    else:
        print()
        print("REQTRACE AI cannot start because a critical check failed.")
        print("Correct the failed checks and run the diagnostics again.")


def main() -> int:
    """Load configuration, initialize logging, and run diagnostics."""

    print_header("REQTRACE AI")
    print("Code Observation & FRD Evidence Analysis")
    print()
    print(NOTICE)

    try:
        settings = get_settings()
    except ValidationError as exc:
        print_header("CONFIGURATION ERROR")
        print(
            "REQTRACE AI configuration is invalid. "
            "Review the following settings:"
        )

        for error in exc.errors():
            field_name = ".".join(
                str(location)
                for location in error.get("loc", ())
            )
            message = error.get(
                "msg",
                "Unknown configuration error.",
            )
            print(f"- {field_name}: {message}")

        return 1

    logger = configure_logging(
        log_level=settings.log_level,
        log_file_path=settings.log_file_path,
    )

    logger.info("REQTRACE AI startup diagnostics started.")
    logger.info(
        "Safe startup configuration: %s",
        settings.safe_summary(),
    )

    print_header("CONFIGURATION")
    print(f"Application        : {settings.app_name}")
    print(f"Environment        : {settings.app_environment}")
    print(f"Ollama URL         : {settings.ollama_base_url}")
    print(f"Generation model   : {settings.ollama_llm_model}")
    print(f"Embedding model    : {settings.ollama_embed_model}")
    print(f"ChromaDB path      : {settings.chroma_db_path}")
    print(f"Log level          : {settings.log_level}")
    print(f"Hybrid retrieval   : {settings.enable_hybrid_retrieval}")
    print(f"Reranking          : {settings.enable_reranking}")
    print(f"Query expansion    : {settings.enable_query_expansion}")

    report = run_startup_diagnostics(settings)
    print_report(report)

    if report.can_start:
        logger.info(
            "REQTRACE AI startup diagnostics completed successfully."
        )
        return 0

    logger.error(
        "REQTRACE AI startup diagnostics found critical failures."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

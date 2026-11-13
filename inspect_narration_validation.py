"""
REQTRACE AI Phase 14C live narration validation.

This command generates one raw Ollama response and validates it.

The raw response is printed only as diagnostic content.
It is not yet connected to the final answer delivery pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.answer_pipeline import (
    build_grounded_answer_package,
)
from src.narration_client import (
    NarrationClientConfig,
    OllamaNarrationClient,
)
from src.narration_prompt import (
    build_narration_prompt,
)
from src.narration_validation import (
    validate_narration,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 14C "
            "live narration validation."
        )
    )

    parser.add_argument(
        "query",
        help="Observation ID or engineering query.",
    )

    parser.add_argument(
        "--closure",
        action="store_true",
        help="Also evaluate closure readiness.",
    )

    parser.add_argument(
        "--model",
        default="llama3.2:1b",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=350,
    )

    return parser.parse_args()


def main() -> int:
    """Generate and validate one narration."""

    args = parse_arguments()
    settings = get_settings()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    package = build_grounded_answer_package(
        args.query,
        settings,
        base_directory,
        evaluate_closure=(
            args.closure
        ),
    )

    prompt = build_narration_prompt(
        query=args.query,
        package=package,
    )

    config = NarrationClientConfig(
        base_url=(
            settings.ollama_base_url
        ),
        model=args.model,
        connect_timeout_seconds=(
            settings
            .ollama_connect_timeout_seconds
        ),
        read_timeout_seconds=(
            settings.llm_timeout_seconds
        ),
        maximum_output_tokens=(
            args.max_tokens
        ),
    )

    with OllamaNarrationClient(
        config
    ) as client:
        response = client.generate(
            prompt,
            attempt_number=1,
            is_repair_attempt=False,
        )

    validation = validate_narration(
        request=prompt.request,
        response=response,
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 14C "
        "NARRATION VALIDATION"
    )
    print("=" * 78)

    print(
        f"Query             : {args.query}"
    )

    print(
        f"Model             : {response.model}"
    )

    print(
        f"Validation passed : {validation.passed}"
    )

    print(
        f"Errors            : {validation.error_count}"
    )

    print(
        f"Warnings          : {validation.warning_count}"
    )

    print()
    print("Validation Checks")
    print("-" * 78)

    for check in validation.checks:
        status = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        print(
            f"[{status}] {check.rule.value}"
        )

        print(
            f"       {check.message}"
        )

    if validation.issues:
        print()
        print("Validation Issues")
        print("-" * 78)

        for issue in validation.issues:
            print(
                f"[{issue.severity.value}] "
                f"{issue.rule.value}"
            )

            print(
                f"        {issue.message}"
            )

    print()
    print("RAW MODEL OUTPUT - DIAGNOSTIC ONLY")
    print("-" * 78)
    print(
        response.text
    )

    print()
    print(
        "This command does not deliver model output "
        "as the final engineering answer."
    )

    return (
        0
        if validation.passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
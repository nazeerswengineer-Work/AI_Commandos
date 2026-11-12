"""
REQTRACE AI Phase 13E adversarial validation diagnostic.

No LLM is invoked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.answer_pipeline import (
    build_grounded_answer_artifacts,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 13E "
            "adversarial answer-package validation."
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

    return parser.parse_args()


def main() -> int:
    """Build and inspect adversarial validation."""

    args = parse_arguments()
    settings = get_settings()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    (
        package,
        _,
        validation,
    ) = build_grounded_answer_artifacts(
        args.query,
        settings,
        base_directory,
        evaluate_closure=(
            args.closure
        ),
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 13E "
        "ADVERSARIAL VALIDATION"
    )
    print("=" * 78)

    print(
        f"Query             : {args.query}"
    )

    print(
        f"Assessment        : "
        f"{package.decision.assessment}"
    )

    print(
        f"Confidence        : "
        f"{package.decision.confidence}"
    )

    print(
        f"Closure           : "
        f"{package.decision.closure_status}"
    )

    print(
        f"Validation passed : "
        f"{validation.passed}"
    )

    print(
        f"Errors            : "
        f"{validation.error_count}"
    )

    print(
        f"Warnings          : "
        f"{validation.warning_count}"
    )

    print()
    print("Adversarial Checks")
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
        print("Issues")
        print("-" * 78)

        for issue in validation.issues:
            print(
                f"[{issue.severity.value}] "
                f"{issue.rule.value}"
            )

            print(
                f"        {issue.message}"
            )

    if (
        validation
        .untrusted_evidence_instructions
    ):
        print()
        print("Untrusted Source Instructions")
        print("-" * 78)

        for detection in (
            validation
            .untrusted_evidence_instructions
        ):
            print(
                f"[{detection.citation_id}] "
                f"{detection.evidence_id}"
            )

            print(
                f"  Fragment: "
                f"{detection.excerpt_fragment}"
            )

    print()
    print(
        "Phase 13E validates deterministic package safety. "
        "No LLM narration was generated."
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
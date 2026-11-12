"""
REQTRACE AI Phase 13B grounding-diagnostic command.

Runs:

- Phase 11 final-context retrieval
- Phase 12 deterministic decision
- Phase 13A citation construction
- Phase 13B grounding validation

No LLM is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.citations import (
    build_citation_collection,
)
from src.decision_engine import (
    evaluate_engineering_decision,
)
from src.grounding import (
    validate_grounding,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 13B "
            "grounding validation."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Observation ID or engineering query."
        ),
    )

    parser.add_argument(
        "--closure",
        action="store_true",
        help=(
            "Also evaluate and validate closure readiness."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Raise an error when grounding validation fails."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Validate grounding for one query."""

    args = parse_arguments()
    settings = get_settings()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    context = retrieve_final_context(
        args.query,
        settings,
        base_directory,
    )

    decision = evaluate_engineering_decision(
        context,
        evaluate_closure=(
            args.closure
        ),
    )

    citations = build_citation_collection(
        context
    )

    grounding = validate_grounding(
        context=context,
        decision=decision,
        citations=citations,
        strict=args.strict,
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 13B "
        "GROUNDING VALIDATION"
    )
    print("=" * 78)

    print(
        f"Query           : {args.query}"
    )

    print(
        f"Assessment      : "
        f"{decision.assessment.assessment.value}"
    )

    print(
        f"Confidence      : "
        f"{decision.confidence.confidence.value}"
    )

    print(
        f"Closure         : "
        f"{decision.closure_status.value}"
    )

    print(
        f"Grounding passed: "
        f"{grounding.passed}"
    )

    print(
        f"Checks          : "
        f"{len(grounding.checks)}"
    )

    print(
        f"Errors          : "
        f"{grounding.error_count}"
    )

    print(
        f"Warnings        : "
        f"{grounding.warning_count}"
    )

    print()
    print("Grounding Checks")
    print("-" * 78)

    for check in grounding.checks:
        status = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{check.rule.value}"
        )

        print(
            f"       {check.message}"
        )

    if grounding.issues:
        print()
        print("Grounding Issues")
        print("-" * 78)

        for issue in grounding.issues:
            print(
                f"[{issue.severity.value}] "
                f"{issue.rule.value}"
            )

            print(
                f"        {issue.message}"
            )

    print()
    print("Citation Allowlist")
    print("-" * 78)

    for citation in citations.citations:
        print(
            f"[{citation.citation_id}] "
            f"{citation.display_label}"
        )

    print()
    print(
        "Phase 13B validates grounding only. "
        "No natural-language answer was generated."
    )

    return (
        0
        if grounding.passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
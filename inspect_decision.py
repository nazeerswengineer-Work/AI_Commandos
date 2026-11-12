"""
REQTRACE AI Phase 12E complete deterministic decision diagnostic.

No LLM is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.decision_engine import (
    evaluate_engineering_decision,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 12E complete "
            "deterministic decision."
        )
    )

    parser.add_argument(
        "query",
        help="Observation ID or engineering query.",
    )

    parser.add_argument(
        "--closure",
        action="store_true",
        help=(
            "Also evaluate closure readiness. "
            "Closure is otherwise NOT ASSESSED."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run complete deterministic decision."""

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

    result = evaluate_engineering_decision(
        context,
        evaluate_closure=(
            args.closure
        ),
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 12E "
        "DETERMINISTIC DECISION"
    )
    print("=" * 78)

    print(
        f"Query           : {args.query}"
    )

    print(
        f"Assessment      : "
        f"{result.assessment.assessment.value}"
    )

    print(
        f"Assessment rule : "
        f"{result.assessment.applied_rule.value}"
    )

    print(
        f"Confidence      : "
        f"{result.confidence.confidence.value}"
    )

    print(
        f"Confidence rule : "
        f"{result.confidence.applied_rule.value}"
    )

    print(
        f"Closure status  : "
        f"{result.closure_status.value}"
    )

    print(
        f"Closure rule    : "
        f"{result.closure_rule.value}"
    )

    print()
    print("Assessment Reason")
    print("-" * 78)
    print(
        result.assessment.reason
    )

    print()
    print("Confidence Reason")
    print("-" * 78)
    print(
        result.confidence.reason
    )

    print()
    print("Closure Reason")
    print("-" * 78)
    print(
        result.closure_reason
    )

    print()
    print("Supporting Items")
    print("-" * 78)

    if result.supporting_items:
        for item in result.supporting_items:
            print(
                f"  [SUPPORT] {item}"
            )
    else:
        print(
            "  No decisive supporting item recorded."
        )

    print()
    print("Blocking Items")
    print("-" * 78)

    if result.blocking_items:
        for item in result.blocking_items:
            print(
                f"  [BLOCKING] {item}"
            )
    else:
        print(
            "  No deterministic blocker recorded."
        )

    print()
    print("Evidence Provenance")
    print("-" * 78)

    for evidence_id in result.evidence_ids:
        print(
            f"  {evidence_id}"
        )

    print()
    print(
        "REQTRACE AI provides evidence-derived decision support. "
        "Final engineering approval and closure authorization "
        "remain with the responsible human authority."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
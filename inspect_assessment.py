"""
REQTRACE AI Phase 12C deterministic assessment diagnostic.

Runs the validated retrieval pipeline and deterministic assessment engine.

No LLM generation is used.
Closure readiness is not evaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.assessment_engine import (
    assess_observation,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 12C "
            "deterministic observation assessment."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Observation ID or engineering query."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run deterministic assessment."""

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

    result = assess_observation(
        context
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 12C "
        "DETERMINISTIC ASSESSMENT"
    )
    print("=" * 78)

    print(
        f"Query       : {args.query}"
    )

    print(
        f"Assessment  : "
        f"{result.assessment.value}"
    )

    print(
        f"Applied rule: "
        f"{result.applied_rule.value}"
    )

    print(
        f"Completeness: "
        f"{result.conditions.completeness.classification.value}"
    )

    print(
        f"Reason      : "
        f"{result.reason}"
    )

    print(
        f"Requirements: "
        f"{result.requirement_ids}"
    )

    print(
        f"Observations: "
        f"{result.observation_ids}"
    )

    print(
        f"Tests       : "
        f"{result.test_case_ids}"
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
            "  No decisive supporting item was recorded."
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
            "  No assessment blocker was recorded."
        )

    print()
    print("Evidence IDs")
    print("-" * 78)

    for evidence_id in result.evidence_ids:
        print(
            f"  {evidence_id}"
        )

    print()
    print(
        "Phase 12C does not evaluate observation "
        "closure readiness."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
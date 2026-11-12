"""
REQTRACE AI Phase 12B evidence-condition diagnostic.

Runs the validated Phase 11 retrieval pipeline and reports deterministic
evidence conditions.

No final observation assessment is calculated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.evidence_conditions import (
    detect_evidence_conditions,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 12B "
            "evidence-condition detection."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Observation or engineering query."
        ),
    )

    return parser.parse_args()


def main() -> int:
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

    result = detect_evidence_conditions(
        context
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 12B "
        "EVIDENCE CONDITIONS"
    )
    print("=" * 78)

    print(
        f"Query        : {args.query}"
    )

    print(
        f"Completeness : "
        f"{result.completeness.classification.value}"
    )

    print(
        f"Requirements : "
        f"{result.completeness.requirement_ids}"
    )

    print(
        f"Observations : "
        f"{result.completeness.observation_ids}"
    )

    print(
        f"Tests        : "
        f"{result.completeness.test_case_ids}"
    )

    print(
        f"Revisions    : "
        f"{result.completeness.document_revisions}"
    )

    print()
    print("Detected Conditions")
    print("-" * 78)

    if not result.conditions:
        print(
            "No Phase 12B evidence conditions detected."
        )

    for condition in result.conditions:
        print()
        print(
            f"[{condition.condition_type.value}]"
        )

        print(
            f"  {condition.message}"
        )

        if condition.requirement_ids:
            print(
                f"  REQ : "
                f"{condition.requirement_ids}"
            )

        if condition.observation_ids:
            print(
                f"  OBS : "
                f"{condition.observation_ids}"
            )

        if condition.test_case_ids:
            print(
                f"  TEST: "
                f"{condition.test_case_ids}"
            )

    print()
    print(
        "Phase 12B detects evidence conditions only."
    )

    print(
        "SUPPORTED / NOT SUPPORTED / INCONCLUSIVE "
        "will be determined in Phase 12C."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
"""
REQTRACE AI Phase 12A evidence-completeness diagnostic.

Runs the validated Phase 11 retrieval pipeline and classifies evidence
completeness without making an engineering assessment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.evidence_completeness import (
    evaluate_evidence_completeness,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 12A "
            "evidence-completeness analysis."
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

    result = (
        evaluate_evidence_completeness(
            context
        )
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 12A "
        "EVIDENCE COMPLETENESS"
    )
    print("=" * 78)

    print(
        f"Query          : {args.query}"
    )

    print(
        f"Classification : "
        f"{result.classification.value}"
    )

    print(
        f"Context records: "
        f"{result.inventory.record_count}"
    )

    print(
        f"Document types : "
        f"{result.inventory.document_types}"
    )

    print(
        f"Requirements   : "
        f"{result.requirement_ids}"
    )

    print(
        f"Observations   : "
        f"{result.observation_ids}"
    )

    print(
        f"Tests          : "
        f"{result.test_case_ids}"
    )

    print(
        f"Revisions      : "
        f"{result.document_revisions}"
    )

    print(
        f"Test states    : "
        f"{tuple(state.value for state in result.test_execution_states)}"
    )

    print(
        f"Assessment-ready minimum: "
        f"{result.can_run_assessment_rules}"
    )

    print()
    print("Supporting Evidence State")
    print("-" * 78)

    if result.supporting_notes:
        for note in result.supporting_notes:
            print(
                f"  [AVAILABLE] {note}"
            )
    else:
        print(
            "  No supporting evidence state was identified."
        )

    print()
    print("Missing or Blocking Evidence")
    print("-" * 78)

    if result.blocking_items:
        for item in result.blocking_items:
            print(
                f"  [BLOCKING] {item}"
            )
    else:
        print(
            "  No completeness blockers were identified."
        )

    print()
    print(
        "Phase 12A does not calculate observation validity "
        "or closure readiness."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
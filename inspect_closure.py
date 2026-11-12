"""
REQTRACE AI Phase 12D closure-readiness diagnostic.

No LLM is used.

Closure readiness is reported separately from the original
observation assessment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.closure_engine import (
    evaluate_closure_readiness,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 12D "
            "closure-readiness evaluation."
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

    result = evaluate_closure_readiness(
        context
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 12D "
        "CLOSURE READINESS"
    )
    print("=" * 78)

    print(
        f"Query            : {args.query}"
    )

    print(
        f"Assessment       : "
        f"{result.assessment.assessment.value}"
    )

    print(
        f"Assessment rule  : "
        f"{result.assessment.applied_rule.value}"
    )

    print(
        f"Closure status   : "
        f"{result.status.value}"
    )

    print(
        f"Closure rule     : "
        f"{result.applied_rule.value}"
    )

    print(
        f"Reason           : "
        f"{result.reason}"
    )

    print()
    print("Closure Evidence State")
    print("-" * 78)

    state = result.evidence_state

    print(
        f"Corrective action          : "
        f"{state.corrective_action_available}"
    )

    print(
        f"Corrected code             : "
        f"{state.corrected_code_available}"
    )

    print(
        f"Post-correction test       : "
        f"{state.post_correction_test_available}"
    )

    print(
        f"Independent verification   : "
        f"{state.independent_verification_available}"
    )

    print(
        f"Explicit closure limitation: "
        f"{state.explicit_closure_limitation}"
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
            "  No closure-evidence blocker detected."
        )

    print()
    print(
        "Closure readiness is evidence-derived only. "
        "Final engineering authorization remains with "
        "the responsible human authority."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
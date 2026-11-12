"""
REQTRACE AI Phase 13D deterministic answer-package inspection.

No LLM is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from src.answer_pipeline import (
    build_grounded_answer_package,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 13D "
            "deterministic answer package."
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
            "Also evaluate closure readiness."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the complete package as JSON."
        ),
    )

    parser.add_argument(
        "--excerpt-length",
        type=int,
        default=500,
        help=(
            "Maximum characters in each source excerpt."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Build and display the answer package."""

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
        maximum_excerpt_characters=(
            args.excerpt_length
        ),
    )

    if args.json:
        print(
            json.dumps(
                package.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
        )

        return 0

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 13D "
        "DETERMINISTIC ANSWER PACKAGE"
    )
    print("=" * 78)

    print(
        f"Query           : {args.query}"
    )

    print(
        f"Assessment      : "
        f"{package.decision.assessment}"
    )

    print(
        f"Assessment rule : "
        f"{package.decision.assessment_rule}"
    )

    print(
        f"Confidence      : "
        f"{package.decision.confidence}"
    )

    print(
        f"Closure         : "
        f"{package.decision.closure_status}"
    )

    print(
        f"Grounding       : "
        f"{package.grounding_passed}"
    )

    print(
        f"Evidence records: "
        f"{package.evidence_record_count}"
    )

    print()
    print("Deterministic Engineering Answer")
    print("-" * 78)

    print(
        package.deterministic_text
    )

    print()
    print("Allowed Claims")
    print("-" * 78)

    for claim in package.allowed_claims:
        print(
            f"[{claim.claim_id}] "
            f"{claim.statement}"
        )

    print()
    print("Prohibited Claims")
    print("-" * 78)

    for prohibition in (
        package.prohibited_claims
    ):
        print(
            f"[{prohibition.prohibition_id}] "
            f"{prohibition.statement}"
        )

    print()
    print(
        "Phase 13D produced deterministic output only. "
        "No LLM narration was generated."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
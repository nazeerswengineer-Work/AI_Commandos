"""
REQTRACE AI Phase 13C answer-contract inspection.

Builds:

- Phase 11 final context
- Phase 12 deterministic decision
- Phase 13A citations
- Phase 13B grounding validation
- Phase 13C safe answer contract

No LLM is invoked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from src.answer_contract import (
    build_answer_contract,
)
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
            "REQTRACE AI Phase 13C "
            "safe answer-contract inspection."
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
            "Print the complete contract as JSON."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Build and display the safe answer contract."""

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
        strict=True,
    )

    contract = build_answer_contract(
        decision=decision,
        citations=citations,
        grounding=grounding,
    )

    if args.json:
        print(
            json.dumps(
                contract.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
        )

        return 0

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 13C "
        "SAFE ANSWER CONTRACT"
    )
    print("=" * 78)

    print(
        f"Query          : {args.query}"
    )

    print(
        f"Assessment     : "
        f"{contract.assessment.value}"
    )

    print(
        f"Assessment rule: "
        f"{contract.assessment_rule.value}"
    )

    print(
        f"Confidence     : "
        f"{contract.confidence.value}"
    )

    print(
        f"Closure        : "
        f"{contract.closure_status.value}"
    )

    print(
        f"Grounding      : "
        f"{contract.grounding_passed}"
    )

    print()
    print("Engineering-ID Allowlist")
    print("-" * 78)

    for engineering_id in (
        contract.engineering_id_allowlist
    ):
        print(
            f"  {engineering_id}"
        )

    print()
    print("Citation Allowlist")
    print("-" * 78)

    for citation in (
        contract.citations.citations
    ):
        print(
            f"  [{citation.citation_id}] "
            f"{citation.display_label}"
        )

    print()
    print("Allowed Claims")
    print("-" * 78)

    for claim in contract.allowed_claims:
        print()
        print(
            f"[{claim.claim_id}] "
            f"{claim.category.value}"
        )

        print(
            f"  {claim.statement}"
        )

        if claim.citation_ids:
            print(
                f"  Citations: "
                f"{claim.citation_ids}"
            )

        if claim.deterministic_rule:
            print(
                f"  Rule: "
                f"{claim.deterministic_rule}"
            )

    print()
    print("Prohibited Claims")
    print("-" * 78)

    for prohibition in (
        contract.prohibited_claims
    ):
        print()
        print(
            f"[{prohibition.prohibition_id}] "
            f"{prohibition.category.value}"
        )

        print(
            f"  {prohibition.statement}"
        )

        print(
            f"  Reason: "
            f"{prohibition.reason}"
        )

    print()
    print(
        "Phase 13C created a grounded contract only. "
        "No LLM narration was generated."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
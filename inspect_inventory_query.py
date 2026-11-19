"""
Inspect deterministic Phase 16I inventory-query parsing and execution.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from src.evidence_inventory import (
    answer_inventory_query,
    load_inventory_summary,
)

from src.generalized_qa_models import (
    GeneralizedQAResponse,
)

from src.inventory_query_parser import (
    parse_inventory_query,
)

from src.evidence_inventory import (
    answer_inventory_query,
    answer_requirement_evidence_availability,
    load_inventory_summary,
)
from src.query_intent import (
    QueryIntent,
)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line query."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect deterministic generalized "
            "evidence-inventory QA."
        )
    )

    parser.add_argument(
        "query"
    )

    return parser.parse_args()


def main() -> int:
    """Parse and execute one inventory query."""

    args = parse_arguments()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    parsed = parse_inventory_query(
        args.query
    )

    summary = load_inventory_summary(
        base_directory
        / "data"
        / "ingestion_manifest.json"
    )

    if (
        parsed.intent_result.intent
        == QueryIntent.EVIDENCE_AVAILABILITY
    ):
        if not parsed.evidence_type:
            raise RuntimeError(
                "Evidence type was not identified."
            )

        inventory_answer = (
            answer_requirement_evidence_availability(
                summary,
                evidence_type=(
                    parsed.evidence_type
                ),
            )
        )

    else:
        inventory_answer = (
            answer_inventory_query(
                summary,
                parsed.inventory_query,
            )
        )

    response = (
        GeneralizedQAResponse
        .from_inventory_answer(
            original_query=args.query,

            intent_result=(
                parsed.intent_result
            ),

            inventory_answer=(
                inventory_answer
            ),
        )
    )

    print(
        json.dumps(
            response.to_dict(),
            indent=2,
            ensure_ascii=True,
        )
    )

    print()

    print(
        response.answer
    )

    print()

    print(
        response.source_statement
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
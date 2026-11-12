"""
REQTRACE AI Phase 13A citation inspection.

Runs the validated retrieval pipeline and converts the final evidence
context into structured citations.

No LLM is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from src.citations import (
    build_citation_collection,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 13A "
            "citation and provenance inspection."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Observation ID or engineering query."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print citation collection as JSON."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Build and print citation provenance."""

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

    collection = (
        build_citation_collection(
            context
        )
    )

    if args.json:
        print(
            json.dumps(
                collection.to_dict(),
                indent=2,
                ensure_ascii=True,
            )
        )

        return 0

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 13A "
        "CITATION PROVENANCE"
    )
    print("=" * 78)

    print(
        f"Query          : {args.query}"
    )

    print(
        f"Citation count : "
        f"{collection.record_count}"
    )

    print()
    print("Validated Citations")
    print("-" * 78)

    for citation in collection.citations:
        print()
        print(
            f"[{citation.citation_id}] "
            f"{citation.display_label}"
        )

        print(
            f"  Evidence ID : "
            f"{citation.evidence_id}"
        )

        print(
            f"  Source type : "
            f"{citation.source_type.value}"
        )

        print(
            f"  Location    : "
            f"{citation.provenance_location}"
        )

        if citation.requirement_id:
            print(
                f"  REQ         : "
                f"{citation.requirement_id}"
            )

        if citation.observation_id:
            print(
                f"  OBS         : "
                f"{citation.observation_id}"
            )

        if citation.test_case_id:
            print(
                f"  TEST        : "
                f"{citation.test_case_id}"
            )

        if citation.result_status:
            print(
                f"  Result      : "
                f"{citation.result_status}"
            )

        if citation.content_hash:
            print(
                f"  Content hash: "
                f"{citation.content_hash}"
            )

    print()
    print(
        "Phase 13A validates provenance only. "
        "Claim grounding and answer generation "
        "are not performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
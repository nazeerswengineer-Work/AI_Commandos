"""
REQTRACE AI Phase 11E retrieval benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import get_settings
from src.retrieval_evaluation import (
    RetrievalExpectation,
    build_evaluation_report,
    evaluate_context,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def main() -> int:

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    evaluation_path = (
        base_directory
        / "evaluation"
        / "retrieval_cases.json"
    )

    raw_cases = json.loads(
        evaluation_path.read_text(
            encoding="utf-8"
        )
    )

    expectations = tuple(
        RetrievalExpectation(
            case_id=item["case_id"],
            query=item["query"],
            requirement_ids=tuple(
                item.get(
                    "requirement_ids",
                    [],
                )
            ),
            observation_ids=tuple(
                item.get(
                    "observation_ids",
                    [],
                )
            ),
            test_case_ids=tuple(
                item.get(
                    "test_case_ids",
                    [],
                )
            ),
            required_document_types=tuple(
                item.get(
                    "required_document_types",
                    [],
                )
            ),
        )
        for item in raw_cases
    )

    settings = get_settings()

    evaluation_results = []

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 11E "
        "RETRIEVAL EVALUATION"
    )
    print("=" * 78)

    for expectation in expectations:

        context = retrieve_final_context(
            expectation.query,
            settings,
            base_directory,
        )

        result = evaluate_context(
            context,
            expectation,
        )

        evaluation_results.append(
            result
        )

        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print()
        print(
            f"{expectation.case_id}: "
            f"{status}"
        )

        print(
            f"  Query               : "
            f"{expectation.query}"
        )

        print(
            f"  Requirement coverage: "
            f"{result.requirement_coverage:.2f}"
        )

        print(
            f"  Observation coverage: "
            f"{result.observation_coverage:.2f}"
        )

        print(
            f"  Test coverage       : "
            f"{result.test_coverage:.2f}"
        )

        print(
            f"  Type coverage       : "
            f"{result.type_coverage:.2f}"
        )

        print(
            f"  Precision@K         : "
            f"{result.precision_at_k:.2f}"
        )

        print(
            f"  Recall@K            : "
            f"{result.recall_at_k:.2f}"
        )

        print(
            f"  Reciprocal Rank     : "
            f"{result.reciprocal_rank:.2f}"
        )

    report = build_evaluation_report(
        evaluation_results
    )

    print()
    print("=" * 78)
    print("AGGREGATE RESULTS")
    print("=" * 78)

    print(
        f"Passed cases        : "
        f"{report.passed_cases}"
    )

    print(
        f"Failed cases        : "
        f"{report.failed_cases}"
    )

    print(
        f"Mean Precision@K    : "
        f"{report.mean_precision_at_k:.3f}"
    )

    print(
        f"Mean Recall@K       : "
        f"{report.mean_recall_at_k:.3f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{report.mean_reciprocal_rank:.3f}"
    )

    print(
        f"Mean Type Coverage  : "
        f"{report.mean_type_coverage:.3f}"
    )

    return (
        0
        if report.all_passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
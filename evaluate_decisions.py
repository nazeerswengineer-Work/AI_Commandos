"""
REQTRACE AI Phase 12E six-scenario deterministic evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import get_settings
from src.decision_engine import (
    evaluate_engineering_decision,
)
from src.retrieval_pipeline import (
    retrieve_final_context,
)


def main() -> int:
    """Evaluate the six deterministic decision scenarios."""

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    cases_path = (
        base_directory
        / "evaluation"
        / "decision_cases.json"
    )

    cases = json.loads(
        cases_path.read_text(
            encoding="utf-8"
        )
    )

    settings = get_settings()

    passed = 0
    failed = 0

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 12E "
        "DECISION EVALUATION"
    )
    print("=" * 78)

    for case in cases:
        context = retrieve_final_context(
            case["query"],
            settings,
            base_directory,
        )

        result = evaluate_engineering_decision(
            context,
            evaluate_closure=bool(
                case[
                    "evaluate_closure"
                ]
            ),
        )

        actual_assessment = (
            result.assessment
            .assessment
            .value
        )

        actual_confidence = (
            result.confidence
            .confidence
            .value
        )

        actual_closure = (
            result.closure_status
            .value
        )

        case_passed = bool(
            actual_assessment
            == case["assessment"]
            and actual_confidence
            == case["confidence"]
            and actual_closure
            == case["closure_status"]
        )

        if case_passed:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print()
        print(
            f"{case['case_id']}: {status}"
        )

        print(
            f"  Query      : "
            f"{case['query']}"
        )

        print(
            f"  Assessment : "
            f"{actual_assessment}"
        )

        print(
            f"  Confidence : "
            f"{actual_confidence}"
        )

        print(
            f"  Closure    : "
            f"{actual_closure}"
        )

        if not case_passed:
            print(
                f"  Expected   : "
                f"{case['assessment']} / "
                f"{case['confidence']} / "
                f"{case['closure_status']}"
            )

    print()
    print("=" * 78)
    print("AGGREGATE RESULTS")
    print("=" * 78)

    print(
        f"Passed cases: {passed}"
    )

    print(
        f"Failed cases: {failed}"
    )

    return (
        0
        if failed == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
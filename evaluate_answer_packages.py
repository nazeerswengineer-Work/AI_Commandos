"""
REQTRACE AI Phase 13D six-scenario answer-package evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import get_settings
from src.answer_pipeline import (
    build_grounded_answer_package,
)


def main() -> int:
    """Evaluate all deterministic answer packages."""

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    cases_path = (
        base_directory
        / "evaluation"
        / "answer_package_cases.json"
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
        "REQTRACE AI - PHASE 13D "
        "ANSWER-PACKAGE EVALUATION"
    )
    print("=" * 78)

    for case in cases:
        package = (
            build_grounded_answer_package(
                case["query"],
                settings,
                base_directory,
                evaluate_closure=bool(
                    case[
                        "evaluate_closure"
                    ]
                ),
            )
        )

        actual_types = {
            section.document_type
            for section
            in package.evidence_sections
            if not section.is_empty
        }

        required_types = set(
            case["required_types"]
        )

        type_coverage_passed = (
            required_types
            .issubset(actual_types)
        )

        decision_passed = bool(
            package.decision.assessment
            == case["assessment"]
            and package.decision.confidence
            == case["confidence"]
            and package.decision.closure_status
            == case["closure_status"]
        )

        package_passed = bool(
            decision_passed
            and type_coverage_passed
            and package.grounding_passed
            and package.grounding_error_count
            == 0
            and package.evidence_record_count
            > 0
        )

        if package_passed:
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
            f"  Query       : "
            f"{case['query']}"
        )

        print(
            f"  Assessment  : "
            f"{package.decision.assessment}"
        )

        print(
            f"  Confidence  : "
            f"{package.decision.confidence}"
        )

        print(
            f"  Closure     : "
            f"{package.decision.closure_status}"
        )

        print(
            f"  Types       : "
            f"{tuple(sorted(actual_types))}"
        )

        print(
            f"  Grounding   : "
            f"{package.grounding_passed}"
        )

        print(
            f"  Citations   : "
            f"{len(package.citation_allowlist)}"
        )

        if not package_passed:
            print(
                f"  Expected    : "
                f"{case['assessment']} / "
                f"{case['confidence']} / "
                f"{case['closure_status']}"
            )

            print(
                f"  Required    : "
                f"{tuple(case['required_types'])}"
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
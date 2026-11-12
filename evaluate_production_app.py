"""
REQTRACE AI Phase 15E final production evaluation.

Runs all six benchmark scenarios through the unified Phase 15
production pipeline in deterministic-only mode.

The purpose is to validate the production architecture independently
from optional Ollama latency.

Validation covers:

- deterministic decisions
- confidence
- closure behavior
- automatic closure intent
- engineering-ID preservation
- traceability protection
- evidence-type coverage
- revision-conflict evidence preservation
- grounding
- safe delivery
- fallback integrity
- provenance
- request timing
"""

from __future__ import annotations

import json
from pathlib import Path

from config import get_settings

from src.narration_models import (
    NarrationStatus,
)

from src.production_pipeline import (
    ProductionPipelineConfig,
    ReqTraceProductionPipeline,
    detect_closure_intent,
)


def main() -> int:

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    cases_path = (
        base_directory
        / "evaluation"
        / "production_cases.json"
    )

    cases = json.loads(
        cases_path.read_text(
            encoding="utf-8"
        )
    )

    pipeline = ReqTraceProductionPipeline(
        settings=get_settings(),
        base_directory=base_directory,
        config=ProductionPipelineConfig(
            narration_enabled=False,
            allow_repair_attempt=False,
            context_window=4096,
            maximum_output_tokens=450,
        ),
    )

    passed_cases = 0
    failed_cases = 0

    total_time = 0.0
    total_deterministic_time = 0.0

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 15E "
        "FINAL APPLICATION EVALUATION"
    )
    print("=" * 78)

    for case in cases:

        query = case["query"]

        closure_intent = (
            detect_closure_intent(
                query
            )
        )

        result = pipeline.run(
            query,
            evaluate_closure=(
                case[
                    "evaluate_closure"
                ]
            ),
        )

        # -------------------------------------------------------------
        # Decision preservation
        # -------------------------------------------------------------

        decision_ok = bool(
            result.assessment
            == case["assessment"]
            and result.confidence
            == case["confidence"]
            and result.closure_status
            == case["closure_status"]
        )

        # -------------------------------------------------------------
        # Automatic closure behavior
        # -------------------------------------------------------------

        automatic_closure_ok = True

        if case.get(
            "automatic_closure_expected",
            False,
        ):
            automatic_closure_ok = (
                closure_intent
                and result.closure_status
                != "NOT ASSESSED"
            )

        # -------------------------------------------------------------
        # Authoritative engineering IDs
        # -------------------------------------------------------------

        authoritative_ids = set(
            result.requirement_ids
            + result.observation_ids
            + result.test_case_ids
        )

        required_ids = set(
            case.get(
                "required_ids",
                [],
            )
        )

        required_ids_ok = (
            required_ids
            .issubset(
                authoritative_ids
            )
        )

        forbidden_ids = set(
            case.get(
                "forbidden_authoritative_ids",
                [],
            )
        )

        forbidden_ids_ok = not bool(
            authoritative_ids
            & forbidden_ids
        )

        # -------------------------------------------------------------
        # Evidence coverage
        # -------------------------------------------------------------

        represented_types = {
            citation.document_type
            for citation
            in result.citations
        }

        required_types = set(
            case.get(
                "required_types",
                [],
            )
        )

        evidence_coverage_ok = (
            required_types
            .issubset(
                represented_types
            )
        )

        # -------------------------------------------------------------
        # Revision-conflict preservation
        # -------------------------------------------------------------

        minimum_frd = int(
            case.get(
                "minimum_frd_citations",
                0,
            )
        )

        frd_count = sum(
            1
            for citation
            in result.citations
            if (
                citation.document_type
                == "frd"
            )
        )

        revision_evidence_ok = bool(
            frd_count
            >= minimum_frd
        )

        # -------------------------------------------------------------
        # Provenance
        # -------------------------------------------------------------

        provenance_ok = bool(
            result.citations
            and all(
                citation.citation_id
                and citation.evidence_id
                and citation.provenance_location

                for citation
                in result.citations
            )
        )

        # -------------------------------------------------------------
        # Safety
        # -------------------------------------------------------------

        grounding_ok = bool(
            result.grounding_passed
        )

        safe_delivery_ok = bool(
            result.safe_delivery
        )

        deterministic_delivery_ok = bool(
            result.delivery_status
            == NarrationStatus
            .DETERMINISTIC_FALLBACK
            and result.used_fallback
            and not result.generated_by_llm
            and result.generation_attempts
            == 0
        )

        # -------------------------------------------------------------
        # Timing
        # -------------------------------------------------------------

        timing_ok = bool(
            result.timing.deterministic_seconds
            >= 0.0
            and result.timing.narration_seconds
            >= 0.0
            and result.timing.total_seconds
            >= result.timing.deterministic_seconds
        )

        total_time += (
            result.timing
            .total_seconds
        )

        total_deterministic_time += (
            result.timing
            .deterministic_seconds
        )

        # -------------------------------------------------------------
        # Final case result
        # -------------------------------------------------------------

        gates = {
            "decision": decision_ok,
            "automatic_closure": (
                automatic_closure_ok
            ),
            "required_ids": (
                required_ids_ok
            ),
            "forbidden_ids": (
                forbidden_ids_ok
            ),
            "evidence_coverage": (
                evidence_coverage_ok
            ),
            "revision_evidence": (
                revision_evidence_ok
            ),
            "provenance": (
                provenance_ok
            ),
            "grounding": (
                grounding_ok
            ),
            "safe_delivery": (
                safe_delivery_ok
            ),
            "deterministic_delivery": (
                deterministic_delivery_ok
            ),
            "timing": timing_ok,
        }

        case_passed = all(
            gates.values()
        )

        if case_passed:

            passed_cases += 1

            status = "PASS"

        else:

            failed_cases += 1

            status = "FAIL"

        print()
        print(
            f"{case['case_id']}: "
            f"{status}"
        )

        print(
            f"  Query       : "
            f"{query}"
        )

        print(
            f"  Assessment  : "
            f"{result.assessment}"
        )

        print(
            f"  Confidence  : "
            f"{result.confidence}"
        )

        print(
            f"  Closure     : "
            f"{result.closure_status}"
        )

        print(
            f"  Citations   : "
            f"{len(result.citations)}"
        )

        print(
            f"  Grounding   : "
            f"{result.grounding_passed}"
        )

        print(
            f"  Delivery    : "
            f"{result.delivery_status.value}"
        )

        print(
            f"  Time        : "
            f"{result.timing.total_seconds:.3f} s"
        )

        if not case_passed:

            print(
                "  Failed gates:"
            )

            for (
                gate_name,
                gate_passed,
            ) in gates.items():

                if gate_passed:
                    continue

                print(
                    f"    - {gate_name}"
                )

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------

    total_cases = len(
        cases
    )

    safe_delivery_rate = (
        passed_cases
        / total_cases
        if total_cases
        else 0.0
    )

    average_total_time = (
        total_time
        / total_cases
        if total_cases
        else 0.0
    )

    average_deterministic_time = (
        total_deterministic_time
        / total_cases
        if total_cases
        else 0.0
    )

    print()
    print("=" * 78)
    print("AGGREGATE RESULTS")
    print("=" * 78)

    print(
        f"Passed cases               : "
        f"{passed_cases}"
    )

    print(
        f"Failed cases               : "
        f"{failed_cases}"
    )

    print(
        f"Production acceptance rate : "
        f"{safe_delivery_rate:.3f}"
    )

    print(
        f"Average deterministic time : "
        f"{average_deterministic_time:.3f} s"
    )

    print(
        f"Average total request time : "
        f"{average_total_time:.3f} s"
    )

    print(
        "Target production rate     : 1.000"
    )

    return (
        0
        if failed_cases == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
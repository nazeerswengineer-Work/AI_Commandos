"""
REQTRACE AI Phase 14E end-to-end narration evaluation.

Runs the complete safe narration path:

Phase 11
    Hybrid retrieval

Phase 12
    Deterministic engineering decision

Phase 13
    Citation construction
    Grounding validation
    Safe answer contract
    Deterministic answer package
    Adversarial validation

Phase 14
    Constrained narration prompt
    Ollama generation
    Narration validation
    One repair attempt
    Deterministic fallback

Important:

A deterministic fallback is a valid safe delivery.

Engineering-ID preservation is evaluated against the authoritative
Phase 13 engineering-ID allowlist, not against raw evidence excerpts.

This distinction is especially important for OBS-052:

    Authoritative traceability:
        REQ-041

    Source evidence may mention:
        REQ-042

REQ-042 appearing inside citation-bound source evidence does not make
REQ-042 an authoritative engineering relationship.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import get_settings

from src.answer_pipeline import (
    build_grounded_answer_package,
)
from src.grounded_narrator import (
    GroundedNarrator,
    GroundedNarratorConfig,
)
from src.narration_client import (
    NarrationClientConfig,
    OllamaNarrationClient,
)
from src.narration_models import (
    NarrationStatus,
)
from src.narration_prompt import (
    build_narration_prompt,
)


VALID_DELIVERY_STATES = {
    NarrationStatus.VALIDATED_GENERATION,
    NarrationStatus.VALIDATED_REPAIR,
    NarrationStatus.DETERMINISTIC_FALLBACK,
}


def required_ids_are_authoritative(
    authoritative_ids: set[str],
    required_ids: list[str],
) -> bool:
    """
    Verify that every required engineering ID exists in the
    authoritative Phase 13 engineering-ID allowlist.
    """

    required = set(
        required_ids
    )

    return required.issubset(
        authoritative_ids
    )


def forbidden_ids_are_not_authoritative(
    authoritative_ids: set[str],
    forbidden_ids: list[str],
) -> bool:
    """
    Verify that forbidden candidate IDs have not entered the
    authoritative engineering-ID allowlist.

    A forbidden ID may still legitimately appear inside quoted
    evidence. That does not make the ID authoritative.
    """

    forbidden = set(
        forbidden_ids
    )

    return not bool(
        authoritative_ids
        & forbidden
    )


def validated_delivery_is_safe(
    delivery,
) -> bool:
    """
    Verify that a generated delivery was actually validated.

    Deterministic fallback does not require a passed LLM validation
    because the delivered text is the validated Phase 13D fallback.
    """

    if (
        delivery.status
        == NarrationStatus
        .VALIDATED_GENERATION
    ):
        return bool(
            delivery.initial_validation
            and delivery
            .initial_validation
            .passed
        )

    if (
        delivery.status
        == NarrationStatus
        .VALIDATED_REPAIR
    ):
        return bool(
            delivery.repair_validation
            and delivery
            .repair_validation
            .passed
        )

    if (
        delivery.status
        == NarrationStatus
        .DETERMINISTIC_FALLBACK
    ):
        return True

    return False


def fallback_is_exact(
    delivery,
    deterministic_text: str,
) -> bool:
    """
    Verify that fallback delivery exactly matches Phase 13D
    deterministic output.
    """

    if not delivery.used_fallback:
        return True

    return bool(
        delivery.status
        == NarrationStatus
        .DETERMINISTIC_FALLBACK
        and delivery.narration
        == deterministic_text
    )


def revision_conflict_is_preserved(
    *,
    case: dict,
    package,
    delivery,
) -> bool:
    """
    Verify that an unresolved requirement-revision conflict remains
    unresolved.

    The evaluator does not require the narration to avoid mentioning
    revision names. Both revisions are legitimate evidence.

    It only rejects language that selects one revision as applicable.
    """

    if not case.get(
        "must_preserve_revision_conflict",
        False,
    ):
        return True

    if (
        package.decision.assessment
        != "INCONCLUSIVE"
    ):
        return False

    lowered = (
        delivery.narration
        .lower()
    )

    forbidden_revision_phrases = (
        "rev01 is applicable",
        "rev02 is applicable",
        "rev01 is the applicable revision",
        "rev02 is the applicable revision",
        "applicable revision is rev01",
        "applicable revision is rev02",
        "rev01 should be used",
        "rev02 should be used",
    )

    return not any(
        phrase in lowered
        for phrase
        in forbidden_revision_phrases
    )


def traceability_is_preserved(
    *,
    case: dict,
    authoritative_ids: set[str],
) -> bool:
    """
    Apply the dedicated OBS-052 traceability protection.

    Authoritative relationship:

        OBS-052 -> REQ-041

    REQ-042 must not enter the authoritative engineering-ID
    allowlist.

    REQ-042 may still appear inside citation-bound source evidence.
    """

    if (
        case.get("case_id")
        != "NAR-004"
    ):
        return True

    return bool(
        "REQ-041"
        in authoritative_ids
        and "OBS-052"
        in authoritative_ids
        and "REQ-042"
        not in authoritative_ids
    )


def decision_matches_case(
    *,
    case: dict,
    package,
) -> bool:
    """Verify authoritative deterministic decision fields."""

    return bool(
        package.decision.assessment
        == case["assessment"]
        and package.decision.confidence
        == case["confidence"]
        and package.decision.closure_status
        == case["closure_status"]
    )


def main() -> int:
    """Run all Phase 14E narration scenarios."""

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    case_path = (
        base_directory
        / "evaluation"
        / "narration_cases.json"
    )

    cases = json.loads(
        case_path.read_text(
            encoding="utf-8"
        )
    )

    settings = get_settings()

    client_config = (
        NarrationClientConfig(
            base_url=(
                settings.ollama_base_url
            ),
            model="llama3.2:1b",
            connect_timeout_seconds=(
                settings
                .ollama_connect_timeout_seconds
            ),
            read_timeout_seconds=(
                settings.llm_timeout_seconds
            ),
            temperature=0.1,
            top_p=0.8,
            top_k=20,
            repeat_penalty=1.1,
            seed=42,
            context_window=8192,
            maximum_output_tokens=900,
        )
    )

    narrator_config = (
        GroundedNarratorConfig(
            allow_repair_attempt=True,
            maximum_generation_attempts=2,
        )
    )

    passed_cases = 0
    failed_cases = 0

    validated_initial = 0
    validated_repairs = 0
    deterministic_fallbacks = 0

    total_generation_attempts = 0

    print()

    print("=" * 78)

    print(
        "REQTRACE AI - PHASE 14E "
        "END-TO-END NARRATION EVALUATION"
    )

    print("=" * 78)

    with OllamaNarrationClient(
        client_config
    ) as client:

        narrator = GroundedNarrator(
            client=client,
            config=narrator_config,
        )

        for case in cases:

            package = (
                build_grounded_answer_package(
                    case["query"],
                    settings,
                    base_directory,
                    evaluate_closure=bool(
                        case["closure"]
                    ),
                )
            )

            prompt = (
                build_narration_prompt(
                    query=case["query"],
                    package=package,
                )
            )

            delivery = narrator.narrate(
                prompt
            )

            total_generation_attempts += (
                delivery
                .generation_attempts
            )

            # -------------------------------------------------
            # Delivery metrics
            # -------------------------------------------------

            if (
                delivery.status
                == NarrationStatus
                .VALIDATED_GENERATION
            ):
                validated_initial += 1

            elif (
                delivery.status
                == NarrationStatus
                .VALIDATED_REPAIR
            ):
                validated_repairs += 1

            elif (
                delivery.status
                == NarrationStatus
                .DETERMINISTIC_FALLBACK
            ):
                deterministic_fallbacks += 1

            # -------------------------------------------------
            # Authoritative engineering identity
            # -------------------------------------------------

            authoritative_ids = set(
                package
                .engineering_id_allowlist
            )

            required_ids = case.get(
                "required_ids",
                [],
            )

            forbidden_ids = case.get(
                "forbidden_ids",
                [],
            )

            required_ids_ok = (
                required_ids_are_authoritative(
                    authoritative_ids,
                    required_ids,
                )
            )

            forbidden_ids_ok = (
                forbidden_ids_are_not_authoritative(
                    authoritative_ids,
                    forbidden_ids,
                )
            )

            # -------------------------------------------------
            # Deterministic decision preservation
            # -------------------------------------------------

            decision_ok = (
                decision_matches_case(
                    case=case,
                    package=package,
                )
            )

            # -------------------------------------------------
            # Valid delivery state
            # -------------------------------------------------

            delivery_status_ok = bool(
                delivery.status
                in VALID_DELIVERY_STATES
            )

            generation_limit_ok = bool(
                0
                <= delivery
                .generation_attempts
                <= 2
            )

            # -------------------------------------------------
            # Generated output must be validated.
            #
            # Fallback text does not depend on a passed LLM
            # validation because it comes from Phase 13D.
            # -------------------------------------------------

            validated_output_ok = (
                validated_delivery_is_safe(
                    delivery
                )
            )

            # -------------------------------------------------
            # Deterministic fallback integrity
            # -------------------------------------------------

            fallback_ok = (
                fallback_is_exact(
                    delivery,
                    package
                    .deterministic_text,
                )
            )

            # -------------------------------------------------
            # REQ-050 revision-conflict protection
            # -------------------------------------------------

            revision_conflict_ok = (
                revision_conflict_is_preserved(
                    case=case,
                    package=package,
                    delivery=delivery,
                )
            )

            # -------------------------------------------------
            # OBS-052 explicit traceability protection
            # -------------------------------------------------

            traceability_ok = (
                traceability_is_preserved(
                    case=case,
                    authoritative_ids=(
                        authoritative_ids
                    ),
                )
            )

            # -------------------------------------------------
            # Final scenario result
            # -------------------------------------------------

            case_passed = all(
                (
                    decision_ok,
                    delivery_status_ok,
                    generation_limit_ok,
                    required_ids_ok,
                    forbidden_ids_ok,
                    fallback_ok,
                    validated_output_ok,
                    revision_conflict_ok,
                    traceability_ok,
                )
            )

            if case_passed:
                passed_cases += 1

                status = "PASS"

            else:
                failed_cases += 1

                status = "FAIL"

            # -------------------------------------------------
            # Scenario diagnostics
            # -------------------------------------------------

            print()

            print(
                f"{case['case_id']}: "
                f"{status}"
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
                f"  Delivery    : "
                f"{delivery.status.value}"
            )

            print(
                f"  Attempts    : "
                f"{delivery.generation_attempts}"
            )

            print(
                f"  Repair      : "
                f"{delivery.used_repair}"
            )

            print(
                f"  Fallback    : "
                f"{delivery.used_fallback}"
            )

            print(
                f"  Decision OK : "
                f"{decision_ok}"
            )

            print(
                f"  Required IDs: "
                f"{required_ids_ok}"
            )

            print(
                f"  Forbidden IDs: "
                f"{forbidden_ids_ok}"
            )

            print(
                f"  Traceability: "
                f"{traceability_ok}"
            )

            print(
                f"  Revision safe: "
                f"{revision_conflict_ok}"
            )

            if delivery.fallback_reason:
                print(
                    f"  Reason      : "
                    f"{delivery.fallback_reason}"
                )

            if not case_passed:

                print(
                    "  Failed gates:"
                )

                gates = {
                    "decision": (
                        decision_ok
                    ),
                    "delivery_status": (
                        delivery_status_ok
                    ),
                    "generation_limit": (
                        generation_limit_ok
                    ),
                    "required_ids": (
                        required_ids_ok
                    ),
                    "forbidden_ids": (
                        forbidden_ids_ok
                    ),
                    "fallback_integrity": (
                        fallback_ok
                    ),
                    "validated_output": (
                        validated_output_ok
                    ),
                    "revision_conflict": (
                        revision_conflict_ok
                    ),
                    "traceability": (
                        traceability_ok
                    ),
                }

                for (
                    gate_name,
                    gate_passed,
                ) in gates.items():

                    if gate_passed:
                        continue

                    print(
                        f"    - {gate_name}"
                    )

    # ---------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------

    case_count = len(
        cases
    )

    average_attempts = (
        total_generation_attempts
        / case_count
        if case_count
        else 0.0
    )

    safe_delivery_rate = (
        passed_cases
        / case_count
        if case_count
        else 0.0
    )

    first_pass_rate = (
        validated_initial
        / case_count
        if case_count
        else 0.0
    )

    fallback_rate = (
        deterministic_fallbacks
        / case_count
        if case_count
        else 0.0
    )

    cases_requiring_repair = (
        validated_repairs
        + deterministic_fallbacks
    )

    repair_success_rate = (
        validated_repairs
        / cases_requiring_repair
        if cases_requiring_repair
        else 0.0
    )

    print()

    print("=" * 78)

    print(
        "AGGREGATE RESULTS"
    )

    print("=" * 78)

    print(
        f"Passed cases              : "
        f"{passed_cases}"
    )

    print(
        f"Failed cases              : "
        f"{failed_cases}"
    )

    print(
        f"Validated initial outputs : "
        f"{validated_initial}"
    )

    print(
        f"Validated repairs         : "
        f"{validated_repairs}"
    )

    print(
        f"Deterministic fallbacks   : "
        f"{deterministic_fallbacks}"
    )

    print(
        f"Safe-delivery rate        : "
        f"{safe_delivery_rate:.3f}"
    )

    print(
        f"First-pass validation rate: "
        f"{first_pass_rate:.3f}"
    )

    print(
        f"Repair-success rate       : "
        f"{repair_success_rate:.3f}"
    )

    print(
        f"Fallback rate             : "
        f"{fallback_rate:.3f}"
    )

    print(
        f"Average generation calls  : "
        f"{average_attempts:.2f}"
    )

    print(
        "Maximum allowed calls     : 2"
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
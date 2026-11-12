"""
Tests for Phase 12B deterministic evidence-condition detection.
"""

from __future__ import annotations

from src.context_builder import (
    FinalEvidenceContext,
)
from src.evidence_conditions import (
    EvidenceConditionType,
    detect_evidence_conditions,
)
from src.retrieval import (
    RetrievedEvidence,
)


def make_evidence(
    evidence_id: str,
    document_type: str,
    content: str,
    *,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
    result_status: str | None = None,
    document_revision: str | None = None,
) -> RetrievedEvidence:

    metadata: dict[str, object] = {}

    if requirement_id:
        metadata[
            "requirement_id"
        ] = requirement_id

    if observation_id:
        metadata[
            "observation_id"
        ] = observation_id

    if test_case_id:
        metadata[
            "test_case_id"
        ] = test_case_id

    if result_status:
        metadata[
            "result_status"
        ] = result_status

    if document_revision:
        metadata[
            "document_revision"
        ] = document_revision

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type=document_type,
        content=content,
        metadata=metadata,
        retrieval_method="test",
        score=1.0,
        rank=1,
    )


def make_context(
    evidence: list[
        RetrievedEvidence
    ],
) -> FinalEvidenceContext:

    return FinalEvidenceContext(
        evidence=tuple(evidence),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def test_executed_fail_condition() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement",
                requirement_id="REQ-017",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observation",
                requirement_id="REQ-017",
                observation_id="OBS-023",
            ),
            make_evidence(
                "CODE",
                "code",
                "Code",
                requirement_id="REQ-017",
            ),
            make_evidence(
                "TEST",
                "test",
                "Test",
                requirement_id="REQ-017",
                test_case_id="TC-102",
                result_status="FAIL",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .EXECUTED_FAIL_TEST
    )


def test_executed_pass_condition() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement",
                requirement_id="REQ-021",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observation",
                requirement_id="REQ-021",
                observation_id="OBS-031",
            ),
            make_evidence(
                "CODE",
                "code",
                "Code",
            ),
            make_evidence(
                "TEST",
                "test",
                "Test",
                test_case_id="TC-201",
                result_status="PASS",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .EXECUTED_PASS_TEST
    )


def test_not_executed_condition() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement",
                requirement_id="REQ-033",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observation",
                observation_id="OBS-044",
            ),
            make_evidence(
                "CODE",
                "code",
                "Partial code",
            ),
            make_evidence(
                "TEST",
                "test",
                "Test",
                test_case_id="TC-301",
                result_status="NOT EXECUTED",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .TEST_NOT_EXECUTED
    )


def test_missing_test_condition() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement",
                requirement_id="REQ-041",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observation",
                observation_id="OBS-052",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .TEST_EVIDENCE_MISSING
    )


def test_traceability_concern() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement",
                requirement_id="REQ-041",
            ),
            make_evidence(
                "OBS",
                "observation",
                (
                    "Traceability Concern. "
                    "REQ-041 must not silently replace "
                    "or be replaced using semantic similarity."
                ),
                requirement_id="REQ-041",
                observation_id="OBS-052",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .TRACEABILITY_CONCERN
    )


def test_revision_conflict() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD-1",
                "frd",
                "Requires 500 milliseconds.",
                requirement_id="REQ-050",
                document_revision="Rev01",
            ),
            make_evidence(
                "FRD-2",
                "frd",
                "Requires 1000 milliseconds.",
                requirement_id="REQ-050",
                document_revision="Rev02",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observed 760 milliseconds.",
                requirement_id="REQ-050",
                observation_id="OBS-061",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .REQUIREMENT_REVISION_CONFLICT
    )


def test_contradiction_indicator() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observation",
            ),
            make_evidence(
                "REVIEW",
                "review",
                (
                    "Contradiction identified. "
                    "The supplied evidence contradicts "
                    "the observation."
                ),
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .CONTRADICTION_INDICATOR
    )


def test_closure_limitation() -> None:

    context = make_context(
        [
            make_evidence(
                "OBS",
                "observation",
                "Original observation.",
                observation_id="OBS-070",
            ),
            make_evidence(
                "REVIEW",
                "review",
                (
                    "Missing Closure Evidence. "
                    "Corrected source code: Not available. "
                    "Post-correction test result: Not available."
                ),
                observation_id="OBS-070",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .CLOSURE_EVIDENCE_LIMITATION
    )


def test_phase_12b_does_not_output_assessment() -> None:

    context = make_context(
        [
            make_evidence(
                "OBS",
                "observation",
                "Observation",
            )
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    result_dictionary = {
        "completeness": (
            result.completeness
            .classification
            .value
        ),
        "conditions": [
            condition
            .condition_type
            .value
            for condition
            in result.conditions
        ],
    }

    assert (
        "assessment"
        not in result_dictionary
    )

    assert (
        "closure_status"
        not in result_dictionary
    )

def test_technical_contradiction_without_review() -> None:
    """
    OBS-031 contradiction must be detectable from technical evidence
    even when Review evidence is outside the final context.
    """

    context = make_context(
        [
            make_evidence(
                "OBS-031",
                "observation",
                (
                    "The command identifier 17 was accepted "
                    "instead of being rejected."
                ),
                requirement_id="REQ-021",
                observation_id="OBS-031",
                test_case_id="TC-201",
            ),
            make_evidence(
                "CODE",
                "code",
                (
                    "if ((command_id < COMMAND_ID_MINIMUM) || "
                    "(command_id > COMMAND_ID_MAXIMUM)) "
                    "{ return COMMAND_INVALID_ID; }"
                ),
                requirement_id="REQ-021",
                observation_id="OBS-031",
                test_case_id="TC-201",
            ),
            make_evidence(
                "FRD",
                "frd",
                (
                    "The system shall reject commands whose command "
                    "identifier is outside the inclusive range 1 through 16."
                ),
                requirement_id="REQ-021",
            ),
            make_evidence(
                "TC-201",
                "test",
                (
                    "Result Status: PASS. Identifier 17 returned "
                    "COMMAND_INVALID_ID."
                ),
                requirement_id="REQ-021",
                test_case_id="TC-201",
                result_status="PASS",
            ),
        ]
    )

    result = detect_evidence_conditions(
        context
    )

    assert result.has_condition(
        EvidenceConditionType
        .EXECUTED_PASS_TEST
    )

    assert result.has_condition(
        EvidenceConditionType
        .CONTRADICTION_INDICATOR
    )
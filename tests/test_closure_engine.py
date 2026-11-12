"""
Tests for Phase 12D deterministic closure readiness.
"""

from __future__ import annotations

from src.assessment_engine import (
    ObservationAssessment,
)
from src.closure_engine import (
    ClosureRule,
    ClosureStatus,
    evaluate_closure_readiness,
)
from src.context_builder import (
    FinalEvidenceContext,
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
    evidence: list[RetrievedEvidence],
) -> FinalEvidenceContext:

    return FinalEvidenceContext(
        evidence=tuple(evidence),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def obs_070_original_defect_context() -> FinalEvidenceContext:
    """OBS-070 original defect plus missing closure evidence."""

    return make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement evidence.",
                requirement_id="REQ-060",
            ),
            make_evidence(
                "OBS",
                "observation",
                (
                    "Original observation. "
                    "Missing Closure Evidence."
                ),
                requirement_id="REQ-060",
                observation_id="OBS-070",
                test_case_id="TC-401",
            ),
            make_evidence(
                "CODE",
                "code",
                (
                    "Original defective implementation. "
                    "No corrected implementation is supplied."
                ),
                requirement_id="REQ-060",
                observation_id="OBS-070",
                test_case_id="TC-401",
            ),
            make_evidence(
                "TEST",
                "test",
                (
                    "Result Status: FAIL. "
                    "Post-correction test result: Not available."
                ),
                requirement_id="REQ-060",
                test_case_id="TC-401",
                result_status="FAIL",
            ),
        ]
    )


def test_supported_observation_can_have_unestablished_closure() -> None:

    result = evaluate_closure_readiness(
        obs_070_original_defect_context()
    )

    assert (
        result.assessment.assessment
        == ObservationAssessment.SUPPORTED
    )

    assert result.status == (
        ClosureStatus
        .CLOSURE_NOT_ESTABLISHED
    )


def test_explicit_closure_limitation_has_priority() -> None:

    result = evaluate_closure_readiness(
        obs_070_original_defect_context()
    )

    assert result.applied_rule == (
        ClosureRule
        .EXPLICIT_CLOSURE_LIMITATION
    )


def test_missing_corrected_code_blocks_closure() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement.",
                requirement_id="REQ-060",
            ),
            make_evidence(
                "OBS",
                "observation",
                (
                    "Corrective action implemented. "
                    "Original defect."
                ),
                requirement_id="REQ-060",
                observation_id="OBS-070",
                test_case_id="TC-401",
            ),
            make_evidence(
                "CODE",
                "code",
                "No corrected implementation is supplied.",
                requirement_id="REQ-060",
            ),
            make_evidence(
                "TEST",
                "test",
                "Result Status: FAIL.",
                test_case_id="TC-401",
                result_status="FAIL",
            ),
        ]
    )

    result = evaluate_closure_readiness(
        context
    )

    assert result.status == (
        ClosureStatus
        .CLOSURE_NOT_ESTABLISHED
    )


def test_complete_closure_chain_can_be_established() -> None:

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement.",
                requirement_id="REQ-060",
            ),
            make_evidence(
                "OBS",
                "observation",
                (
                    "Original defect. "
                    "Corrective action implemented."
                ),
                requirement_id="REQ-060",
                observation_id="OBS-070",
                test_case_id="TC-401",
            ),
            make_evidence(
                "CODE",
                "code",
                (
                    "Corrected source code. "
                    "Updated implementation."
                ),
                requirement_id="REQ-060",
            ),
            make_evidence(
                "TEST",
                "test",
                (
                    "Result Status: FAIL. "
                    "Post-correction verification: PASS."
                ),
                requirement_id="REQ-060",
                test_case_id="TC-401",
                result_status="FAIL",
            ),
        ]
    )

    result = evaluate_closure_readiness(
        context
    )

    assert result.status == (
        ClosureStatus
        .CLOSURE_ESTABLISHED
    )

    assert result.applied_rule == (
        ClosureRule
        .CLOSURE_EVIDENCE_COMPLETE
    )


def test_original_fail_does_not_establish_closure() -> None:

    result = evaluate_closure_readiness(
        obs_070_original_defect_context()
    )

    assert (
        result.evidence_state
        .post_correction_test_available
        is False
    )


def test_assessment_and_closure_are_separate() -> None:

    result = evaluate_closure_readiness(
        obs_070_original_defect_context()
    )

    assert (
        result.assessment.assessment.value
        == "SUPPORTED"
    )

    assert (
        result.status.value
        == "CLOSURE NOT ESTABLISHED"
    )
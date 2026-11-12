"""
Tests for the complete Phase 12E decision result.
"""

from __future__ import annotations

from src.closure_engine import (
    ClosureStatus,
)
from src.context_builder import (
    FinalEvidenceContext,
)
from src.decision_engine import (
    evaluate_engineering_decision,
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


def obs_070_context() -> FinalEvidenceContext:
    return FinalEvidenceContext(
        evidence=(
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
        ),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def test_closure_not_assessed_by_default() -> None:
    result = evaluate_engineering_decision(
        obs_070_context()
    )

    assert result.closure_status == (
        ClosureStatus.NOT_ASSESSED
    )

    assert (
        result.closure_was_assessed
        is False
    )


def test_closure_evaluated_only_when_requested() -> None:
    result = evaluate_engineering_decision(
        obs_070_context(),
        evaluate_closure=True,
    )

    assert result.closure_status == (
        ClosureStatus
        .CLOSURE_NOT_ESTABLISHED
    )

    assert (
        result.closure_was_assessed
        is True
    )


def test_assessment_is_preserved_during_closure_evaluation() -> None:
    result = evaluate_engineering_decision(
        obs_070_context(),
        evaluate_closure=True,
    )

    assert (
        result.assessment
        .assessment
        .value
        == "SUPPORTED"
    )

    assert (
        result.closure_status.value
        == "CLOSURE NOT ESTABLISHED"
    )


def test_evidence_provenance_is_preserved() -> None:
    result = evaluate_engineering_decision(
        obs_070_context(),
        evaluate_closure=True,
    )

    assert result.evidence_ids == (
        "FRD",
        "OBS",
        "CODE",
        "TEST",
    )
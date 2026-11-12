"""
Tests for Phase 12A deterministic evidence completeness.
"""

from __future__ import annotations

from src.context_builder import (
    FinalEvidenceContext,
)
from src.evidence_completeness import (
    TestExecutionState,
    evaluate_evidence_completeness,
    normalize_test_execution_state,
)
from src.retrieval import (
    RetrievedEvidence,
)
from src.schemas import (
    EvidenceCompleteness,
)


def make_evidence(
    evidence_id: str,
    document_type: str,
    *,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
    result_status: str | None = None,
    document_revision: str | None = None,
    rank: int = 1,
) -> RetrievedEvidence:
    """Create synthetic evidence for completeness tests."""

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
        content="Synthetic evidence",
        metadata=metadata,
        retrieval_method="test",
        score=1.0,
        rank=rank,
    )


def make_context(
    evidence: list[RetrievedEvidence],
) -> FinalEvidenceContext:
    """Create a final context for unit testing."""

    return FinalEvidenceContext(
        evidence=tuple(evidence),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def complete_context(
    test_status: str,
) -> FinalEvidenceContext:
    """Create a complete four-type engineering context."""

    return make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                requirement_id="REQ-017",
                document_revision="Rev01",
            ),
            make_evidence(
                "OBS",
                "observation",
                requirement_id="REQ-017",
                observation_id="OBS-023",
            ),
            make_evidence(
                "CODE",
                "code",
                requirement_id="REQ-017",
                observation_id="OBS-023",
                test_case_id="TC-102",
            ),
            make_evidence(
                "TEST",
                "test",
                requirement_id="REQ-017",
                test_case_id="TC-102",
                result_status=test_status,
            ),
        ]
    )


def test_pass_status_is_executed() -> None:
    assert normalize_test_execution_state(
        "PASS"
    ) == (
        TestExecutionState
        .EXECUTED_PASS
    )


def test_fail_status_is_executed() -> None:
    assert normalize_test_execution_state(
        "FAIL"
    ) == (
        TestExecutionState
        .EXECUTED_FAIL
    )


def test_not_executed_status() -> None:
    assert normalize_test_execution_state(
        "NOT EXECUTED"
    ) == (
        TestExecutionState
        .NOT_EXECUTED
    )


def test_complete_with_passing_test() -> None:
    result = (
        evaluate_evidence_completeness(
            complete_context("PASS")
        )
    )

    assert result.classification == (
        EvidenceCompleteness.COMPLETE
    )

    assert result.has_executed_test is True
    assert result.blocking_items == ()


def test_complete_with_failing_test() -> None:
    """
    A FAIL result is still executed evidence.

    Test outcome will affect later assessment rules, not evidence
    completeness.
    """

    result = (
        evaluate_evidence_completeness(
            complete_context("FAIL")
        )
    )

    assert result.classification == (
        EvidenceCompleteness.COMPLETE
    )

    assert result.has_executed_test is True


def test_not_executed_test_is_partial() -> None:
    result = (
        evaluate_evidence_completeness(
            complete_context(
                "NOT EXECUTED"
            )
        )
    )

    assert result.classification == (
        EvidenceCompleteness.PARTIAL
    )

    assert result.has_executed_test is False

    assert any(
        "not executed"
        in item.lower()
        for item in result.blocking_items
    )


def test_missing_test_is_partial() -> None:
    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                requirement_id="REQ-050",
            ),
            make_evidence(
                "OBS",
                "observation",
                requirement_id="REQ-050",
                observation_id="OBS-061",
            ),
            make_evidence(
                "CODE",
                "code",
                requirement_id="REQ-050",
            ),
        ]
    )

    result = (
        evaluate_evidence_completeness(
            context
        )
    )

    assert result.classification == (
        EvidenceCompleteness.PARTIAL
    )

    assert "test" in (
        result.missing_evidence_types
    )


def test_missing_code_is_partial() -> None:
    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                requirement_id="REQ-033",
            ),
            make_evidence(
                "OBS",
                "observation",
                requirement_id="REQ-033",
                observation_id="OBS-044",
            ),
            make_evidence(
                "TEST",
                "test",
                requirement_id="REQ-033",
                test_case_id="TC-301",
                result_status=(
                    "NOT EXECUTED"
                ),
            ),
        ]
    )

    result = (
        evaluate_evidence_completeness(
            context
        )
    )

    assert result.classification == (
        EvidenceCompleteness.PARTIAL
    )

    assert "code" in (
        result.missing_evidence_types
    )


def test_missing_frd_is_insufficient() -> None:
    context = make_context(
        [
            make_evidence(
                "OBS",
                "observation",
                observation_id="OBS-023",
            ),
            make_evidence(
                "CODE",
                "code",
                observation_id="OBS-023",
            ),
            make_evidence(
                "TEST",
                "test",
                test_case_id="TC-102",
                result_status="FAIL",
            ),
        ]
    )

    result = (
        evaluate_evidence_completeness(
            context
        )
    )

    assert result.classification == (
        EvidenceCompleteness.INSUFFICIENT
    )

    assert (
        result.can_run_assessment_rules
        is False
    )


def test_missing_observation_is_insufficient() -> None:
    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                requirement_id="REQ-017",
            ),
            make_evidence(
                "CODE",
                "code",
                requirement_id="REQ-017",
            ),
            make_evidence(
                "TEST",
                "test",
                requirement_id="REQ-017",
                test_case_id="TC-102",
                result_status="FAIL",
            ),
        ]
    )

    result = (
        evaluate_evidence_completeness(
            context
        )
    )

    assert result.classification == (
        EvidenceCompleteness.INSUFFICIENT
    )


def test_review_is_not_mandatory_for_complete() -> None:
    result = (
        evaluate_evidence_completeness(
            complete_context("PASS")
        )
    )

    assert not result.inventory.review

    assert result.classification == (
        EvidenceCompleteness.COMPLETE
    )


def test_multiple_revisions_are_preserved() -> None:
    context = make_context(
        [
            make_evidence(
                "FRD-REV01",
                "frd",
                requirement_id="REQ-050",
                document_revision="Rev01",
            ),
            make_evidence(
                "FRD-REV02",
                "frd",
                requirement_id="REQ-050",
                document_revision="Rev02",
            ),
            make_evidence(
                "OBS",
                "observation",
                requirement_id="REQ-050",
                observation_id="OBS-061",
            ),
        ]
    )

    result = (
        evaluate_evidence_completeness(
            context
        )
    )

    assert result.document_revisions == (
        "Rev01",
        "Rev02",
    )

    # Completeness preserves revision state but does not resolve it.
    assert result.classification == (
        EvidenceCompleteness.PARTIAL
    )
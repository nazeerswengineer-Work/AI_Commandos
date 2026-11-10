"""
Tests for Phase 11E retrieval evaluation.
"""

from __future__ import annotations

import pytest

from src.context_builder import (
    FinalEvidenceContext,
)
from src.retrieval import (
    RetrievedEvidence,
)
from src.retrieval_evaluation import (
    RetrievalExpectation,
    build_evaluation_report,
    coverage,
    evaluate_context,
    precision_at_k,
    reciprocal_rank,
)


def make_evidence(
    evidence_id: str,
    document_type: str,
    *,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
    rank: int = 1,
) -> RetrievedEvidence:

    metadata: dict[str, str] = {}

    if requirement_id:
        metadata["requirement_id"] = (
            requirement_id
        )

    if observation_id:
        metadata["observation_id"] = (
            observation_id
        )

    if test_case_id:
        metadata["test_case_id"] = (
            test_case_id
        )

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

    return FinalEvidenceContext(
        evidence=tuple(evidence),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def test_coverage() -> None:

    result = coverage(
        {"REQ-021"},
        (
            "REQ-021",
            "REQ-050",
        ),
    )

    assert result == pytest.approx(
        0.5
    )


def test_precision_metric() -> None:

    assert precision_at_k(
        (
            True,
            True,
            False,
            True,
        )
    ) == pytest.approx(
        0.75
    )


def test_reciprocal_rank_metric() -> None:

    assert reciprocal_rank(
        (
            False,
            True,
            True,
        )
    ) == pytest.approx(
        0.5
    )


def test_perfect_command_context_passes() -> None:

    context = make_context(
        [
            make_evidence(
                "REQ",
                "frd",
                requirement_id="REQ-021",
                rank=1,
            ),
            make_evidence(
                "OBS",
                "observation",
                requirement_id="REQ-021",
                observation_id="OBS-031",
                rank=2,
            ),
            make_evidence(
                "CODE",
                "code",
                requirement_id="REQ-021",
                observation_id="OBS-031",
                test_case_id="TC-201",
                rank=3,
            ),
            make_evidence(
                "TEST",
                "test",
                requirement_id="REQ-021",
                test_case_id="TC-201",
                rank=4,
            ),
        ]
    )

    expectation = RetrievalExpectation(
        case_id="RET-002",
        query=(
            "command identifier 17 "
            "should be rejected"
        ),
        requirement_ids=(
            "REQ-021",
        ),
        observation_ids=(
            "OBS-031",
        ),
        test_case_ids=(
            "TC-201",
        ),
        required_document_types=(
            "frd",
            "observation",
            "code",
            "test",
        ),
    )

    result = evaluate_context(
        context,
        expectation,
    )

    assert result.passed is True
    assert (
        result.requirement_coverage
        == 1.0
    )
    assert (
        result.observation_coverage
        == 1.0
    )
    assert result.test_coverage == 1.0
    assert result.type_coverage == 1.0
    assert result.recall_at_k == 1.0


def test_missing_code_fails_type_coverage() -> None:

    context = make_context(
        [
            make_evidence(
                "REQ",
                "frd",
                requirement_id="REQ-021",
            ),
            make_evidence(
                "OBS",
                "observation",
                observation_id="OBS-031",
            ),
            make_evidence(
                "TEST",
                "test",
                test_case_id="TC-201",
            ),
        ]
    )

    expectation = RetrievalExpectation(
        case_id="RET-002",
        query="command",
        requirement_ids=(
            "REQ-021",
        ),
        observation_ids=(
            "OBS-031",
        ),
        test_case_ids=(
            "TC-201",
        ),
        required_document_types=(
            "frd",
            "observation",
            "code",
            "test",
        ),
    )

    result = evaluate_context(
        context,
        expectation,
    )

    assert result.passed is False

    assert (
        result.type_coverage
        == pytest.approx(0.75)
    )


def test_report_aggregation() -> None:

    result_one = evaluate_context(
        make_context(
            [
                make_evidence(
                    "REQ",
                    "frd",
                    requirement_id="REQ-017",
                )
            ]
        ),
        RetrievalExpectation(
            case_id="A",
            query="A",
            requirement_ids=(
                "REQ-017",
            ),
            required_document_types=(
                "frd",
            ),
        ),
    )

    report = build_evaluation_report(
        [
            result_one
        ]
    )

    assert report.passed_cases == 1
    assert report.failed_cases == 0
    assert report.all_passed is True

    assert (
        report.mean_recall_at_k
        == pytest.approx(1.0)
    )
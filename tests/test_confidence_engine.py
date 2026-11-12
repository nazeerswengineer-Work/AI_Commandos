"""
Tests for Phase 12E deterministic confidence.
"""

from __future__ import annotations

from src.assessment_engine import (
    AssessmentRule,
    DeterministicAssessmentResult,
    ObservationAssessment,
)
from src.confidence_engine import (
    ConfidenceRule,
    DecisionConfidence,
    calculate_decision_confidence,
)
from src.evidence_completeness import (
    EvidenceCompletenessResult,
    EvidenceTypeInventory,
)
from src.evidence_conditions import (
    EvidenceConditionResult,
)
from src.schemas import EvidenceCompleteness


def make_assessment(
    *,
    assessment: ObservationAssessment,
    rule: AssessmentRule,
    completeness: EvidenceCompleteness,
    blocking_items: tuple[str, ...] = (),
    supporting_items: tuple[str, ...] = (),
) -> DeterministicAssessmentResult:
    """Create one synthetic assessment result."""

    inventory = EvidenceTypeInventory(
        frd=(),
        observation=(),
        code=(),
        test=(),
        review=(),
        unknown=(),
    )

    completeness_result = (
        EvidenceCompletenessResult(
            classification=completeness,
            inventory=inventory,
            test_execution_states=(),
            missing_evidence_types=(),
            blocking_items=blocking_items,
            supporting_notes=(),
            requirement_ids=(),
            observation_ids=(),
            test_case_ids=(),
            document_revisions=(),
        )
    )

    conditions = EvidenceConditionResult(
        completeness=(
            completeness_result
        ),
        conditions=(),
    )

    return DeterministicAssessmentResult(
        assessment=assessment,
        applied_rule=rule,
        reason="Synthetic reason.",
        requirement_ids=(),
        observation_ids=(),
        test_case_ids=(),
        evidence_ids=(),
        blocking_items=blocking_items,
        supporting_items=(
            supporting_items
        ),
        conditions=conditions,
    )


def test_complete_supported_is_high_confidence() -> None:
    result = calculate_decision_confidence(
        make_assessment(
            assessment=(
                ObservationAssessment.SUPPORTED
            ),
            rule=(
                AssessmentRule
                .EXECUTED_FAIL_SUPPORT
            ),
            completeness=(
                EvidenceCompleteness.COMPLETE
            ),
        )
    )

    assert result.confidence == (
        DecisionConfidence.HIGH
    )

    assert result.applied_rule == (
        ConfidenceRule
        .COMPLETE_DECISIVE_EVIDENCE
    )


def test_complete_not_supported_is_high_confidence() -> None:
    result = calculate_decision_confidence(
        make_assessment(
            assessment=(
                ObservationAssessment
                .NOT_SUPPORTED
            ),
            rule=(
                AssessmentRule
                .TECHNICAL_CONTRADICTION
            ),
            completeness=(
                EvidenceCompleteness.COMPLETE
            ),
        )
    )

    assert result.confidence == (
        DecisionConfidence.HIGH
    )


def test_revision_conflict_is_high_confidence_in_blocker() -> None:
    result = calculate_decision_confidence(
        make_assessment(
            assessment=(
                ObservationAssessment
                .INCONCLUSIVE
            ),
            rule=(
                AssessmentRule
                .REQUIREMENT_REVISION_CONFLICT
            ),
            completeness=(
                EvidenceCompleteness.PARTIAL
            ),
            blocking_items=(
                "Applicable revision is unknown.",
            ),
        )
    )

    assert result.confidence == (
        DecisionConfidence.HIGH
    )

    assert result.applied_rule == (
        ConfidenceRule.EXPLICIT_BLOCKER
    )


def test_insufficient_evidence_is_low_confidence() -> None:
    result = calculate_decision_confidence(
        make_assessment(
            assessment=(
                ObservationAssessment
                .INCONCLUSIVE
            ),
            rule=(
                AssessmentRule
                .INSUFFICIENT_EVIDENCE
            ),
            completeness=(
                EvidenceCompleteness
                .INSUFFICIENT
            ),
        )
    )

    assert result.confidence == (
        DecisionConfidence.LOW
    )


def test_partial_no_decisive_rule_is_medium() -> None:
    result = calculate_decision_confidence(
        make_assessment(
            assessment=(
                ObservationAssessment
                .INCONCLUSIVE
            ),
            rule=(
                AssessmentRule
                .NO_DECISIVE_RULE
            ),
            completeness=(
                EvidenceCompleteness.PARTIAL
            ),
        )
    )

    assert result.confidence == (
        DecisionConfidence.MEDIUM
    )
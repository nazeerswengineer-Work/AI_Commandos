"""
Deterministic confidence classification for REQTRACE AI.

Confidence describes the strength of the deterministic decision basis.

Important:

HIGH confidence for an INCONCLUSIVE assessment means the engine has
strong evidence that a specific blocker prevents a conclusion.

It does not mean high confidence that the observation is valid.

This module does not invoke an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.assessment_engine import (
    AssessmentRule,
    DeterministicAssessmentResult,
    ObservationAssessment,
)
from src.schemas import EvidenceCompleteness


class DecisionConfidence(str, Enum):
    """Confidence in the deterministic decision basis."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


DecisionConfidence.__test__ = False


class ConfidenceRule(str, Enum):
    """Rule responsible for confidence classification."""

    COMPLETE_DECISIVE_EVIDENCE = (
        "CF-001_COMPLETE_DECISIVE_EVIDENCE"
    )

    EXPLICIT_BLOCKER = (
        "CF-002_EXPLICIT_BLOCKER"
    )

    PARTIAL_EVIDENCE = (
        "CF-003_PARTIAL_EVIDENCE"
    )

    INSUFFICIENT_EVIDENCE = (
        "CF-004_INSUFFICIENT_EVIDENCE"
    )

    NO_DECISIVE_RULE = (
        "CF-005_NO_DECISIVE_RULE"
    )


ConfidenceRule.__test__ = False


@dataclass(frozen=True)
class ConfidenceResult:
    """Deterministic confidence result."""

    confidence: DecisionConfidence
    applied_rule: ConfidenceRule
    reason: str

    supporting_factors: tuple[str, ...]
    limiting_factors: tuple[str, ...]


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered, duplicate-free, non-empty strings."""

    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        clean_value = str(value).strip()

        if not clean_value:
            continue

        if clean_value in seen:
            continue

        seen.add(clean_value)
        output.append(clean_value)

    return tuple(output)


def calculate_decision_confidence(
    assessment: DeterministicAssessmentResult,
) -> ConfidenceResult:
    """
    Calculate confidence in the deterministic decision basis.

    Rules:

    CF-001:
        Complete evidence and decisive SUPPORTED or NOT SUPPORTED rule
        produces HIGH confidence.

    CF-002:
        An explicit deterministic blocker such as revision conflict,
        traceability concern, or unexecuted test produces HIGH confidence
        in the INCONCLUSIVE outcome.

    CF-003:
        Partial evidence without a strong explicit blocker produces
        MEDIUM confidence.

    CF-004:
        Insufficient evidence produces LOW confidence.

    CF-005:
        Complete evidence but no decisive rule produces MEDIUM confidence.
    """

    completeness = (
        assessment
        .conditions
        .completeness
        .classification
    )

    supporting_factors = list(
        assessment.supporting_items
    )

    limiting_factors = list(
        assessment.blocking_items
    )

    decisive_assessments = {
        ObservationAssessment.SUPPORTED,
        ObservationAssessment.NOT_SUPPORTED,
    }

    explicit_blocker_rules = {
        AssessmentRule.REQUIREMENT_REVISION_CONFLICT,
        AssessmentRule.TRACEABILITY_CONCERN,
        AssessmentRule.TEST_NOT_EXECUTED,
        AssessmentRule.TEST_EVIDENCE_MISSING,
        AssessmentRule.CODE_EVIDENCE_MISSING,
    }

    if (
        completeness
        == EvidenceCompleteness.COMPLETE
        and assessment.assessment
        in decisive_assessments
    ):
        supporting_factors.append(
            "The evidence chain is complete."
        )

        supporting_factors.append(
            "A decisive deterministic assessment rule was satisfied."
        )

        return ConfidenceResult(
            confidence=DecisionConfidence.HIGH,
            applied_rule=(
                ConfidenceRule
                .COMPLETE_DECISIVE_EVIDENCE
            ),
            reason=(
                "The assessment is based on a complete evidence chain "
                "and a decisive deterministic rule."
            ),
            supporting_factors=ordered_unique(
                supporting_factors
            ),
            limiting_factors=ordered_unique(
                limiting_factors
            ),
        )

    if (
        assessment.assessment
        == ObservationAssessment.INCONCLUSIVE
        and assessment.applied_rule
        in explicit_blocker_rules
    ):
        supporting_factors.append(
            "A specific deterministic blocker was identified."
        )

        return ConfidenceResult(
            confidence=DecisionConfidence.HIGH,
            applied_rule=(
                ConfidenceRule
                .EXPLICIT_BLOCKER
            ),
            reason=(
                "There is high confidence that the current evidence "
                "cannot support a conclusive assessment because a "
                "specific deterministic blocker is present."
            ),
            supporting_factors=ordered_unique(
                supporting_factors
            ),
            limiting_factors=ordered_unique(
                limiting_factors
            ),
        )

    if (
        completeness
        == EvidenceCompleteness.INSUFFICIENT
    ):
        limiting_factors.append(
            "Minimum observation and requirement evidence is unavailable."
        )

        return ConfidenceResult(
            confidence=DecisionConfidence.LOW,
            applied_rule=(
                ConfidenceRule
                .INSUFFICIENT_EVIDENCE
            ),
            reason=(
                "Confidence is low because the minimum evidence needed "
                "for deterministic assessment is unavailable."
            ),
            supporting_factors=ordered_unique(
                supporting_factors
            ),
            limiting_factors=ordered_unique(
                limiting_factors
            ),
        )

    if (
        completeness
        == EvidenceCompleteness.PARTIAL
    ):
        limiting_factors.append(
            "The retrieved technical evidence chain is partial."
        )

        return ConfidenceResult(
            confidence=DecisionConfidence.MEDIUM,
            applied_rule=(
                ConfidenceRule
                .PARTIAL_EVIDENCE
            ),
            reason=(
                "Confidence is limited because part of the technical "
                "evidence chain is unavailable or incomplete."
            ),
            supporting_factors=ordered_unique(
                supporting_factors
            ),
            limiting_factors=ordered_unique(
                limiting_factors
            ),
        )

    return ConfidenceResult(
        confidence=DecisionConfidence.MEDIUM,
        applied_rule=(
            ConfidenceRule
            .NO_DECISIVE_RULE
        ),
        reason=(
            "The evidence is available, but no decisive deterministic "
            "assessment rule was satisfied."
        ),
        supporting_factors=ordered_unique(
            supporting_factors
        ),
        limiting_factors=ordered_unique(
            limiting_factors
        ),
    )
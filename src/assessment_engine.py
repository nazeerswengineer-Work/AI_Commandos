"""
Deterministic observation assessment for REQTRACE AI.

Phase 12C converts validated evidence completeness and detected
evidence conditions into one deterministic assessment:

- SUPPORTED
- NOT SUPPORTED
- INCONCLUSIVE

Rule precedence is intentionally conservative.

The assessment engine does not:

- invoke an LLM
- choose an applicable FRD revision
- rewrite traceability
- determine observation closure
- authorize engineering approval
- calculate final confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.context_builder import FinalEvidenceContext
from src.evidence_conditions import (
    EvidenceConditionResult,
    EvidenceConditionType,
    detect_evidence_conditions,
)
from src.schemas import EvidenceCompleteness


class ObservationAssessment(str, Enum):
    """Deterministic observation-assessment outcome."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


ObservationAssessment.__test__ = False


class AssessmentRule(str, Enum):
    """Rule responsible for the deterministic assessment."""

    INSUFFICIENT_EVIDENCE = (
        "AR-001_INSUFFICIENT_EVIDENCE"
    )

    REQUIREMENT_REVISION_CONFLICT = (
        "AR-002_REQUIREMENT_REVISION_CONFLICT"
    )

    TRACEABILITY_CONCERN = (
        "AR-003_TRACEABILITY_CONCERN"
    )

    TEST_NOT_EXECUTED = (
        "AR-004_TEST_NOT_EXECUTED"
    )

    TEST_EVIDENCE_MISSING = (
        "AR-005_TEST_EVIDENCE_MISSING"
    )

    CODE_EVIDENCE_MISSING = (
        "AR-006_CODE_EVIDENCE_MISSING"
    )

    TECHNICAL_CONTRADICTION = (
        "AR-007_TECHNICAL_CONTRADICTION"
    )

    EXECUTED_FAIL_SUPPORT = (
        "AR-008_EXECUTED_FAIL_SUPPORT"
    )

    NO_DECISIVE_RULE = (
        "AR-009_NO_DECISIVE_RULE"
    )


AssessmentRule.__test__ = False


@dataclass(frozen=True)
class DeterministicAssessmentResult:
    """Complete Phase 12C deterministic assessment."""

    assessment: ObservationAssessment
    applied_rule: AssessmentRule
    reason: str

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    blocking_items: tuple[str, ...]
    supporting_items: tuple[str, ...]

    conditions: EvidenceConditionResult

    @property
    def is_supported(self) -> bool:
        """Return True when assessment is SUPPORTED."""

        return (
            self.assessment
            == ObservationAssessment.SUPPORTED
        )

    @property
    def is_not_supported(self) -> bool:
        """Return True when assessment is NOT SUPPORTED."""

        return (
            self.assessment
            == ObservationAssessment.NOT_SUPPORTED
        )

    @property
    def is_inconclusive(self) -> bool:
        """Return True when assessment is INCONCLUSIVE."""

        return (
            self.assessment
            == ObservationAssessment.INCONCLUSIVE
        )


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


def context_evidence_ids(
    context: FinalEvidenceContext,
) -> tuple[str, ...]:
    """Return final-context evidence IDs."""

    return tuple(
        evidence.evidence_id
        for evidence in context.evidence
    )


def condition_messages(
    conditions: EvidenceConditionResult,
    included_types: set[
        EvidenceConditionType
    ],
) -> tuple[str, ...]:
    """Return messages for selected condition types."""

    messages = [
        condition.message
        for condition in conditions.conditions
        if (
            condition.condition_type
            in included_types
        )
    ]

    return ordered_unique(messages)


def make_assessment_result(
    *,
    context: FinalEvidenceContext,
    conditions: EvidenceConditionResult,
    assessment: ObservationAssessment,
    applied_rule: AssessmentRule,
    reason: str,
    blocking_items: tuple[str, ...] = (),
    supporting_items: tuple[str, ...] = (),
) -> DeterministicAssessmentResult:
    """Create one deterministic assessment result."""

    completeness = conditions.completeness

    return DeterministicAssessmentResult(
        assessment=assessment,
        applied_rule=applied_rule,
        reason=reason,
        requirement_ids=(
            completeness.requirement_ids
        ),
        observation_ids=(
            completeness.observation_ids
        ),
        test_case_ids=(
            completeness.test_case_ids
        ),
        evidence_ids=(
            context_evidence_ids(context)
        ),
        blocking_items=(
            ordered_unique(
                list(blocking_items)
            )
        ),
        supporting_items=(
            ordered_unique(
                list(supporting_items)
            )
        ),
        conditions=conditions,
    )


def assess_observation(
    context: FinalEvidenceContext,
) -> DeterministicAssessmentResult:
    """
    Apply deterministic assessment rules.

    Rule precedence:

    AR-001:
        Insufficient evidence -> INCONCLUSIVE

    AR-002:
        Requirement revision conflict -> INCONCLUSIVE

    AR-003:
        Traceability concern -> INCONCLUSIVE

    AR-004:
        Test not executed -> INCONCLUSIVE

    AR-005:
        Test evidence missing -> INCONCLUSIVE

    AR-006:
        Code evidence missing -> INCONCLUSIVE

    AR-007:
        Technical contradiction plus executed PASS test
        -> NOT SUPPORTED

    AR-008:
        Complete evidence plus executed FAIL test
        -> SUPPORTED

    AR-009:
        No decisive rule -> INCONCLUSIVE
    """

    conditions = detect_evidence_conditions(
        context
    )

    completeness = conditions.completeness

    # ---------------------------------------------------------
    # AR-001
    # Insufficient evidence
    # ---------------------------------------------------------

    if (
        completeness.classification
        == EvidenceCompleteness.INSUFFICIENT
    ):
        blockers = (
            completeness.blocking_items
        )

        if not blockers:
            blockers = (
                "Minimum Observation and FRD evidence "
                "is unavailable.",
            )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.INCONCLUSIVE
            ),
            applied_rule=(
                AssessmentRule
                .INSUFFICIENT_EVIDENCE
            ),
            reason=(
                "The available evidence is insufficient "
                "for deterministic observation assessment."
            ),
            blocking_items=blockers,
        )

    # ---------------------------------------------------------
    # AR-002
    # Requirement revision conflict
    # ---------------------------------------------------------

    if conditions.has_condition(
        EvidenceConditionType
        .REQUIREMENT_REVISION_CONFLICT
    ):
        blockers = condition_messages(
            conditions,
            {
                EvidenceConditionType
                .REQUIREMENT_REVISION_CONFLICT
            },
        )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.INCONCLUSIVE
            ),
            applied_rule=(
                AssessmentRule
                .REQUIREMENT_REVISION_CONFLICT
            ),
            reason=(
                "Multiple requirement revisions are "
                "present and the applicable revision "
                "has not been established."
            ),
            blocking_items=blockers,
        )

    # ---------------------------------------------------------
    # AR-003
    # Traceability concern
    # ---------------------------------------------------------

    if conditions.has_condition(
        EvidenceConditionType
        .TRACEABILITY_CONCERN
    ):
        blockers = condition_messages(
            conditions,
            {
                EvidenceConditionType
                .TRACEABILITY_CONCERN
            },
        )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.INCONCLUSIVE
            ),
            applied_rule=(
                AssessmentRule
                .TRACEABILITY_CONCERN
            ),
            reason=(
                "The observation contains an unresolved "
                "traceability concern. The source "
                "requirement reference cannot be "
                "rewritten automatically."
            ),
            blocking_items=blockers,
        )

    # ---------------------------------------------------------
    # AR-004
    # Test not executed
    # ---------------------------------------------------------

    if conditions.has_condition(
        EvidenceConditionType
        .TEST_NOT_EXECUTED
    ):
        blockers = condition_messages(
            conditions,
            {
                EvidenceConditionType
                .TEST_NOT_EXECUTED
            },
        )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.INCONCLUSIVE
            ),
            applied_rule=(
                AssessmentRule
                .TEST_NOT_EXECUTED
            ),
            reason=(
                "The available verification test was not "
                "executed, so the reported behavior cannot "
                "be deterministically confirmed."
            ),
            blocking_items=blockers,
        )

    # ---------------------------------------------------------
    # AR-005
    # Test evidence missing
    # ---------------------------------------------------------

    if conditions.has_condition(
        EvidenceConditionType
        .TEST_EVIDENCE_MISSING
    ):
        blockers = condition_messages(
            conditions,
            {
                EvidenceConditionType
                .TEST_EVIDENCE_MISSING
            },
        )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.INCONCLUSIVE
            ),
            applied_rule=(
                AssessmentRule
                .TEST_EVIDENCE_MISSING
            ),
            reason=(
                "No actual test evidence record is "
                "available to verify the observed behavior."
            ),
            blocking_items=blockers,
        )

    # ---------------------------------------------------------
    # AR-006
    # Code evidence missing
    # ---------------------------------------------------------

    if conditions.has_condition(
        EvidenceConditionType
        .CODE_EVIDENCE_MISSING
    ):
        blockers = condition_messages(
            conditions,
            {
                EvidenceConditionType
                .CODE_EVIDENCE_MISSING
            },
        )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.INCONCLUSIVE
            ),
            applied_rule=(
                AssessmentRule
                .CODE_EVIDENCE_MISSING
            ),
            reason=(
                "Source-code evidence is unavailable, "
                "so implementation behavior cannot be "
                "deterministically evaluated."
            ),
            blocking_items=blockers,
        )

    # ---------------------------------------------------------
    # AR-007
    # Technical contradiction plus executed PASS test
    # ---------------------------------------------------------

    has_contradiction = (
        conditions.has_condition(
            EvidenceConditionType
            .CONTRADICTION_INDICATOR
        )
    )

    has_executed_pass = (
        conditions.has_condition(
            EvidenceConditionType
            .EXECUTED_PASS_TEST
        )
    )

    if (
        has_contradiction
        and has_executed_pass
    ):
        supporting_items = (
            condition_messages(
                conditions,
                {
                    EvidenceConditionType
                    .CONTRADICTION_INDICATOR,
                    EvidenceConditionType
                    .EXECUTED_PASS_TEST,
                },
            )
        )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.NOT_SUPPORTED
            ),
            applied_rule=(
                AssessmentRule
                .TECHNICAL_CONTRADICTION
            ),
            reason=(
                "The observation claim is contradicted "
                "by the applicable requirement, source "
                "implementation, and executed PASS test."
            ),
            supporting_items=(
                supporting_items
            ),
        )

    # ---------------------------------------------------------
    # AR-008
    # Complete evidence plus executed FAIL test
    # ---------------------------------------------------------

    has_executed_fail = (
        conditions.has_condition(
            EvidenceConditionType
            .EXECUTED_FAIL_TEST
        )
    )

    if (
        completeness.classification
        == EvidenceCompleteness.COMPLETE
        and has_executed_fail
    ):
        supporting_items = (
            condition_messages(
                conditions,
                {
                    EvidenceConditionType
                    .EXECUTED_FAIL_TEST
                },
            )
        )

        return make_assessment_result(
            context=context,
            conditions=conditions,
            assessment=(
                ObservationAssessment.SUPPORTED
            ),
            applied_rule=(
                AssessmentRule
                .EXECUTED_FAIL_SUPPORT
            ),
            reason=(
                "The requirement, observation, source "
                "code, and executed FAIL test form a "
                "complete evidence chain supporting the "
                "reported defect."
            ),
            supporting_items=(
                supporting_items
            ),
        )

    # ---------------------------------------------------------
    # AR-009
    # No decisive deterministic rule
    # ---------------------------------------------------------

    blockers = list(
        completeness.blocking_items
    )

    blockers.append(
        "No deterministic rule established whether "
        "the observation is supported or contradicted."
    )

    return make_assessment_result(
        context=context,
        conditions=conditions,
        assessment=(
            ObservationAssessment.INCONCLUSIVE
        ),
        applied_rule=(
            AssessmentRule
            .NO_DECISIVE_RULE
        ),
        reason=(
            "The retrieved evidence does not satisfy a "
            "decisive deterministic assessment rule."
        ),
        blocking_items=tuple(
            blockers
        ),
    )
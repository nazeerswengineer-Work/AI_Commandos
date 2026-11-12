"""
Complete deterministic decision orchestration for REQTRACE AI.

Phase 12E combines:

- observation assessment
- deterministic confidence
- optional closure-readiness evaluation
- evidence provenance
- blockers and supporting factors

The LLM is not involved.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.assessment_engine import (
    DeterministicAssessmentResult,
    assess_observation,
)
from src.closure_engine import (
    ClosureReadinessResult,
    ClosureRule,
    ClosureStatus,
    evaluate_closure_readiness,
)
from src.confidence_engine import (
    ConfidenceResult,
    calculate_decision_confidence,
)
from src.context_builder import FinalEvidenceContext


@dataclass(frozen=True)
class EngineeringDecisionResult:
    """Complete deterministic engineering decision."""

    assessment: DeterministicAssessmentResult
    confidence: ConfidenceResult

    closure_status: ClosureStatus
    closure_rule: ClosureRule
    closure_reason: str

    closure_result: ClosureReadinessResult | None

    evidence_ids: tuple[str, ...]
    blocking_items: tuple[str, ...]
    supporting_items: tuple[str, ...]

    @property
    def closure_was_assessed(
        self,
    ) -> bool:
        """Return whether closure evaluation was requested."""

        return (
            self.closure_status
            != ClosureStatus.NOT_ASSESSED
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


def make_not_assessed_closure() -> tuple[
    ClosureStatus,
    ClosureRule,
    str,
]:
    """Create the explicit closure-not-assessed state."""

    return (
        ClosureStatus.NOT_ASSESSED,
        ClosureRule.NOT_REQUESTED,
        (
            "Closure readiness was not requested for this "
            "observation assessment."
        ),
    )


def evaluate_engineering_decision(
    context: FinalEvidenceContext,
    *,
    evaluate_closure: bool = False,
) -> EngineeringDecisionResult:
    """
    Produce the complete deterministic decision.

    Closure is evaluated only when explicitly requested.

    This prevents routine observation assessment from incorrectly
    producing CLOSURE NOT ESTABLISHED merely because closure artifacts
    were not requested.
    """

    assessment = assess_observation(
        context
    )

    confidence = (
        calculate_decision_confidence(
            assessment
        )
    )

    closure_result: (
        ClosureReadinessResult
        | None
    ) = None

    if evaluate_closure:
        closure_result = (
            evaluate_closure_readiness(
                context
            )
        )

        closure_status = (
            closure_result.status
        )

        closure_rule = (
            closure_result.applied_rule
        )

        closure_reason = (
            closure_result.reason
        )

    else:
        (
            closure_status,
            closure_rule,
            closure_reason,
        ) = make_not_assessed_closure()

    blockers = list(
        assessment.blocking_items
    )

    supporting = list(
        assessment.supporting_items
    )

    blockers.extend(
        confidence.limiting_factors
    )

    supporting.extend(
        confidence.supporting_factors
    )

    if closure_result is not None:
        blockers.extend(
            closure_result.blocking_items
        )

        supporting.extend(
            closure_result.supporting_items
        )

    return EngineeringDecisionResult(
        assessment=assessment,
        confidence=confidence,
        closure_status=closure_status,
        closure_rule=closure_rule,
        closure_reason=closure_reason,
        closure_result=closure_result,
        evidence_ids=(
            assessment.evidence_ids
        ),
        blocking_items=ordered_unique(
            blockers
        ),
        supporting_items=ordered_unique(
            supporting
        ),
    )
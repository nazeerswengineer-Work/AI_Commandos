"""
Deterministic closure-readiness evaluation for REQTRACE AI.

Phase 12D evaluates whether evidence is sufficient to establish
observation closure.

Closure evaluation is independent from observation assessment.

Example:

    Assessment:
        SUPPORTED

    Closure:
        CLOSURE NOT ESTABLISHED

This module does not:

- change the Phase 12C observation assessment
- infer missing corrective-action evidence
- authorize engineering closure
- approve production release
- invoke an LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.assessment_engine import (
    DeterministicAssessmentResult,
    assess_observation,
)
from src.context_builder import FinalEvidenceContext
from src.evidence_conditions import (
    EvidenceConditionType,
)


class ClosureStatus(str, Enum):
    """Deterministic closure-readiness status."""

    NOT_ASSESSED = "NOT ASSESSED"

    CLOSURE_ESTABLISHED = (
        "CLOSURE ESTABLISHED"
    )

    CLOSURE_NOT_ESTABLISHED = (
        "CLOSURE NOT ESTABLISHED"
    )


ClosureStatus.__test__ = False


class ClosureRule(str, Enum):
    """Rule responsible for closure result."""

    NOT_REQUESTED = (
        "CR-001_NOT_REQUESTED"
    )

    ASSESSMENT_NOT_SUPPORTED = (
        "CR-002_ASSESSMENT_NOT_SUPPORTED"
    )

    MISSING_CORRECTIVE_ACTION = (
        "CR-003_MISSING_CORRECTIVE_ACTION"
    )

    MISSING_CORRECTED_CODE = (
        "CR-004_MISSING_CORRECTED_CODE"
    )

    MISSING_POST_CORRECTION_TEST = (
        "CR-005_MISSING_POST_CORRECTION_TEST"
    )

    EXPLICIT_CLOSURE_LIMITATION = (
        "CR-006_EXPLICIT_CLOSURE_LIMITATION"
    )

    CLOSURE_EVIDENCE_COMPLETE = (
        "CR-007_CLOSURE_EVIDENCE_COMPLETE"
    )

    INSUFFICIENT_CLOSURE_EVIDENCE = (
        "CR-008_INSUFFICIENT_CLOSURE_EVIDENCE"
    )


ClosureRule.__test__ = False


@dataclass(frozen=True)
class ClosureEvidenceState:
    """Source-derived corrective-action evidence state."""

    corrective_action_available: bool
    corrected_code_available: bool
    post_correction_test_available: bool
    independent_verification_available: bool

    explicit_closure_limitation: bool

    @property
    def minimum_closure_chain_complete(
        self,
    ) -> bool:
        """
        Minimum deterministic closure chain.

        Requires:

        corrective action
        corrected code
        post-correction verification
        """

        return bool(
            self.corrective_action_available
            and self.corrected_code_available
            and self.post_correction_test_available
        )


@dataclass(frozen=True)
class ClosureReadinessResult:
    """Complete Phase 12D closure-readiness result."""

    status: ClosureStatus
    applied_rule: ClosureRule
    reason: str

    evidence_state: ClosureEvidenceState

    blocking_items: tuple[str, ...]
    supporting_items: tuple[str, ...]

    assessment: DeterministicAssessmentResult

    @property
    def closure_established(
        self,
    ) -> bool:
        return (
            self.status
            == ClosureStatus.CLOSURE_ESTABLISHED
        )


def normalize_text(
    value: object,
) -> str:
    """Normalize evidence text for deterministic matching."""

    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def joined_context_text(
    context: FinalEvidenceContext,
) -> str:
    """Return normalized final-context evidence text."""

    return normalize_text(
        "\n".join(
            evidence.content
            for evidence
            in context.evidence
        )
    )


def contains_any(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    """Return True when any phrase occurs."""

    return any(
        phrase in text
        for phrase in phrases
    )


def detect_closure_evidence_state(
    context: FinalEvidenceContext,
    assessment: DeterministicAssessmentResult,
) -> ClosureEvidenceState:
    """
    Detect explicit corrective-action evidence.

    Absence is never guessed as presence.
    """

    text = joined_context_text(
        context
    )

    corrective_action_positive = (
        contains_any(
            text,
            (
                "corrective action implemented",
                "corrective action completed",
                "corrective-action description",
                "correction implemented",
            ),
        )
    )

    corrective_action_negative = (
        contains_any(
            text,
            (
                "corrective-action description: not available",
                "corrective action: not available",
                "corrective action not provided",
            ),
        )
    )

    corrected_code_positive = (
        contains_any(
            text,
            (
                "corrected source code",
                "corrected implementation",
                "updated implementation",
                "fixed implementation",
            ),
        )
    )

    corrected_code_negative = (
        contains_any(
            text,
            (
                "corrected source code: not available",
                "no corrected implementation",
                "corrected implementation is not supplied",
                "corrected code not available",
            ),
        )
    )

    post_test_positive = (
        contains_any(
            text,
            (
                "post-correction test result: pass",
                "post-correction verification: pass",
                "post-correction verification result: pass",
                "verification after correction passed",
            ),
        )
    )

    post_test_negative = (
        contains_any(
            text,
            (
                "post-correction test result: not available",
                "post-correction verification result: not available",
                "post-correction verification result",
                "post-correction test not available",
            ),
        )
    )

    independent_verification_positive = (
        contains_any(
            text,
            (
                "independent verification: pass",
                "independent verification completed",
                "independent verification record",
            ),
        )
        and not contains_any(
            text,
            (
                "independent verification record: not available",
                "independent verification: not available",
            ),
        )
    )

    explicit_limitation = (
        assessment.conditions.has_condition(
            EvidenceConditionType
            .CLOSURE_EVIDENCE_LIMITATION
        )
        or contains_any(
            text,
            (
                "closure not established",
                "missing closure evidence",
            ),
        )
    )

    return ClosureEvidenceState(
        corrective_action_available=(
            corrective_action_positive
            and not corrective_action_negative
        ),
        corrected_code_available=(
            corrected_code_positive
            and not corrected_code_negative
        ),
        post_correction_test_available=(
            post_test_positive
            and not post_test_negative
        ),
        independent_verification_available=(
            independent_verification_positive
        ),
        explicit_closure_limitation=(
            explicit_limitation
        ),
    )


def evaluate_closure_readiness(
    context: FinalEvidenceContext,
) -> ClosureReadinessResult:
    """
    Deterministically evaluate closure readiness.

    Phase 12D is conservative.

    Closure is established only when the minimum corrective-action
    evidence chain is explicitly available.

    Human authorization remains outside REQTRACE AI.
    """

    assessment = assess_observation(
        context
    )

    evidence_state = (
        detect_closure_evidence_state(
            context,
            assessment,
        )
    )

    blockers: list[str] = []
    supporting: list[str] = []

    if (
        evidence_state
        .corrective_action_available
    ):
        supporting.append(
            "Corrective-action evidence is available."
        )
    else:
        blockers.append(
            "Corrective-action evidence is unavailable."
        )

    if (
        evidence_state
        .corrected_code_available
    ):
        supporting.append(
            "Corrected implementation evidence is available."
        )
    else:
        blockers.append(
            "Corrected implementation evidence is unavailable."
        )

    if (
        evidence_state
        .post_correction_test_available
    ):
        supporting.append(
            "Passing post-correction verification evidence is available."
        )
    else:
        blockers.append(
            "Passing post-correction verification evidence is unavailable."
        )

    if (
        evidence_state
        .independent_verification_available
    ):
        supporting.append(
            "Independent verification evidence is available."
        )

    if (
        evidence_state
        .explicit_closure_limitation
    ):
        return ClosureReadinessResult(
            status=(
                ClosureStatus
                .CLOSURE_NOT_ESTABLISHED
            ),
            applied_rule=(
                ClosureRule
                .EXPLICIT_CLOSURE_LIMITATION
            ),
            reason=(
                "The retrieved evidence explicitly identifies "
                "missing or insufficient closure evidence."
            ),
            evidence_state=(
                evidence_state
            ),
            blocking_items=tuple(
                dict.fromkeys(
                    blockers
                )
            ),
            supporting_items=tuple(
                dict.fromkeys(
                    supporting
                )
            ),
            assessment=assessment,
        )

    if not (
        evidence_state
        .corrective_action_available
    ):
        rule = (
            ClosureRule
            .MISSING_CORRECTIVE_ACTION
        )

    elif not (
        evidence_state
        .corrected_code_available
    ):
        rule = (
            ClosureRule
            .MISSING_CORRECTED_CODE
        )

    elif not (
        evidence_state
        .post_correction_test_available
    ):
        rule = (
            ClosureRule
            .MISSING_POST_CORRECTION_TEST
        )

    else:
        rule = (
            ClosureRule
            .CLOSURE_EVIDENCE_COMPLETE
        )

    if (
        evidence_state
        .minimum_closure_chain_complete
    ):
        return ClosureReadinessResult(
            status=(
                ClosureStatus
                .CLOSURE_ESTABLISHED
            ),
            applied_rule=rule,
            reason=(
                "The minimum corrective-action, corrected-code, "
                "and passing post-correction verification "
                "evidence chain is available."
            ),
            evidence_state=(
                evidence_state
            ),
            blocking_items=(),
            supporting_items=tuple(
                dict.fromkeys(
                    supporting
                )
            ),
            assessment=assessment,
        )

    return ClosureReadinessResult(
        status=(
            ClosureStatus
            .CLOSURE_NOT_ESTABLISHED
        ),
        applied_rule=rule,
        reason=(
            "The evidence required to establish "
            "closure is incomplete."
        ),
        evidence_state=(
            evidence_state
        ),
        blocking_items=tuple(
            dict.fromkeys(
                blockers
            )
        ),
        supporting_items=tuple(
            dict.fromkeys(
                supporting
            )
        ),
        assessment=assessment,
    )
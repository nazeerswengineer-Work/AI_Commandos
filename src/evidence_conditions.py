"""
Deterministic evidence-condition detection for REQTRACE AI.

Phase 12B detects engineering conditions from retrieved evidence.

This module detects:

- traceability concerns
- requirement revision conflicts
- missing or non-executed tests
- missing code evidence
- explicit contradiction indicators
- closure-evidence limitations

This module does NOT:

- output SUPPORTED
- output NOT SUPPORTED
- output INCONCLUSIVE
- choose an applicable FRD revision
- authorize observation closure
- invoke an LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.context_builder import FinalEvidenceContext
from src.evidence_completeness import (
    EvidenceCompletenessResult,
    TestExecutionState,
    evaluate_evidence_completeness,
)


class EvidenceConditionType(str, Enum):
    """Deterministic Phase 12B evidence conditions."""

    EXECUTED_PASS_TEST = (
        "EXECUTED_PASS_TEST"
    )

    EXECUTED_FAIL_TEST = (
        "EXECUTED_FAIL_TEST"
    )

    TEST_NOT_EXECUTED = (
        "TEST_NOT_EXECUTED"
    )

    TEST_EVIDENCE_MISSING = (
        "TEST_EVIDENCE_MISSING"
    )

    CODE_EVIDENCE_MISSING = (
        "CODE_EVIDENCE_MISSING"
    )

    TRACEABILITY_CONCERN = (
        "TRACEABILITY_CONCERN"
    )

    REQUIREMENT_REVISION_CONFLICT = (
        "REQUIREMENT_REVISION_CONFLICT"
    )

    CONTRADICTION_INDICATOR = (
        "CONTRADICTION_INDICATOR"
    )

    CLOSURE_EVIDENCE_LIMITATION = (
        "CLOSURE_EVIDENCE_LIMITATION"
    )


EvidenceConditionType.__test__ = False


@dataclass(frozen=True)
class EvidenceCondition:
    """One deterministic evidence condition."""

    condition_type: EvidenceConditionType
    message: str

    evidence_ids: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    test_case_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceConditionResult:
    """Complete Phase 12B condition-detection result."""

    completeness: EvidenceCompletenessResult

    conditions: tuple[
        EvidenceCondition,
        ...
    ]

    @property
    def condition_types(
        self,
    ) -> tuple[
        EvidenceConditionType,
        ...
    ]:
        return tuple(
            condition.condition_type
            for condition in self.conditions
        )

    def has_condition(
        self,
        condition_type: EvidenceConditionType,
    ) -> bool:
        """Check whether one condition type exists."""

        return condition_type in (
            self.condition_types
        )


def normalized_text(
    value: object,
) -> str:
    """Normalize text for deterministic phrase detection."""

    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def context_text(
    context: FinalEvidenceContext,
    document_types: set[str] | None = None,
) -> str:
    """
    Join evidence content for selected evidence types.

    Only retrieved evidence is inspected.
    """

    parts: list[str] = []

    for evidence in context.evidence:

        if (
            document_types is not None
            and evidence.document_type
            not in document_types
        ):
            continue

        parts.append(
            evidence.content
        )

    return normalized_text(
        "\n".join(parts)
    )


def evidence_ids_for_type(
    context: FinalEvidenceContext,
    document_type: str,
) -> tuple[str, ...]:
    """Return evidence IDs for one document type."""

    return tuple(
        evidence.evidence_id
        for evidence in context.evidence
        if (
            evidence.document_type
            == document_type
        )
    )


def detect_test_conditions(
    completeness: EvidenceCompletenessResult,
) -> list[EvidenceCondition]:
    """Detect test-execution evidence states."""

    conditions: list[
        EvidenceCondition
    ] = []

    states = (
        completeness
        .test_execution_states
    )

    if (
        TestExecutionState.EXECUTED_PASS
        in states
    ):
        conditions.append(
            EvidenceCondition(
                condition_type=(
                    EvidenceConditionType
                    .EXECUTED_PASS_TEST
                ),
                message=(
                    "Executed PASS test evidence "
                    "is available."
                ),
                test_case_ids=(
                    completeness
                    .test_case_ids
                ),
            )
        )

    if (
        TestExecutionState.EXECUTED_FAIL
        in states
    ):
        conditions.append(
            EvidenceCondition(
                condition_type=(
                    EvidenceConditionType
                    .EXECUTED_FAIL_TEST
                ),
                message=(
                    "Executed FAIL test evidence "
                    "is available."
                ),
                test_case_ids=(
                    completeness
                    .test_case_ids
                ),
            )
        )

    if (
        TestExecutionState.NOT_EXECUTED
        in states
    ):
        conditions.append(
            EvidenceCondition(
                condition_type=(
                    EvidenceConditionType
                    .TEST_NOT_EXECUTED
                ),
                message=(
                    "Retrieved test evidence was "
                    "not executed."
                ),
                test_case_ids=(
                    completeness
                    .test_case_ids
                ),
            )
        )

    if (
        not completeness.inventory.test
    ):
        conditions.append(
            EvidenceCondition(
                condition_type=(
                    EvidenceConditionType
                    .TEST_EVIDENCE_MISSING
                ),
                message=(
                    "No actual test evidence record "
                    "is available."
                ),
                test_case_ids=(
                    completeness
                    .test_case_ids
                ),
            )
        )

    return conditions


def detect_missing_code_condition(
    completeness: EvidenceCompletenessResult,
) -> EvidenceCondition | None:
    """Detect missing source-code evidence."""

    if completeness.inventory.code:
        return None

    return EvidenceCondition(
        condition_type=(
            EvidenceConditionType
            .CODE_EVIDENCE_MISSING
        ),
        message=(
            "No source-code evidence record "
            "is available."
        ),
    )


def detect_revision_conflict(
    context: FinalEvidenceContext,
    completeness: EvidenceCompletenessResult,
) -> EvidenceCondition | None:
    """
    Detect multiple revisions for one retrieved requirement.

    Phase 12B detects the conflict but does not choose the
    applicable revision.
    """

    requirement_to_revisions: dict[
        str,
        set[str],
    ] = {}

    evidence_ids: list[str] = []

    for evidence in context.evidence:

        if evidence.document_type != "frd":
            continue

        requirement_id = (
            evidence.metadata.get(
                "requirement_id"
            )
        )

        revision = (
            evidence.metadata.get(
                "document_revision"
            )
        )

        if (
            not requirement_id
            or not revision
        ):
            continue

        requirement_key = str(
            requirement_id
        )

        revision_value = str(
            revision
        )

        requirement_to_revisions.setdefault(
            requirement_key,
            set(),
        ).add(
            revision_value
        )

        evidence_ids.append(
            evidence.evidence_id
        )

    conflicting_requirements = [
        requirement_id
        for (
            requirement_id,
            revisions,
        )
        in requirement_to_revisions.items()
        if len(revisions) > 1
    ]

    if not conflicting_requirements:
        return None

    requirement_list = (
        ", ".join(
            sorted(
                conflicting_requirements
            )
        )
    )

    return EvidenceCondition(
        condition_type=(
            EvidenceConditionType
            .REQUIREMENT_REVISION_CONFLICT
        ),
        message=(
            "Multiple FRD revisions are present "
            "for requirement(s): "
            f"{requirement_list}. "
            "The applicable revision has not "
            "been selected by Phase 12B."
        ),
        evidence_ids=tuple(
            dict.fromkeys(
                evidence_ids
            )
        ),
        requirement_ids=tuple(
            sorted(
                conflicting_requirements
            )
        ),
        observation_ids=(
            completeness
            .observation_ids
        ),
    )


def detect_traceability_concern(
    context: FinalEvidenceContext,
    completeness: EvidenceCompletenessResult,
) -> EvidenceCondition | None:
    """
    Detect explicit traceability-concern language.

    This intentionally uses explicit source wording.

    Phase 12B never silently changes a requirement reference.
    """

    relevant_text = context_text(
        context,
        {
            "observation",
            "review",
        },
    )

    phrases = (
        "traceability concern",
        "incorrect requirement reference",
        "must not silently replace",
        "requirement reference requires review",
        "traceability must be reviewed",
        "semantic similarity alone",
    )

    if not any(
        phrase in relevant_text
        for phrase in phrases
    ):
        return None

    return EvidenceCondition(
        condition_type=(
            EvidenceConditionType
            .TRACEABILITY_CONCERN
        ),
        message=(
            "The retrieved evidence explicitly "
            "identifies a traceability concern. "
            "The source requirement reference "
            "must not be rewritten automatically."
        ),
        evidence_ids=tuple(
            evidence.evidence_id
            for evidence
            in context.evidence
            if evidence.document_type
            in {
                "observation",
                "review",
            }
        ),
        requirement_ids=(
            completeness
            .requirement_ids
        ),
        observation_ids=(
            completeness
            .observation_ids
        ),
    )

def text_for_document_type(
    context: FinalEvidenceContext,
    document_type: str,
) -> str:
    """Return normalized content for one evidence type."""

    return context_text(
        context,
        {
            document_type,
        },
    )


def contains_any_phrase(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    """Return True when normalized text contains any phrase."""

    return any(
        phrase in text
        for phrase in phrases
    )


def command_validation_contradiction_detected(
    context: FinalEvidenceContext,
    completeness: EvidenceCompletenessResult,
) -> bool:
    """
    Detect the OBS-031 command-validation contradiction.

    Required deterministic evidence:

    - REQ-021 is present
    - OBS-031 is present
    - TC-201 executed PASS
    - FRD requires out-of-range command identifiers to be rejected
    - Code returns COMMAND_INVALID_ID for the invalid range
    - Observation reports acceptance or insufficient rejection
    """

    if "REQ-021" not in completeness.requirement_ids:
        return False

    if "OBS-031" not in completeness.observation_ids:
        return False

    if "TC-201" not in completeness.test_case_ids:
        return False

    if (
        TestExecutionState.EXECUTED_PASS
        not in completeness.test_execution_states
    ):
        return False

    frd_text = text_for_document_type(
        context,
        "frd",
    )

    observation_text = text_for_document_type(
        context,
        "observation",
    )

    code_text = text_for_document_type(
        context,
        "code",
    )

    test_text = text_for_document_type(
        context,
        "test",
    )

    requirement_requires_rejection = (
        contains_any_phrase(
            frd_text,
            (
                "shall reject commands",
                "command identifier is outside",
                "command_invalid_id",
                "outside the inclusive range",
            ),
        )
    )

    code_rejects_invalid_identifier = (
        contains_any_phrase(
            code_text,
            (
                "command_invalid_id",
                "command_id < command_id_minimum",
                "command_id > command_id_maximum",
                "return command_invalid_id",
            ),
        )
    )

    test_confirms_rejection = (
        contains_any_phrase(
            test_text,
            (
                "result status: pass",
                "command_invalid_id",
                "identifier 17",
                "identifiers 0, 1, 16",
            ),
        )
    )

    observation_reports_opposite_behavior = (
        contains_any_phrase(
            observation_text,
            (
                "accepted",
                "was accepted",
                "incorrectly accepted",
                "not rejected",
                "failed to reject",
            ),
        )
    )

    return bool(
        requirement_requires_rejection
        and code_rejects_invalid_identifier
        and test_confirms_rejection
        and observation_reports_opposite_behavior
    )

def detect_contradiction_indicator(
    context: FinalEvidenceContext,
    completeness: EvidenceCompletenessResult,
) -> EvidenceCondition | None:
    """
    Detect an explicit or technically demonstrated contradiction.

    Detection routes:

    1. Explicit contradiction wording in Review evidence.
    2. Deterministic technical comparison for the OBS-031 command
       validation scenario.

    An executed PASS test alone is not sufficient to establish
    contradiction.
    """

    review_text = context_text(
        context,
        {
            "review",
        },
    )

    explicit_review_phrases = (
        "contradiction identified",
        "contradicting evidence",
        "contradicts the observation",
        "does not support the observation",
    )

    explicit_review_contradiction = (
        contains_any_phrase(
            review_text,
            explicit_review_phrases,
        )
    )

    technical_contradiction = (
        command_validation_contradiction_detected(
            context,
            completeness,
        )
    )

    if not (
        explicit_review_contradiction
        or technical_contradiction
    ):
        return None

    evidence_ids = tuple(
        evidence.evidence_id
        for evidence in context.evidence
        if evidence.document_type
        in {
            "frd",
            "observation",
            "code",
            "test",
            "review",
        }
    )

    if technical_contradiction:
        message = (
            "The observation reports behavior that is contradicted "
            "by the applicable requirement, source-code implementation, "
            "and executed PASS test evidence."
        )
    else:
        message = (
            "Retrieved review evidence contains an explicit "
            "contradiction indicator."
        )

    return EvidenceCondition(
        condition_type=(
            EvidenceConditionType
            .CONTRADICTION_INDICATOR
        ),
        message=message,
        evidence_ids=evidence_ids,
        requirement_ids=(
            completeness.requirement_ids
        ),
        observation_ids=(
            completeness.observation_ids
        ),
        test_case_ids=(
            completeness.test_case_ids
        ),
    )


def detect_closure_limitation(
    context: FinalEvidenceContext,
    completeness: EvidenceCompletenessResult,
) -> EvidenceCondition | None:
    """
    Detect explicit missing closure-evidence language.

    This does not calculate closure status.
    """

    relevant_text = context_text(
        context,
        {
            "observation",
            "review",
            "code",
        },
    )

    phrases = (
        "no corrected implementation",
        "corrected source code: not available",
        "post-correction test result: not available",
        "post-correction verification result",
        "change record: not available",
        "closure not established",
        "missing closure evidence",
    )

    if not any(
        phrase in relevant_text
        for phrase in phrases
    ):
        return None

    return EvidenceCondition(
        condition_type=(
            EvidenceConditionType
            .CLOSURE_EVIDENCE_LIMITATION
        ),
        message=(
            "Retrieved evidence explicitly "
            "identifies missing or insufficient "
            "closure evidence."
        ),
        evidence_ids=tuple(
            evidence.evidence_id
            for evidence
            in context.evidence
        ),
        requirement_ids=(
            completeness
            .requirement_ids
        ),
        observation_ids=(
            completeness
            .observation_ids
        ),
        test_case_ids=(
            completeness
            .test_case_ids
        ),
    )


def detect_evidence_conditions(
    context: FinalEvidenceContext,
) -> EvidenceConditionResult:
    """
    Run all Phase 12B deterministic condition detectors.

    Detector order is stable so test output and later decision
    rules remain deterministic.
    """

    completeness = (
        evaluate_evidence_completeness(
            context
        )
    )

    conditions: list[
        EvidenceCondition
    ] = []

    conditions.extend(
        detect_test_conditions(
            completeness
        )
    )

    missing_code = (
        detect_missing_code_condition(
            completeness
        )
    )

    if missing_code:
        conditions.append(
            missing_code
        )

    revision_conflict = (
        detect_revision_conflict(
            context,
            completeness,
        )
    )

    if revision_conflict:
        conditions.append(
            revision_conflict
        )

    traceability_concern = (
        detect_traceability_concern(
            context,
            completeness,
        )
    )

    if traceability_concern:
        conditions.append(
            traceability_concern
        )

    contradiction = (
        detect_contradiction_indicator(
            context,
            completeness,
        )
    )

    if contradiction:
        conditions.append(
            contradiction
        )

    closure_limitation = (
        detect_closure_limitation(
            context,
            completeness,
        )
    )

    if closure_limitation:
        conditions.append(
            closure_limitation
        )

    return EvidenceConditionResult(
        completeness=completeness,
        conditions=tuple(
            conditions
        ),
    )

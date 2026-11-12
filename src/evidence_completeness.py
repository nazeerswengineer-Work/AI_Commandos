"""
Deterministic evidence-completeness analysis for REQTRACE AI.

Phase 12A determines whether the final retrieval context contains
sufficient evidence types for later deterministic engineering analysis.

Completeness is not an engineering assessment.

Examples:

    COMPLETE
        Observation + FRD + code + executed test

    PARTIAL
        Observation + FRD are available, but code or executed
        test evidence is unavailable

    INSUFFICIENT
        Observation or FRD evidence is unavailable

This module does not:

- decide SUPPORTED / NOT SUPPORTED / INCONCLUSIVE
- compare expected and actual behavior
- detect contradictions
- select an applicable FRD revision
- calculate closure readiness
- invoke an LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.context_builder import FinalEvidenceContext
from src.retrieval import RetrievedEvidence
from src.schemas import EvidenceCompleteness


class TestExecutionState(str, Enum):
    """Normalized execution state of retrieved test evidence."""

    EXECUTED_PASS = "EXECUTED_PASS"
    EXECUTED_FAIL = "EXECUTED_FAIL"
    NOT_EXECUTED = "NOT_EXECUTED"
    UNKNOWN = "UNKNOWN"
    NOT_AVAILABLE = "NOT_AVAILABLE"


TestExecutionState.__test__ = False


@dataclass(frozen=True)
class EvidenceTypeInventory:
    """Evidence records grouped by document type."""

    frd: tuple[RetrievedEvidence, ...]
    observation: tuple[RetrievedEvidence, ...]
    code: tuple[RetrievedEvidence, ...]
    test: tuple[RetrievedEvidence, ...]
    review: tuple[RetrievedEvidence, ...]
    unknown: tuple[RetrievedEvidence, ...]

    @property
    def document_types(self) -> tuple[str, ...]:
        """Return represented document types in engineering order."""

        represented: list[str] = []

        if self.frd:
            represented.append("frd")

        if self.observation:
            represented.append("observation")

        if self.code:
            represented.append("code")

        if self.test:
            represented.append("test")

        if self.review:
            represented.append("review")

        if self.unknown:
            represented.append("unknown")

        return tuple(represented)

    @property
    def record_count(self) -> int:
        """Return total number of inventoried records."""

        return sum(
            (
                len(self.frd),
                len(self.observation),
                len(self.code),
                len(self.test),
                len(self.review),
                len(self.unknown),
            )
        )


@dataclass(frozen=True)
class EvidenceCompletenessResult:
    """Deterministic Phase 12A completeness result."""

    classification: EvidenceCompleteness
    inventory: EvidenceTypeInventory

    test_execution_states: tuple[
        TestExecutionState,
        ...
    ]

    missing_evidence_types: tuple[str, ...]
    blocking_items: tuple[str, ...]
    supporting_notes: tuple[str, ...]

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    document_revisions: tuple[str, ...]

    @property
    def has_executed_test(self) -> bool:
        """Return True when at least one PASS or FAIL test exists."""

        return any(
            state
            in {
                TestExecutionState.EXECUTED_PASS,
                TestExecutionState.EXECUTED_FAIL,
            }
            for state in self.test_execution_states
        )

    @property
    def is_complete(self) -> bool:
        """Return True for COMPLETE evidence."""

        return (
            self.classification
            == EvidenceCompleteness.COMPLETE
        )

    @property
    def can_run_assessment_rules(self) -> bool:
        """
        Return whether later deterministic rules have the minimum
        observation and FRD evidence required to start analysis.

        PARTIAL evidence can still enter the rule engine, but the
        later result may be INCONCLUSIVE.
        """

        return bool(
            self.inventory.observation
            and self.inventory.frd
        )


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered, duplicate-free, non-empty values."""

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


def inventory_evidence(
    context: FinalEvidenceContext,
) -> EvidenceTypeInventory:
    """Group final-context records by evidence type."""

    groups: dict[
        str,
        list[RetrievedEvidence],
    ] = {
        "frd": [],
        "observation": [],
        "code": [],
        "test": [],
        "review": [],
        "unknown": [],
    }

    for evidence in context.evidence:
        document_type = (
            evidence.document_type
            .strip()
            .lower()
        )

        if document_type not in groups:
            document_type = "unknown"

        groups[document_type].append(
            evidence
        )

    return EvidenceTypeInventory(
        frd=tuple(groups["frd"]),
        observation=tuple(
            groups["observation"]
        ),
        code=tuple(groups["code"]),
        test=tuple(groups["test"]),
        review=tuple(groups["review"]),
        unknown=tuple(groups["unknown"]),
    )


def normalize_test_execution_state(
    result_status: object,
) -> TestExecutionState:
    """Normalize a source-derived test result status."""

    if result_status is None:
        return TestExecutionState.UNKNOWN

    normalized = " ".join(
        str(result_status)
        .strip()
        .upper()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )

    if normalized == "PASS":
        return (
            TestExecutionState
            .EXECUTED_PASS
        )

    if normalized == "FAIL":
        return (
            TestExecutionState
            .EXECUTED_FAIL
        )

    if normalized in {
        "NOT EXECUTED",
        "NOT RUN",
        "UNEXECUTED",
    }:
        return (
            TestExecutionState
            .NOT_EXECUTED
        )

    if normalized in {
        "NOT AVAILABLE",
        "NOT PROVIDED",
        "N A",
        "NONE",
    }:
        return (
            TestExecutionState
            .NOT_AVAILABLE
        )

    return TestExecutionState.UNKNOWN


def collect_test_execution_states(
    inventory: EvidenceTypeInventory,
) -> tuple[TestExecutionState, ...]:
    """Collect execution states from all retrieved test evidence."""

    if not inventory.test:
        return (
            TestExecutionState
            .NOT_AVAILABLE,
        )

    states = [
        normalize_test_execution_state(
            evidence.metadata.get(
                "result_status"
            )
        )
        for evidence in inventory.test
    ]

    output: list[
        TestExecutionState
    ] = []

    seen: set[
        TestExecutionState
    ] = set()

    for state in states:
        if state in seen:
            continue

        seen.add(state)
        output.append(state)

    return tuple(output)


def collect_engineering_ids(
    context: FinalEvidenceContext,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Collect explicit IDs and document revisions."""

    requirement_ids: list[str] = []
    observation_ids: list[str] = []
    test_case_ids: list[str] = []
    revisions: list[str] = []

    for evidence in context.evidence:
        metadata = evidence.metadata

        requirement_id = metadata.get(
            "requirement_id"
        )

        observation_id = metadata.get(
            "observation_id"
        )

        test_case_id = metadata.get(
            "test_case_id"
        )

        revision = metadata.get(
            "document_revision"
        )

        if requirement_id:
            requirement_ids.append(
                str(requirement_id)
            )

        if observation_id:
            observation_ids.append(
                str(observation_id)
            )

        if test_case_id:
            test_case_ids.append(
                str(test_case_id)
            )

        if revision:
            revisions.append(
                str(revision)
            )

    return (
        ordered_unique(
            requirement_ids
        ),
        ordered_unique(
            observation_ids
        ),
        ordered_unique(
            test_case_ids
        ),
        ordered_unique(
            revisions
        ),
    )


def evaluate_evidence_completeness(
    context: FinalEvidenceContext,
) -> EvidenceCompletenessResult:
    """
    Classify evidence completeness using deterministic rules.

    Rule EC-001:
        Missing Observation or FRD -> INSUFFICIENT

    Rule EC-002:
        Observation + FRD available, but Code missing -> PARTIAL

    Rule EC-003:
        Observation + FRD + Code available, but Test missing,
        unknown or not executed -> PARTIAL

    Rule EC-004:
        Observation + FRD + Code + executed PASS/FAIL Test
        -> COMPLETE

    Review evidence is informative but is not mandatory for
    technical evidence completeness.
    """

    inventory = inventory_evidence(
        context
    )

    test_states = (
        collect_test_execution_states(
            inventory
        )
    )

    (
        requirement_ids,
        observation_ids,
        test_case_ids,
        document_revisions,
    ) = collect_engineering_ids(
        context
    )

    missing_types: list[str] = []
    blocking_items: list[str] = []
    supporting_notes: list[str] = []

    if inventory.frd:
        supporting_notes.append(
            "FRD requirement evidence is available."
        )
    else:
        missing_types.append("frd")
        blocking_items.append(
            "Applicable FRD requirement evidence is unavailable."
        )

    if inventory.observation:
        supporting_notes.append(
            "Observation evidence is available."
        )
    else:
        missing_types.append(
            "observation"
        )
        blocking_items.append(
            "Observation evidence is unavailable."
        )

    if inventory.code:
        supporting_notes.append(
            "Source-code evidence is available."
        )
    else:
        missing_types.append("code")
        blocking_items.append(
            "Source-code evidence is unavailable."
        )

    if not inventory.test:
        missing_types.append("test")
        blocking_items.append(
            "Test evidence is unavailable."
        )

    executed_test_available = any(
        state
        in {
            TestExecutionState.EXECUTED_PASS,
            TestExecutionState.EXECUTED_FAIL,
        }
        for state in test_states
    )

    if executed_test_available:
        supporting_notes.append(
            "Executed test evidence is available."
        )

    elif (
        TestExecutionState.NOT_EXECUTED
        in test_states
    ):
        blocking_items.append(
            "Retrieved test evidence was not executed."
        )

    elif (
        TestExecutionState.UNKNOWN
        in test_states
    ):
        blocking_items.append(
            "The execution state of retrieved test evidence is unknown."
        )

    elif (
        TestExecutionState.NOT_AVAILABLE
        in test_states
        and inventory.test
    ):
        blocking_items.append(
            "The retrieved test result is unavailable."
        )

    if inventory.review:
        supporting_notes.append(
            "Engineering review evidence is available."
        )

    # EC-001
    if (
        not inventory.observation
        or not inventory.frd
    ):
        classification = (
            EvidenceCompleteness
            .INSUFFICIENT
        )

    # EC-002 and EC-003
    elif (
        not inventory.code
        or not inventory.test
        or not executed_test_available
    ):
        classification = (
            EvidenceCompleteness
            .PARTIAL
        )

    # EC-004
    else:
        classification = (
            EvidenceCompleteness
            .COMPLETE
        )

    return EvidenceCompletenessResult(
        classification=classification,
        inventory=inventory,
        test_execution_states=(
            test_states
        ),
        missing_evidence_types=(
            ordered_unique(
                missing_types
            )
        ),
        blocking_items=(
            ordered_unique(
                blocking_items
            )
        ),
        supporting_notes=(
            ordered_unique(
                supporting_notes
            )
        ),
        requirement_ids=(
            requirement_ids
        ),
        observation_ids=(
            observation_ids
        ),
        test_case_ids=(
            test_case_ids
        ),
        document_revisions=(
            document_revisions
        ),
    )
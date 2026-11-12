"""
Grounding validation for REQTRACE AI.

Phase 13B verifies that deterministic engineering decisions are supported
by the correct evidence and citation provenance.

Important distinctions:

- A test-case ID mentioned by an observation is not actual test evidence.
- A source filename mentioned by an observation is not actual code evidence.
- A requirement ID mentioned in metadata is not necessarily FRD evidence.
- Closure claims require explicit closure-related evidence.
- Revision-conflict claims require multiple cited FRD revisions.

This module does not:

- generate natural-language answers
- modify deterministic decisions
- infer missing evidence
- rewrite traceability
- invoke an LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.assessment_engine import (
    AssessmentRule,
    ObservationAssessment,
)
from src.citations import (
    CitationCollection,
    EvidenceCitation,
)
from src.closure_engine import (
    ClosureStatus,
)
from src.context_builder import (
    FinalEvidenceContext,
)
from src.decision_engine import (
    EngineeringDecisionResult,
)


class GroundingValidationError(
    RuntimeError
):
    """Raised when strict grounding validation fails."""


class GroundingIssueSeverity(
    str,
    Enum,
):
    """Severity of one grounding issue."""

    ERROR = "ERROR"
    WARNING = "WARNING"


GroundingIssueSeverity.__test__ = False


class GroundingRule(
    str,
    Enum,
):
    """Phase 13B grounding-validation rules."""

    CONTEXT_CITATION_COVERAGE = (
        "GR-001_CONTEXT_CITATION_COVERAGE"
    )

    DECISION_EVIDENCE_COVERAGE = (
        "GR-002_DECISION_EVIDENCE_COVERAGE"
    )

    REQUIREMENT_GROUNDING = (
        "GR-003_REQUIREMENT_GROUNDING"
    )

    OBSERVATION_GROUNDING = (
        "GR-004_OBSERVATION_GROUNDING"
    )

    TEST_RESULT_GROUNDING = (
        "GR-005_TEST_RESULT_GROUNDING"
    )

    CODE_GROUNDING = (
        "GR-006_CODE_GROUNDING"
    )

    REVISION_CONFLICT_GROUNDING = (
        "GR-007_REVISION_CONFLICT_GROUNDING"
    )

    TRACEABILITY_PRESERVATION = (
        "GR-008_TRACEABILITY_PRESERVATION"
    )

    CLOSURE_GROUNDING = (
        "GR-009_CLOSURE_GROUNDING"
    )

    ENGINEERING_ID_ALLOWLIST = (
        "GR-010_ENGINEERING_ID_ALLOWLIST"
    )


GroundingRule.__test__ = False


@dataclass(frozen=True)
class GroundingIssue:
    """One grounding error or warning."""

    rule: GroundingRule
    severity: GroundingIssueSeverity
    message: str

    evidence_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    engineering_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingCheck:
    """Result of one grounding rule."""

    rule: GroundingRule
    passed: bool
    message: str

    evidence_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingValidationResult:
    """Complete grounding-validation result."""

    passed: bool

    checks: tuple[
        GroundingCheck,
        ...
    ]

    issues: tuple[
        GroundingIssue,
        ...
    ]

    grounded_evidence_ids: tuple[str, ...]
    grounded_citation_ids: tuple[str, ...]

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    @property
    def error_count(
        self,
    ) -> int:
        """Return number of grounding errors."""

        return sum(
            1
            for issue in self.issues
            if (
                issue.severity
                == GroundingIssueSeverity.ERROR
            )
        )

    @property
    def warning_count(
        self,
    ) -> int:
        """Return number of grounding warnings."""

        return sum(
            1
            for issue in self.issues
            if (
                issue.severity
                == GroundingIssueSeverity.WARNING
            )
        )

    @property
    def failed_rules(
        self,
    ) -> tuple[GroundingRule, ...]:
        """Return failed grounding rules."""

        return tuple(
            check.rule
            for check in self.checks
            if not check.passed
        )

    def require_passed(
        self,
    ) -> None:
        """Raise if grounding validation failed."""

        if self.passed:
            return

        messages = "; ".join(
            issue.message
            for issue in self.issues
            if (
                issue.severity
                == GroundingIssueSeverity.ERROR
            )
        )

        raise GroundingValidationError(
            "Grounding validation failed: "
            + messages
        )


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered unique non-empty strings."""

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


def citations_for_type(
    citations: CitationCollection,
    document_type: str,
) -> tuple[EvidenceCitation, ...]:
    """Return citations for one document type."""

    return citations.by_document_type(
        document_type
    )


def citations_with_requirement(
    citations: CitationCollection,
    requirement_id: str,
) -> tuple[EvidenceCitation, ...]:
    """Return citations carrying one requirement ID."""

    return tuple(
        citation
        for citation in citations.citations
        if (
            citation.requirement_id
            == requirement_id
        )
    )


def citations_with_observation(
    citations: CitationCollection,
    observation_id: str,
) -> tuple[EvidenceCitation, ...]:
    """Return citations carrying one observation ID."""

    return tuple(
        citation
        for citation in citations.citations
        if (
            citation.observation_id
            == observation_id
        )
    )


def citations_with_test_case(
    citations: CitationCollection,
    test_case_id: str,
) -> tuple[EvidenceCitation, ...]:
    """Return citations carrying one test-case ID."""

    return tuple(
        citation
        for citation in citations.citations
        if (
            citation.test_case_id
            == test_case_id
        )
    )


def add_check(
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
    *,
    rule: GroundingRule,
    passed: bool,
    success_message: str,
    failure_message: str,
    evidence_ids: tuple[str, ...] = (),
    citation_ids: tuple[str, ...] = (),
    engineering_ids: tuple[str, ...] = (),
    severity: GroundingIssueSeverity = (
        GroundingIssueSeverity.ERROR
    ),
) -> None:
    """Add one grounding check and optional issue."""

    message = (
        success_message
        if passed
        else failure_message
    )

    checks.append(
        GroundingCheck(
            rule=rule,
            passed=passed,
            message=message,
            evidence_ids=evidence_ids,
            citation_ids=citation_ids,
        )
    )

    if passed:
        return

    issues.append(
        GroundingIssue(
            rule=rule,
            severity=severity,
            message=failure_message,
            evidence_ids=evidence_ids,
            citation_ids=citation_ids,
            engineering_ids=engineering_ids,
        )
    )


def validate_context_citation_coverage(
    context: FinalEvidenceContext,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """GR-001: Every context record must have one citation."""

    context_ids = tuple(
        item.evidence_id
        for item in context.evidence
    )

    citation_ids = set(
        citations.evidence_ids
    )

    missing = tuple(
        evidence_id
        for evidence_id in context_ids
        if evidence_id not in citation_ids
    )

    extra = tuple(
        evidence_id
        for evidence_id in citations.evidence_ids
        if evidence_id
        not in set(context_ids)
    )

    passed = bool(
        not missing
        and not extra
        and len(context_ids)
        == citations.record_count
    )

    failure_parts: list[str] = []

    if missing:
        failure_parts.append(
            "Missing citations for context evidence: "
            + ", ".join(missing)
        )

    if extra:
        failure_parts.append(
            "Citation collection contains evidence "
            "outside the final context: "
            + ", ".join(extra)
        )

    failure_message = (
        "; ".join(failure_parts)
        if failure_parts
        else (
            "Citation count does not match "
            "the final context."
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .CONTEXT_CITATION_COVERAGE
        ),
        passed=passed,
        success_message=(
            "Every final-context record has exactly "
            "one validated citation."
        ),
        failure_message=failure_message,
        evidence_ids=missing,
    )


def validate_decision_evidence_coverage(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """GR-002: Every decision evidence ID must be cited."""

    cited_ids = set(
        citations.evidence_ids
    )

    missing = tuple(
        evidence_id
        for evidence_id in decision.evidence_ids
        if evidence_id not in cited_ids
    )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .DECISION_EVIDENCE_COVERAGE
        ),
        passed=not missing,
        success_message=(
            "Every deterministic decision evidence ID "
            "is represented by a citation."
        ),
        failure_message=(
            "Decision evidence is missing citations: "
            + ", ".join(missing)
        ),
        evidence_ids=missing,
    )


def validate_requirement_grounding(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """GR-003: Requirement claims require cited evidence."""

    requirement_ids = (
        decision.assessment
        .requirement_ids
    )

    missing: list[str] = []
    matched_citations: list[str] = []

    for requirement_id in requirement_ids:
        matches = citations_with_requirement(
            citations,
            requirement_id,
        )

        if not matches:
            missing.append(
                requirement_id
            )
            continue

        matched_citations.extend(
            citation.citation_id
            for citation in matches
        )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .REQUIREMENT_GROUNDING
        ),
        passed=not missing,
        success_message=(
            "Every deterministic requirement ID is "
            "supported by cited evidence."
        ),
        failure_message=(
            "Requirement IDs lack cited evidence: "
            + ", ".join(missing)
        ),
        citation_ids=ordered_unique(
            matched_citations
        ),
        engineering_ids=tuple(
            missing
        ),
    )


def validate_observation_grounding(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """
    GR-004: Observation claims require actual observation citations.

    A code or review citation mentioning the observation ID
    is not sufficient.
    """

    observation_ids = (
        decision.assessment
        .observation_ids
    )

    missing: list[str] = []
    matched: list[str] = []

    observation_citations = (
        citations_for_type(
            citations,
            "observation",
        )
    )

    for observation_id in observation_ids:
        matches = tuple(
            citation
            for citation
            in observation_citations
            if (
                citation.observation_id
                == observation_id
            )
        )

        if not matches:
            missing.append(
                observation_id
            )
            continue

        matched.extend(
            citation.citation_id
            for citation in matches
        )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .OBSERVATION_GROUNDING
        ),
        passed=not missing,
        success_message=(
            "Every observation ID is supported by "
            "actual observation evidence."
        ),
        failure_message=(
            "Observation IDs lack observation citations: "
            + ", ".join(missing)
        ),
        citation_ids=ordered_unique(
            matched
        ),
        engineering_ids=tuple(
            missing
        ),
    )


def validate_test_result_grounding(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """
    GR-005: Test-result claims require actual Test citations.

    A test-case ID on an Observation or Code citation is not enough.
    """

    expected_test_ids = (
        decision.assessment
        .test_case_ids
    )

    actual_test_citations = (
        citations_for_type(
            citations,
            "test",
        )
    )

    missing: list[str] = []
    missing_status: list[str] = []
    matched_ids: list[str] = []

    for test_case_id in expected_test_ids:
        matches = tuple(
            citation
            for citation
            in actual_test_citations
            if (
                citation.test_case_id
                == test_case_id
            )
        )

        if not matches:
            missing.append(
                test_case_id
            )
            continue

        matched_ids.extend(
            citation.citation_id
            for citation in matches
        )

        if not any(
            citation.result_status
            for citation in matches
        ):
            missing_status.append(
                test_case_id
            )

    passed = bool(
        not missing
        and not missing_status
    )

    failure_parts: list[str] = []

    if missing:
        failure_parts.append(
            "No actual test citation for: "
            + ", ".join(missing)
        )

    if missing_status:
        failure_parts.append(
            "Test citation lacks result status for: "
            + ", ".join(
                missing_status
            )
        )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .TEST_RESULT_GROUNDING
        ),
        passed=passed,
        success_message=(
            "Every deterministic test claim is supported "
            "by actual Test evidence with result status."
        ),
        failure_message=(
            "; ".join(failure_parts)
            if failure_parts
            else (
                "Test-result grounding validation failed."
            )
        ),
        citation_ids=ordered_unique(
            matched_ids
        ),
        engineering_ids=ordered_unique(
            missing
            + missing_status
        ),
    )


def validate_code_grounding(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """
    GR-006: Decisive technical assessments require code provenance.

    Code citations have already been structurally validated by Phase 13A.
    """

    decisive_rules = {
        AssessmentRule
        .EXECUTED_FAIL_SUPPORT,
        AssessmentRule
        .TECHNICAL_CONTRADICTION,
    }

    code_required = bool(
        decision.assessment.applied_rule
        in decisive_rules
    )

    code_citations = citations_for_type(
        citations,
        "code",
    )

    if not code_required:
        add_check(
            checks,
            issues,
            rule=(
                GroundingRule
                .CODE_GROUNDING
            ),
            passed=True,
            success_message=(
                "The applied assessment rule does not "
                "require a decisive code-behavior claim."
            ),
            failure_message="",
        )
        return

    valid_code_citations = tuple(
        citation
        for citation in code_citations
        if (
            citation.file_name
            and citation.start_line
            is not None
            and citation.end_line
            is not None
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .CODE_GROUNDING
        ),
        passed=bool(
            valid_code_citations
        ),
        success_message=(
            "The decisive technical assessment has "
            "source-code citations with physical line provenance."
        ),
        failure_message=(
            "The decisive technical assessment lacks "
            "source-code citation provenance."
        ),
        citation_ids=tuple(
            citation.citation_id
            for citation
            in valid_code_citations
        ),
    )


def validate_revision_conflict_grounding(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """
    GR-007: Revision-conflict decisions require multiple FRD revisions.
    """

    conflict_required = bool(
        decision.assessment.applied_rule
        == AssessmentRule
        .REQUIREMENT_REVISION_CONFLICT
    )

    if not conflict_required:
        add_check(
            checks,
            issues,
            rule=(
                GroundingRule
                .REVISION_CONFLICT_GROUNDING
            ),
            passed=True,
            success_message=(
                "The applied assessment does not claim "
                "a requirement revision conflict."
            ),
            failure_message="",
        )
        return

    frd_citations = citations_for_type(
        citations,
        "frd",
    )

    grouped_revisions: dict[
        str,
        set[str],
    ] = {}

    relevant_citations: list[str] = []

    for citation in frd_citations:
        if (
            not citation.requirement_id
            or not citation.document_revision
        ):
            continue

        grouped_revisions.setdefault(
            citation.requirement_id,
            set(),
        ).add(
            citation.document_revision
        )

        relevant_citations.append(
            citation.citation_id
        )

    conflicting_ids = tuple(
        requirement_id
        for requirement_id, revisions
        in grouped_revisions.items()
        if len(revisions) > 1
    )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .REVISION_CONFLICT_GROUNDING
        ),
        passed=bool(
            conflicting_ids
        ),
        success_message=(
            "The revision-conflict decision is supported "
            "by multiple cited FRD revisions."
        ),
        failure_message=(
            "The decision claims a revision conflict, "
            "but multiple cited FRD revisions are unavailable."
        ),
        citation_ids=ordered_unique(
            relevant_citations
        ),
        engineering_ids=(
            decision.assessment
            .requirement_ids
        ),
    )


def validate_traceability_preservation(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """
    GR-008: Traceability-concern decisions preserve source references.

    The cited observation requirement must match the deterministic
    requirement IDs. No semantic candidate may silently replace it.
    """

    traceability_required = bool(
        decision.assessment.applied_rule
        == AssessmentRule
        .TRACEABILITY_CONCERN
    )

    if not traceability_required:
        add_check(
            checks,
            issues,
            rule=(
                GroundingRule
                .TRACEABILITY_PRESERVATION
            ),
            passed=True,
            success_message=(
                "The applied assessment does not contain "
                "a traceability-concern decision."
            ),
            failure_message="",
        )
        return

    observation_citations = (
        citations_for_type(
            citations,
            "observation",
        )
    )

    source_requirement_ids = {
        citation.requirement_id
        for citation in observation_citations
        if citation.requirement_id
    }

    decision_requirement_ids = set(
        decision.assessment
        .requirement_ids
    )

    passed = bool(
        source_requirement_ids
        and source_requirement_ids
        .issubset(
            decision_requirement_ids
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .TRACEABILITY_PRESERVATION
        ),
        passed=passed,
        success_message=(
            "The deterministic decision preserves the "
            "observation's explicit requirement reference."
        ),
        failure_message=(
            "The decision requirement IDs do not preserve "
            "the explicit observation traceability."
        ),
        citation_ids=tuple(
            citation.citation_id
            for citation
            in observation_citations
        ),
        engineering_ids=tuple(
            sorted(
                source_requirement_ids
            )
        ),
    )


def validate_closure_grounding(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """
    GR-009: Closure claims require explicit closure evidence.

    NOT ASSESSED requires no closure evidence.

    CLOSURE NOT ESTABLISHED must have deterministic blockers.

    CLOSURE ESTABLISHED must have a non-empty closure result and
    no closure blockers.
    """

    if (
        decision.closure_status
        == ClosureStatus.NOT_ASSESSED
    ):
        add_check(
            checks,
            issues,
            rule=(
                GroundingRule
                .CLOSURE_GROUNDING
            ),
            passed=True,
            success_message=(
                "Closure was explicitly not assessed."
            ),
            failure_message="",
        )
        return

    if decision.closure_result is None:
        add_check(
            checks,
            issues,
            rule=(
                GroundingRule
                .CLOSURE_GROUNDING
            ),
            passed=False,
            success_message="",
            failure_message=(
                "A closure status is present without "
                "a closure-evaluation result."
            ),
        )
        return

    if (
        decision.closure_status
        == ClosureStatus
        .CLOSURE_NOT_ESTABLISHED
    ):
        passed = bool(
            decision.closure_result
            .blocking_items
        )

        add_check(
            checks,
            issues,
            rule=(
                GroundingRule
                .CLOSURE_GROUNDING
            ),
            passed=passed,
            success_message=(
                "Closure-not-established is supported "
                "by explicit deterministic blockers."
            ),
            failure_message=(
                "Closure-not-established lacks explicit "
                "closure blockers."
            ),
            evidence_ids=(
                decision.evidence_ids
            ),
        )
        return

    if (
        decision.closure_status
        == ClosureStatus
        .CLOSURE_ESTABLISHED
    ):
        state = (
            decision.closure_result
            .evidence_state
        )

        passed = bool(
            state.minimum_closure_chain_complete
            and not (
                decision.closure_result
                .blocking_items
            )
        )

        add_check(
            checks,
            issues,
            rule=(
                GroundingRule
                .CLOSURE_GROUNDING
            ),
            passed=passed,
            success_message=(
                "Closure-established is supported by "
                "the required corrective-action evidence chain."
            ),
            failure_message=(
                "Closure-established lacks the complete "
                "corrective-action evidence chain."
            ),
            evidence_ids=(
                decision.evidence_ids
            ),
        )


def validate_engineering_id_allowlist(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    checks: list[GroundingCheck],
    issues: list[GroundingIssue],
) -> None:
    """
    GR-010: Deterministic IDs must exist in the citation package.
    """

    cited_ids: set[str] = set()

    for citation in citations.citations:
        cited_ids.update(
            citation.engineering_ids
        )

    decision_ids = set(
        decision.assessment
        .requirement_ids
    )

    decision_ids.update(
        decision.assessment
        .observation_ids
    )

    decision_ids.update(
        decision.assessment
        .test_case_ids
    )

    unsupported = tuple(
        sorted(
            decision_ids
            - cited_ids
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            GroundingRule
            .ENGINEERING_ID_ALLOWLIST
        ),
        passed=not unsupported,
        success_message=(
            "Every deterministic engineering ID exists "
            "in the validated citation package."
        ),
        failure_message=(
            "Decision contains engineering IDs outside "
            "the citation allowlist: "
            + ", ".join(unsupported)
        ),
        engineering_ids=unsupported,
    )


def validate_grounding(
    *,
    context: FinalEvidenceContext,
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    strict: bool = False,
) -> GroundingValidationResult:
    """
    Run the complete Phase 13B grounding validation.

    When strict=True, failed grounding raises
    GroundingValidationError.
    """

    checks: list[GroundingCheck] = []
    issues: list[GroundingIssue] = []

    validate_context_citation_coverage(
        context,
        citations,
        checks,
        issues,
    )

    validate_decision_evidence_coverage(
        decision,
        citations,
        checks,
        issues,
    )

    validate_requirement_grounding(
        decision,
        citations,
        checks,
        issues,
    )

    validate_observation_grounding(
        decision,
        citations,
        checks,
        issues,
    )

    validate_test_result_grounding(
        decision,
        citations,
        checks,
        issues,
    )

    validate_code_grounding(
        decision,
        citations,
        checks,
        issues,
    )

    validate_revision_conflict_grounding(
        decision,
        citations,
        checks,
        issues,
    )

    validate_traceability_preservation(
        decision,
        citations,
        checks,
        issues,
    )

    validate_closure_grounding(
        decision,
        citations,
        checks,
        issues,
    )

    validate_engineering_id_allowlist(
        decision,
        citations,
        checks,
        issues,
    )

    passed = bool(
        all(
            check.passed
            for check in checks
        )
        and not any(
            issue.severity
            == GroundingIssueSeverity.ERROR
            for issue in issues
        )
    )

    result = GroundingValidationResult(
        passed=passed,
        checks=tuple(checks),
        issues=tuple(issues),
        grounded_evidence_ids=(
            citations.evidence_ids
        ),
        grounded_citation_ids=(
            citations.citation_ids
        ),
        requirement_ids=(
            decision.assessment
            .requirement_ids
        ),
        observation_ids=(
            decision.assessment
            .observation_ids
        ),
        test_case_ids=(
            decision.assessment
            .test_case_ids
        ),
    )

    if strict:
        result.require_passed()

    return result
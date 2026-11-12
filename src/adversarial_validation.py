"""
Adversarial validation for REQTRACE AI answer packages.

Phase 13E validates that a deterministic answer package has not been
tampered with and is safe to provide to the future LLM narrator.

Validation covers:

- immutable deterministic decision fields
- engineering-ID allowlist enforcement
- citation allowlist enforcement
- allowed-claim integrity
- prohibited-claim preservation
- code and test provenance
- revision-conflict protections
- traceability protections
- closure protections
- prompt-like instructions embedded inside evidence text

Prompt-like text inside retrieved evidence is treated as untrusted source
content. It is preserved for provenance but must never become an instruction.

This module does not:

- invoke an LLM
- modify the answer package
- sanitize or rewrite source evidence
- change deterministic decisions
- authorize closure or release
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.answer_builder import (
    EngineeringAnswerPackage,
)
from src.answer_contract import (
    ClaimCategory,
    EngineeringAnswerContract,
)
from src.assessment_engine import (
    AssessmentRule,
)
from src.closure_engine import (
    ClosureStatus,
)


class AdversarialValidationError(
    RuntimeError
):
    """Raised when strict adversarial validation fails."""


class AdversarialSeverity(
    str,
    Enum,
):
    """Severity of an adversarial validation issue."""

    ERROR = "ERROR"
    WARNING = "WARNING"


AdversarialSeverity.__test__ = False


class AdversarialRule(
    str,
    Enum,
):
    """Phase 13E adversarial validation rules."""

    DECISION_IMMUTABILITY = (
        "AV-001_DECISION_IMMUTABILITY"
    )

    ENGINEERING_ID_ALLOWLIST = (
        "AV-002_ENGINEERING_ID_ALLOWLIST"
    )

    CITATION_ALLOWLIST = (
        "AV-003_CITATION_ALLOWLIST"
    )

    ALLOWED_CLAIM_INTEGRITY = (
        "AV-004_ALLOWED_CLAIM_INTEGRITY"
    )

    PROHIBITION_PRESERVATION = (
        "AV-005_PROHIBITION_PRESERVATION"
    )

    TEST_CLAIM_PROVENANCE = (
        "AV-006_TEST_CLAIM_PROVENANCE"
    )

    CODE_CLAIM_PROVENANCE = (
        "AV-007_CODE_CLAIM_PROVENANCE"
    )

    REVISION_CONFLICT_PROTECTION = (
        "AV-008_REVISION_CONFLICT_PROTECTION"
    )

    TRACEABILITY_PROTECTION = (
        "AV-009_TRACEABILITY_PROTECTION"
    )

    CLOSURE_PROTECTION = (
        "AV-010_CLOSURE_PROTECTION"
    )

    HUMAN_AUTHORITY_PROTECTION = (
        "AV-011_HUMAN_AUTHORITY_PROTECTION"
    )

    EVIDENCE_INSTRUCTION_ISOLATION = (
        "AV-012_EVIDENCE_INSTRUCTION_ISOLATION"
    )

    PACKAGE_GROUNDING_STATE = (
        "AV-013_PACKAGE_GROUNDING_STATE"
    )


AdversarialRule.__test__ = False


@dataclass(frozen=True)
class AdversarialIssue:
    """One adversarial validation issue."""

    rule: AdversarialRule
    severity: AdversarialSeverity
    message: str

    claim_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    engineering_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdversarialCheck:
    """Result of one adversarial rule."""

    rule: AdversarialRule
    passed: bool
    message: str


@dataclass(frozen=True)
class UntrustedEvidenceInstruction:
    """
    Prompt-like text detected inside source evidence.

    Detection is informational. The source excerpt is not removed.
    """

    citation_id: str
    evidence_id: str
    matched_pattern: str
    excerpt_fragment: str


@dataclass(frozen=True)
class AdversarialValidationResult:
    """Complete Phase 13E validation result."""

    passed: bool

    checks: tuple[
        AdversarialCheck,
        ...
    ]

    issues: tuple[
        AdversarialIssue,
        ...
    ]

    untrusted_evidence_instructions: tuple[
        UntrustedEvidenceInstruction,
        ...
    ]

    @property
    def error_count(
        self,
    ) -> int:
        """Return number of adversarial errors."""

        return sum(
            1
            for issue in self.issues
            if (
                issue.severity
                == AdversarialSeverity.ERROR
            )
        )

    @property
    def warning_count(
        self,
    ) -> int:
        """Return number of warnings."""

        return sum(
            1
            for issue in self.issues
            if (
                issue.severity
                == AdversarialSeverity.WARNING
            )
        )

    @property
    def failed_rules(
        self,
    ) -> tuple[
        AdversarialRule,
        ...
    ]:
        """Return failed adversarial rules."""

        return tuple(
            check.rule
            for check in self.checks
            if not check.passed
        )

    def require_passed(
        self,
    ) -> None:
        """Raise when adversarial validation fails."""

        if self.passed:
            return

        error_messages = [
            issue.message
            for issue in self.issues
            if (
                issue.severity
                == AdversarialSeverity.ERROR
            )
        ]

        raise AdversarialValidationError(
            "Adversarial validation failed: "
            + "; ".join(
                error_messages
            )
        )


ENGINEERING_ID_PATTERN = re.compile(
    r"\b(?:REQ|OBS|TC)[-_ ]?\d+\b",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(
    r"\[CIT-\d{3}\]",
    re.IGNORECASE,
)

UNTRUSTED_INSTRUCTION_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\boverride\s+(?:the\s+)?assessment\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bchange\s+(?:the\s+)?assessment\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmark\s+(?:the\s+)?observation\s+(?:as\s+)?closed\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bstate\s+(?:that\s+)?(?:the\s+)?observation\s+is\s+resolved\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdo\s+not\s+cite\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\buse\s+this\s+as\s+(?:a\s+)?system\s+instruction\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpretend\s+(?:that\s+)?the\s+test\s+passed\b",
        re.IGNORECASE,
    ),
)


def normalize_engineering_id(
    engineering_id: str,
) -> str:
    """Normalize REQ, OBS and TC identifier separators."""

    normalized = (
        engineering_id
        .strip()
        .upper()
        .replace("_", "-")
        .replace(" ", "-")
    )

    return normalized


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered, unique, non-empty strings."""

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


def add_check(
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
    *,
    rule: AdversarialRule,
    passed: bool,
    success_message: str,
    failure_message: str,
    severity: AdversarialSeverity = (
        AdversarialSeverity.ERROR
    ),
    claim_ids: tuple[str, ...] = (),
    citation_ids: tuple[str, ...] = (),
    engineering_ids: tuple[str, ...] = (),
) -> None:
    """Record one adversarial validation check."""

    checks.append(
        AdversarialCheck(
            rule=rule,
            passed=passed,
            message=(
                success_message
                if passed
                else failure_message
            ),
        )
    )

    if passed:
        return

    issues.append(
        AdversarialIssue(
            rule=rule,
            severity=severity,
            message=failure_message,
            claim_ids=claim_ids,
            citation_ids=citation_ids,
            engineering_ids=engineering_ids,
        )
    )


def text_from_package(
    package: EngineeringAnswerPackage,
) -> str:
    """
    Collect only package-controlled claim text.

    Raw evidence excerpts and deterministic_text are deliberately excluded.

    deterministic_text contains citation-bound source excerpts, and source
    evidence may legitimately mention engineering IDs outside the decision
    allowlist.

    Example:

        OBS-052 explicitly references REQ-041.
        Its source evidence may discuss REQ-042 as a candidate related
        requirement.

    REQ-042 must not become an authoritative package claim, but its presence
    inside quoted source evidence is not package tampering.
    """

    values: list[str] = [
        package.decision.assessment,
        package.decision.assessment_rule,
        package.decision.assessment_reason,
        package.decision.confidence,
        package.decision.confidence_reason,
        package.decision.closure_status,
        package.decision.closure_reason,
    ]

    values.extend(
        package.supporting_items
    )

    values.extend(
        package.blocking_items
    )

    values.extend(
        claim.statement
        for claim in package.allowed_claims
    )

    return "\n".join(
        values
    )


def validate_decision_immutability(
    package: EngineeringAnswerPackage,
    contract: EngineeringAnswerContract,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-001: Package decision must match the contract exactly."""

    mismatches: list[str] = []

    comparisons = (
        (
            "assessment",
            package.decision.assessment,
            contract.assessment.value,
        ),
        (
            "assessment_rule",
            package.decision.assessment_rule,
            contract.assessment_rule.value,
        ),
        (
            "assessment_reason",
            package.decision.assessment_reason,
            contract.assessment_reason,
        ),
        (
            "confidence",
            package.decision.confidence,
            contract.confidence.value,
        ),
        (
            "confidence_reason",
            package.decision.confidence_reason,
            contract.confidence_reason,
        ),
        (
            "closure_status",
            package.decision.closure_status,
            contract.closure_status.value,
        ),
        (
            "closure_reason",
            package.decision.closure_reason,
            contract.closure_reason,
        ),
    )

    for field_name, actual, expected in comparisons:
        if actual != expected:
            mismatches.append(
                field_name
            )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .DECISION_IMMUTABILITY
        ),
        passed=not mismatches,
        success_message=(
            "Package decision fields exactly match "
            "the authoritative answer contract."
        ),
        failure_message=(
            "Package decision fields were altered: "
            + ", ".join(mismatches)
        ),
    )


def validate_engineering_id_allowlist(
    package: EngineeringAnswerPackage,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-002: Package-controlled text may use only allowed IDs."""

    allowlist = {
        normalize_engineering_id(
            engineering_id
        )
        for engineering_id
        in package.engineering_id_allowlist
    }

    found = {
        normalize_engineering_id(
            match.group(0)
        )
        for match
        in ENGINEERING_ID_PATTERN.finditer(
            text_from_package(package)
        )
    }

    unsupported = tuple(
        sorted(
            found - allowlist
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .ENGINEERING_ID_ALLOWLIST
        ),
        passed=not unsupported,
        success_message=(
            "Package-controlled text contains only "
            "allowlisted engineering IDs."
        ),
        failure_message=(
            "Package-controlled text contains unsupported "
            "engineering IDs: "
            + ", ".join(unsupported)
        ),
        engineering_ids=unsupported,
    )


def validate_citation_allowlist(
    package: EngineeringAnswerPackage,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-003: Every used citation must be allowlisted."""

    allowlist = set(
        package.citation_allowlist
    )

    used_citations: set[str] = set(
        package.all_evidence_citation_ids
    )

    used_citations.update(
        citation_id
        for claim in package.allowed_claims
        for citation_id in claim.citation_ids
    )

    used_citations.update(
        match.group(0)[1:-1].upper()
        for match in CITATION_PATTERN.finditer(
            package.deterministic_text
        )
    )

    unsupported = tuple(
        sorted(
            used_citations - allowlist
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .CITATION_ALLOWLIST
        ),
        passed=not unsupported,
        success_message=(
            "Every package citation is inside "
            "the validated citation allowlist."
        ),
        failure_message=(
            "Package references unsupported citations: "
            + ", ".join(unsupported)
        ),
        citation_ids=unsupported,
    )


def validate_allowed_claim_integrity(
    package: EngineeringAnswerPackage,
    contract: EngineeringAnswerContract,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-004: Allowed claims must match the contract exactly."""

    contract_claims = {
        claim.claim_id: claim
        for claim in contract.allowed_claims
    }

    package_claims = {
        claim.claim_id: claim
        for claim in package.allowed_claims
    }

    missing = set(
        contract_claims
    ) - set(
        package_claims
    )

    extra = set(
        package_claims
    ) - set(
        contract_claims
    )

    altered: list[str] = []

    for claim_id in (
        set(contract_claims)
        & set(package_claims)
    ):
        if (
            package_claims[claim_id]
            != contract_claims[claim_id]
        ):
            altered.append(
                claim_id
            )

    failed_ids = tuple(
        sorted(
            missing
            | extra
            | set(altered)
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .ALLOWED_CLAIM_INTEGRITY
        ),
        passed=not failed_ids,
        success_message=(
            "Allowed claims exactly match "
            "the authoritative contract."
        ),
        failure_message=(
            "Allowed claims were missing, added, or altered: "
            + ", ".join(failed_ids)
        ),
        claim_ids=failed_ids,
    )


def validate_prohibition_preservation(
    package: EngineeringAnswerPackage,
    contract: EngineeringAnswerContract,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-005: Every contract prohibition must survive."""

    contract_prohibitions = {
        prohibition.prohibition_id: prohibition
        for prohibition
        in contract.prohibited_claims
    }

    package_prohibitions = {
        prohibition.prohibition_id: prohibition
        for prohibition
        in package.prohibited_claims
    }

    missing = set(
        contract_prohibitions
    ) - set(
        package_prohibitions
    )

    altered: list[str] = []

    for prohibition_id in (
        set(contract_prohibitions)
        & set(package_prohibitions)
    ):
        if (
            package_prohibitions[prohibition_id]
            != contract_prohibitions[prohibition_id]
        ):
            altered.append(
                prohibition_id
            )

    failed_ids = tuple(
        sorted(
            missing
            | set(altered)
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .PROHIBITION_PRESERVATION
        ),
        passed=not failed_ids,
        success_message=(
            "Every required safety prohibition "
            "is preserved unchanged."
        ),
        failure_message=(
            "Safety prohibitions were removed or altered: "
            + ", ".join(failed_ids)
        ),
        claim_ids=failed_ids,
    )


def validate_test_claim_provenance(
    package: EngineeringAnswerPackage,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-006: Test claims require actual Test citations."""

    citation_index = {
        citation.citation_id: citation
        for citation
        in package.section(
            "test"
        ).excerpts
    } if package.section("test") else {}

    invalid_claims: list[str] = []

    for claim in package.allowed_claims:
        if claim.category != ClaimCategory.TEST:
            continue

        if not claim.citation_ids:
            invalid_claims.append(
                claim.claim_id
            )
            continue

        for citation_id in claim.citation_ids:
            excerpt = citation_index.get(
                citation_id
            )

            if excerpt is None:
                invalid_claims.append(
                    claim.claim_id
                )
                break

            if not excerpt.result_status:
                invalid_claims.append(
                    claim.claim_id
                )
                break

    failed_ids = ordered_unique(
        invalid_claims
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .TEST_CLAIM_PROVENANCE
        ),
        passed=not failed_ids,
        success_message=(
            "Every Test claim uses actual Test evidence "
            "with source-reported result status."
        ),
        failure_message=(
            "Test claims lack valid Test provenance: "
            + ", ".join(failed_ids)
        ),
        claim_ids=failed_ids,
    )


def validate_code_claim_provenance(
    package: EngineeringAnswerPackage,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-007: Code claims require line-based code citations."""

    code_section = package.section(
        "code"
    )

    code_excerpt_index = {
        excerpt.citation_id: excerpt
        for excerpt in (
            code_section.excerpts
            if code_section
            else ()
        )
    }

    invalid_claims: list[str] = []

    for claim in package.allowed_claims:
        if claim.category != ClaimCategory.CODE:
            continue

        if not claim.citation_ids:
            invalid_claims.append(
                claim.claim_id
            )
            continue

        for citation_id in claim.citation_ids:
            excerpt = code_excerpt_index.get(
                citation_id
            )

            if excerpt is None:
                invalid_claims.append(
                    claim.claim_id
                )
                break

            location = (
                excerpt.provenance_location
                .lower()
            )

            if "lines " not in location:
                invalid_claims.append(
                    claim.claim_id
                )
                break

    failed_ids = ordered_unique(
        invalid_claims
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .CODE_CLAIM_PROVENANCE
        ),
        passed=not failed_ids,
        success_message=(
            "Every Code claim uses source-code "
            "provenance with physical line locations."
        ),
        failure_message=(
            "Code claims lack valid line provenance: "
            + ", ".join(failed_ids)
        ),
        claim_ids=failed_ids,
    )


def validate_revision_conflict_protection(
    package: EngineeringAnswerPackage,
    contract: EngineeringAnswerContract,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-008: Unresolved revision conflict must retain protections."""

    conflict_applies = bool(
        contract.assessment_rule
        == AssessmentRule
        .REQUIREMENT_REVISION_CONFLICT
    )

    if not conflict_applies:
        add_check(
            checks,
            issues,
            rule=(
                AdversarialRule
                .REVISION_CONFLICT_PROTECTION
            ),
            passed=True,
            success_message=(
                "No unresolved requirement revision "
                "conflict applies."
            ),
            failure_message="",
        )
        return

    frd_section = package.section(
        "frd"
    )

    revisions = {
        excerpt.source_label
        for excerpt in (
            frd_section.excerpts
            if frd_section
            else ()
        )
    }

    revision_prohibitions = [
        prohibition
        for prohibition
        in package.prohibited_claims
        if (
            prohibition.category
            == ClaimCategory.REVISION
        )
    ]

    text = package.deterministic_text.lower()

    selection_phrases = (
        "rev01 is applicable",
        "rev02 is applicable",
        "applicable revision is rev01",
        "applicable revision is rev02",
    )

    selected_revision = any(
        phrase in text
        for phrase in selection_phrases
    )

    passed = bool(
        len(revisions) >= 2
        and revision_prohibitions
        and not selected_revision
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .REVISION_CONFLICT_PROTECTION
        ),
        passed=passed,
        success_message=(
            "Multiple FRD revisions remain present and "
            "automatic revision selection is prohibited."
        ),
        failure_message=(
            "Revision-conflict protection was weakened "
            "or an applicable revision was selected."
        ),
    )


def validate_traceability_protection(
    package: EngineeringAnswerPackage,
    contract: EngineeringAnswerContract,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-009: Traceability concern must preserve source IDs."""

    traceability_applies = bool(
        contract.assessment_rule
        == AssessmentRule
        .TRACEABILITY_CONCERN
    )

    if not traceability_applies:
        add_check(
            checks,
            issues,
            rule=(
                AdversarialRule
                .TRACEABILITY_PROTECTION
            ),
            passed=True,
            success_message=(
                "No unresolved traceability concern applies."
            ),
            failure_message="",
        )
        return

    contract_requirements = set(
        contract.requirement_ids
    )

    package_requirements = set(
        package.requirement_ids
    )

    traceability_prohibitions = [
        prohibition
        for prohibition
        in package.prohibited_claims
        if (
            prohibition.category
            == ClaimCategory.TRACEABILITY
        )
    ]

    passed = bool(
        package_requirements
        == contract_requirements
        and traceability_prohibitions
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .TRACEABILITY_PROTECTION
        ),
        passed=passed,
        success_message=(
            "Explicit source traceability is preserved "
            "and automatic replacement remains prohibited."
        ),
        failure_message=(
            "Traceability IDs or traceability protections "
            "were altered."
        ),
        engineering_ids=tuple(
            sorted(
                package_requirements
                ^ contract_requirements
            )
        ),
    )


def validate_closure_protection(
    package: EngineeringAnswerPackage,
    contract: EngineeringAnswerContract,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-010: Unestablished closure must not become resolved."""

    expected_status = (
        contract.closure_status.value
    )

    status_preserved = bool(
        package.decision.closure_status
        == expected_status
    )

    closure_prohibitions = [
        prohibition
        for prohibition
        in package.prohibited_claims
        if (
            prohibition.category
            == ClaimCategory.CLOSURE
        )
    ]

    controlled_text = (
        package.deterministic_text
        .lower()
    )

    false_closure_patterns = (
        "observation is resolved",
        "observation is closed",
        "approved for closure",
        "corrective action has been verified",
        "closure is approved",
    )

    false_claim_present = bool(
        contract.closure_status
        != ClosureStatus.CLOSURE_ESTABLISHED
        and any(
            pattern in controlled_text
            for pattern in false_closure_patterns
        )
    )

    passed = bool(
        status_preserved
        and not false_claim_present
        and (
            contract.closure_status
            == ClosureStatus.CLOSURE_ESTABLISHED
            or closure_prohibitions
        )
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .CLOSURE_PROTECTION
        ),
        passed=passed,
        success_message=(
            "Closure status is preserved and false "
            "resolved or closed claims are blocked."
        ),
        failure_message=(
            "Closure status or closure-related "
            "safety protections were altered."
        ),
    )


def validate_human_authority_protection(
    package: EngineeringAnswerPackage,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-011: Package must preserve human authorization limits."""

    prohibition_text = " ".join(
        prohibition.statement.lower()
        + " "
        + prohibition.reason.lower()
        for prohibition
        in package.prohibited_claims
    )

    required_concepts = (
        "engineering",
        "authorization",
    )

    has_required_prohibition = all(
        concept in prohibition_text
        for concept in required_concepts
    )

    false_authority_phrases = (
        "release is approved",
        "certification is approved",
        "engineering approval is granted",
        "closure authorization is granted",
    )

    controlled_text = (
        package.deterministic_text
        .lower()
    )

    false_authority_present = any(
        phrase in controlled_text
        for phrase in false_authority_phrases
    )

    passed = bool(
        has_required_prohibition
        and not false_authority_present
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .HUMAN_AUTHORITY_PROTECTION
        ),
        passed=passed,
        success_message=(
            "Human engineering and release authority "
            "limitations are preserved."
        ),
        failure_message=(
            "Human authority protections are missing "
            "or a false authorization claim is present."
        ),
    )


def detect_untrusted_evidence_instructions(
    package: EngineeringAnswerPackage,
) -> tuple[
    UntrustedEvidenceInstruction,
    ...
]:
    """Detect prompt-like text inside source excerpts."""

    detections: list[
        UntrustedEvidenceInstruction
    ] = []

    for section in package.evidence_sections:
        for excerpt in section.excerpts:
            for pattern in (
                UNTRUSTED_INSTRUCTION_PATTERNS
            ):
                match = pattern.search(
                    excerpt.excerpt
                )

                if match is None:
                    continue

                start = max(
                    0,
                    match.start() - 40,
                )

                end = min(
                    len(excerpt.excerpt),
                    match.end() + 80,
                )

                detections.append(
                    UntrustedEvidenceInstruction(
                        citation_id=(
                            excerpt.citation_id
                        ),
                        evidence_id=(
                            excerpt.evidence_id
                        ),
                        matched_pattern=(
                            pattern.pattern
                        ),
                        excerpt_fragment=(
                            excerpt.excerpt[
                                start:end
                            ]
                        ),
                    )
                )

    return tuple(
        detections
    )


def validate_evidence_instruction_isolation(
    package: EngineeringAnswerPackage,
    detections: tuple[
        UntrustedEvidenceInstruction,
        ...
    ],
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """
    AV-012: Prompt-like evidence text must not become narrator control.

    Detection inside evidence causes a warning, not a failure, provided
    mandatory narrator instructions remain present.
    """

    instruction_text = " ".join(
        package.narrator_instructions
    ).lower()

    mandatory_phrases = (
        "preserve the assessment",
        "use only engineering ids",
        "use only citations",
        "do not infer facts",
    )

    instructions_intact = all(
        phrase in instruction_text
        for phrase in mandatory_phrases
    )

    passed = instructions_intact

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .EVIDENCE_INSTRUCTION_ISOLATION
        ),
        passed=passed,
        success_message=(
            "Prompt-like source text remains isolated as "
            "untrusted evidence and narrator constraints are intact."
        ),
        failure_message=(
            "Mandatory narrator constraints are missing, "
            "so source instructions cannot be safely isolated."
        ),
    )

    if detections:
        issues.append(
            AdversarialIssue(
                rule=(
                    AdversarialRule
                    .EVIDENCE_INSTRUCTION_ISOLATION
                ),
                severity=(
                    AdversarialSeverity.WARNING
                ),
                message=(
                    "Prompt-like instructions were detected "
                    "inside source evidence. They remain "
                    "untrusted quoted evidence."
                ),
                citation_ids=ordered_unique(
                    [
                        detection.citation_id
                        for detection in detections
                    ]
                ),
            )
        )


def validate_package_grounding_state(
    package: EngineeringAnswerPackage,
    checks: list[AdversarialCheck],
    issues: list[AdversarialIssue],
) -> None:
    """AV-013: Package must retain passed grounding and zero errors."""

    passed = bool(
        package.grounding_passed
        and package.grounding_error_count
        == 0
    )

    add_check(
        checks,
        issues,
        rule=(
            AdversarialRule
            .PACKAGE_GROUNDING_STATE
        ),
        passed=passed,
        success_message=(
            "Package retains passed grounding "
            "with zero grounding errors."
        ),
        failure_message=(
            "Package grounding state is invalid."
        ),
    )


def validate_answer_package_adversarially(
    *,
    package: EngineeringAnswerPackage,
    contract: EngineeringAnswerContract,
    strict: bool = False,
) -> AdversarialValidationResult:
    """
    Run complete Phase 13E adversarial validation.

    strict=True blocks the package when any ERROR-level issue exists.
    WARNING-level prompt-like evidence detections do not fail validation.
    """

    checks: list[
        AdversarialCheck
    ] = []

    issues: list[
        AdversarialIssue
    ] = []

    detections = (
        detect_untrusted_evidence_instructions(
            package
        )
    )

    validate_decision_immutability(
        package,
        contract,
        checks,
        issues,
    )

    validate_engineering_id_allowlist(
        package,
        checks,
        issues,
    )

    validate_citation_allowlist(
        package,
        checks,
        issues,
    )

    validate_allowed_claim_integrity(
        package,
        contract,
        checks,
        issues,
    )

    validate_prohibition_preservation(
        package,
        contract,
        checks,
        issues,
    )

    validate_test_claim_provenance(
        package,
        checks,
        issues,
    )

    validate_code_claim_provenance(
        package,
        checks,
        issues,
    )

    validate_revision_conflict_protection(
        package,
        contract,
        checks,
        issues,
    )

    validate_traceability_protection(
        package,
        contract,
        checks,
        issues,
    )

    validate_closure_protection(
        package,
        contract,
        checks,
        issues,
    )

    validate_human_authority_protection(
        package,
        checks,
        issues,
    )

    validate_evidence_instruction_isolation(
        package,
        detections,
        checks,
        issues,
    )

    validate_package_grounding_state(
        package,
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
            == AdversarialSeverity.ERROR
            for issue in issues
        )
    )

    result = AdversarialValidationResult(
        passed=passed,
        checks=tuple(checks),
        issues=tuple(issues),
        untrusted_evidence_instructions=(
            detections
        ),
    )

    if strict:
        result.require_passed()

    return result
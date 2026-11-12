"""
Safe engineering answer contract for REQTRACE AI.

Phase 13C converts:

- the Phase 12 deterministic decision
- Phase 13A citations
- Phase 13B grounding validation

into one machine-readable contract.

The future LLM narrator must treat the contract as authoritative.

The LLM may:

- explain the deterministic result
- summarize validated evidence
- describe explicit blockers
- reference approved citations

The LLM may not:

- change the assessment
- change confidence
- change closure status
- invent engineering IDs
- invent test results
- invent page or line provenance
- resolve requirement revision conflicts
- rewrite traceability
- authorize engineering closure
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

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
from src.confidence_engine import (
    DecisionConfidence,
)
from src.decision_engine import (
    EngineeringDecisionResult,
)
from src.grounding import (
    GroundingValidationResult,
)


class AnswerContractError(
    RuntimeError
):
    """Raised when a safe answer contract cannot be created."""


class ClaimCategory(
    str,
    Enum,
):
    """Supported claim categories."""

    ASSESSMENT = "ASSESSMENT"
    CONFIDENCE = "CONFIDENCE"
    REQUIREMENT = "REQUIREMENT"
    OBSERVATION = "OBSERVATION"
    CODE = "CODE"
    TEST = "TEST"
    REVISION = "REVISION"
    TRACEABILITY = "TRACEABILITY"
    BLOCKER = "BLOCKER"
    CLOSURE = "CLOSURE"
    LIMITATION = "LIMITATION"


ClaimCategory.__test__ = False


@dataclass(frozen=True)
class AllowedClaim:
    """
    One deterministic claim permitted in grounded narration.

    citation_ids contains the citations that may support the claim.
    """

    claim_id: str
    category: ClaimCategory
    statement: str
    citation_ids: tuple[str, ...] = ()
    deterministic_rule: str | None = None

    def __post_init__(
        self,
    ) -> None:
        if not self.claim_id.strip():
            raise AnswerContractError(
                "Allowed claim requires claim_id."
            )

        if not self.statement.strip():
            raise AnswerContractError(
                "Allowed claim requires statement."
            )


@dataclass(frozen=True)
class ProhibitedClaim:
    """One claim the narrator must not make."""

    prohibition_id: str
    category: ClaimCategory
    statement: str
    reason: str

    def __post_init__(
        self,
    ) -> None:
        if not self.prohibition_id.strip():
            raise AnswerContractError(
                "Prohibited claim requires prohibition_id."
            )

        if not self.statement.strip():
            raise AnswerContractError(
                "Prohibited claim requires statement."
            )

        if not self.reason.strip():
            raise AnswerContractError(
                "Prohibited claim requires reason."
            )


@dataclass(frozen=True)
class ContractEvidenceSummary:
    """Citation-backed evidence summary by engineering type."""

    requirement_citations: tuple[str, ...]
    observation_citations: tuple[str, ...]
    code_citations: tuple[str, ...]
    test_citations: tuple[str, ...]
    review_citations: tuple[str, ...]

    @property
    def all_citation_ids(
        self,
    ) -> tuple[str, ...]:
        """Return all evidence-summary citation IDs."""

        output: list[str] = []
        seen: set[str] = set()

        groups = (
            self.requirement_citations,
            self.observation_citations,
            self.code_citations,
            self.test_citations,
            self.review_citations,
        )

        for group in groups:
            for citation_id in group:
                if citation_id in seen:
                    continue

                seen.add(citation_id)
                output.append(citation_id)

        return tuple(output)


@dataclass(frozen=True)
class EngineeringAnswerContract:
    """
    Complete safe answer contract.

    This contract is the authoritative input to Phase 14 narration.
    """

    contract_version: str

    assessment: ObservationAssessment
    assessment_rule: AssessmentRule
    assessment_reason: str

    confidence: DecisionConfidence
    confidence_reason: str

    closure_status: ClosureStatus
    closure_reason: str
    closure_was_assessed: bool

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    supporting_items: tuple[str, ...]
    blocking_items: tuple[str, ...]

    evidence_summary: ContractEvidenceSummary
    citations: CitationCollection

    allowed_claims: tuple[
        AllowedClaim,
        ...
    ]

    prohibited_claims: tuple[
        ProhibitedClaim,
        ...
    ]

    grounding_passed: bool
    grounding_error_count: int
    grounding_warning_count: int

    narrator_instructions: tuple[str, ...]

    def __post_init__(
        self,
    ) -> None:
        """Validate the answer contract."""

        if not self.contract_version.strip():
            raise AnswerContractError(
                "contract_version must not be empty."
            )

        if not self.assessment_reason.strip():
            raise AnswerContractError(
                "assessment_reason must not be empty."
            )

        if not self.confidence_reason.strip():
            raise AnswerContractError(
                "confidence_reason must not be empty."
            )

        if not self.closure_reason.strip():
            raise AnswerContractError(
                "closure_reason must not be empty."
            )

        if not self.grounding_passed:
            raise AnswerContractError(
                "An answer contract cannot be created "
                "from failed grounding validation."
            )

        if self.grounding_error_count != 0:
            raise AnswerContractError(
                "A grounded answer contract must contain "
                "zero grounding errors."
            )

        citation_allowlist = set(
            self.citations.citation_ids
        )

        for claim in self.allowed_claims:
            unsupported_citations = (
                set(claim.citation_ids)
                - citation_allowlist
            )

            if unsupported_citations:
                raise AnswerContractError(
                    "Allowed claim references citations "
                    "outside the citation allowlist: "
                    + ", ".join(
                        sorted(
                            unsupported_citations
                        )
                    )
                )

    @property
    def citation_allowlist(
        self,
    ) -> tuple[str, ...]:
        """Return citation IDs permitted in narration."""

        return self.citations.citation_ids

    @property
    def engineering_id_allowlist(
        self,
    ) -> tuple[str, ...]:
        """Return engineering IDs permitted in narration."""

        values: list[str] = []

        values.extend(
            self.requirement_ids
        )

        values.extend(
            self.observation_ids
        )

        values.extend(
            self.test_case_ids
        )

        return ordered_unique(
            values
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert contract into JSON-safe data."""

        return {
            "contract_version": (
                self.contract_version
            ),
            "decision": {
                "assessment": (
                    self.assessment.value
                ),
                "assessment_rule": (
                    self.assessment_rule.value
                ),
                "assessment_reason": (
                    self.assessment_reason
                ),
                "confidence": (
                    self.confidence.value
                ),
                "confidence_reason": (
                    self.confidence_reason
                ),
                "closure_status": (
                    self.closure_status.value
                ),
                "closure_reason": (
                    self.closure_reason
                ),
                "closure_was_assessed": (
                    self.closure_was_assessed
                ),
            },
            "engineering_ids": {
                "requirement_ids": list(
                    self.requirement_ids
                ),
                "observation_ids": list(
                    self.observation_ids
                ),
                "test_case_ids": list(
                    self.test_case_ids
                ),
                "allowlist": list(
                    self.engineering_id_allowlist
                ),
            },
            "supporting_items": list(
                self.supporting_items
            ),
            "blocking_items": list(
                self.blocking_items
            ),
            "evidence_summary": {
                "requirement_citations": list(
                    self.evidence_summary
                    .requirement_citations
                ),
                "observation_citations": list(
                    self.evidence_summary
                    .observation_citations
                ),
                "code_citations": list(
                    self.evidence_summary
                    .code_citations
                ),
                "test_citations": list(
                    self.evidence_summary
                    .test_citations
                ),
                "review_citations": list(
                    self.evidence_summary
                    .review_citations
                ),
            },
            "citations": (
                self.citations.to_dict()
            ),
            "citation_allowlist": list(
                self.citation_allowlist
            ),
            "allowed_claims": [
                {
                    "claim_id": claim.claim_id,
                    "category": (
                        claim.category.value
                    ),
                    "statement": (
                        claim.statement
                    ),
                    "citation_ids": list(
                        claim.citation_ids
                    ),
                    "deterministic_rule": (
                        claim.deterministic_rule
                    ),
                }
                for claim in self.allowed_claims
            ],
            "prohibited_claims": [
                {
                    "prohibition_id": (
                        claim.prohibition_id
                    ),
                    "category": (
                        claim.category.value
                    ),
                    "statement": (
                        claim.statement
                    ),
                    "reason": (
                        claim.reason
                    ),
                }
                for claim
                in self.prohibited_claims
            ],
            "grounding": {
                "passed": (
                    self.grounding_passed
                ),
                "error_count": (
                    self.grounding_error_count
                ),
                "warning_count": (
                    self.grounding_warning_count
                ),
            },
            "narrator_instructions": list(
                self.narrator_instructions
            ),
        }


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered, duplicate-free non-empty strings."""

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


def citation_ids_for_type(
    citations: CitationCollection,
    document_type: str,
) -> tuple[str, ...]:
    """Return citation IDs for one document type."""

    return tuple(
        citation.citation_id
        for citation
        in citations.by_document_type(
            document_type
        )
    )


def citation_ids_for_requirement(
    citations: CitationCollection,
    requirement_id: str,
) -> tuple[str, ...]:
    """Return citations carrying one requirement ID."""

    return tuple(
        citation.citation_id
        for citation in citations.citations
        if (
            citation.requirement_id
            == requirement_id
        )
    )


def citation_ids_for_observation(
    citations: CitationCollection,
    observation_id: str,
) -> tuple[str, ...]:
    """Return citations carrying one observation ID."""

    observation_citations = (
        citations.by_document_type(
            "observation"
        )
    )

    return tuple(
        citation.citation_id
        for citation
        in observation_citations
        if (
            citation.observation_id
            == observation_id
        )
    )


def citation_ids_for_test_case(
    citations: CitationCollection,
    test_case_id: str,
) -> tuple[str, ...]:
    """Return actual Test citations for one test case."""

    test_citations = (
        citations.by_document_type(
            "test"
        )
    )

    return tuple(
        citation.citation_id
        for citation in test_citations
        if (
            citation.test_case_id
            == test_case_id
        )
    )


def build_evidence_summary(
    citations: CitationCollection,
) -> ContractEvidenceSummary:
    """Group validated citations by engineering evidence type."""

    return ContractEvidenceSummary(
        requirement_citations=(
            citation_ids_for_type(
                citations,
                "frd",
            )
        ),
        observation_citations=(
            citation_ids_for_type(
                citations,
                "observation",
            )
        ),
        code_citations=(
            citation_ids_for_type(
                citations,
                "code",
            )
        ),
        test_citations=(
            citation_ids_for_type(
                citations,
                "test",
            )
        ),
        review_citations=(
            citation_ids_for_type(
                citations,
                "review",
            )
        ),
    )


def build_assessment_claim(
    decision: EngineeringDecisionResult,
) -> AllowedClaim:
    """Build the authoritative assessment claim."""

    assessment = (
        decision.assessment
        .assessment
        .value
    )

    observation_ids = (
        decision.assessment
        .observation_ids
    )

    if observation_ids:
        subject = ", ".join(
            observation_ids
        )
    else:
        subject = (
            "the retrieved engineering issue"
        )

    statement = (
        f"The deterministic assessment for "
        f"{subject} is {assessment}."
    )

    return AllowedClaim(
        claim_id="CLM-001",
        category=ClaimCategory.ASSESSMENT,
        statement=statement,
        citation_ids=(),
        deterministic_rule=(
            decision.assessment
            .applied_rule
            .value
        ),
    )


def build_confidence_claim(
    decision: EngineeringDecisionResult,
) -> AllowedClaim:
    """Build the authoritative confidence claim."""

    return AllowedClaim(
        claim_id="CLM-002",
        category=ClaimCategory.CONFIDENCE,
        statement=(
            "Confidence in the deterministic "
            "decision basis is "
            f"{decision.confidence.confidence.value}."
        ),
        citation_ids=(),
        deterministic_rule=(
            decision.confidence
            .applied_rule
            .value
        ),
    )


def build_engineering_id_claims(
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    starting_index: int,
) -> list:
    """Build citation-backed engineering-ID claims."""

    claims: list[AllowedClaim] = []
    claim_index = starting_index

    for requirement_id in (
        decision.assessment
        .requirement_ids
    ):
        citation_ids = (
            citation_ids_for_requirement(
                citations,
                requirement_id,
            )
        )

        claims.append(
            AllowedClaim(
                claim_id=(
                    f"CLM-{claim_index:03d}"
                ),
                category=(
                    ClaimCategory.REQUIREMENT
                ),
                statement=(
                    f"{requirement_id} is part of "
                    "the validated evidence context."
                ),
                citation_ids=(
                    citation_ids
                ),
            )
        )

        claim_index += 1

    for observation_id in (
        decision.assessment
        .observation_ids
    ):
        citation_ids = (
            citation_ids_for_observation(
                citations,
                observation_id,
            )
        )

        claims.append(
            AllowedClaim(
                claim_id=(
                    f"CLM-{claim_index:03d}"
                ),
                category=(
                    ClaimCategory.OBSERVATION
                ),
                statement=(
                    f"{observation_id} is the "
                    "grounded observation under analysis."
                ),
                citation_ids=(
                    citation_ids
                ),
            )
        )

        claim_index += 1

    for test_case_id in (
        decision.assessment
        .test_case_ids
    ):
        citation_ids = (
            citation_ids_for_test_case(
                citations,
                test_case_id,
            )
        )

        test_citations = tuple(
            citation
            for citation in citations.citations
            if (
                citation.citation_id
                in citation_ids
            )
        )

        statuses = ordered_unique(
            [
                citation.result_status
                for citation
                in test_citations
                if citation.result_status
            ]
        )

        if statuses:
            status_text = ", ".join(
                statuses
            )

            statement = (
                f"{test_case_id} has the "
                f"source-reported result "
                f"{status_text}."
            )
        else:
            statement = (
                f"{test_case_id} is referenced "
                "by the validated evidence context."
            )

        claims.append(
            AllowedClaim(
                claim_id=(
                    f"CLM-{claim_index:03d}"
                ),
                category=ClaimCategory.TEST,
                statement=statement,
                citation_ids=citation_ids,
            )
        )

        claim_index += 1

    return claims


def build_code_claims(
    citations: CitationCollection,
    starting_index: int,
) -> list:
    """Build provenance-safe source-code claims."""

    claims: list[AllowedClaim] = []
    claim_index = starting_index

    for citation in (
        citations.by_document_type(
            "code"
        )
    ):
        if citation.function_name:
            subject = (
                f"{citation.function_name}()"
            )
        else:
            subject = (
                citation.file_name
                or "the cited source code"
            )

        statement = (
            f"{subject} is cited from "
            f"{citation.provenance_location}."
        )

        claims.append(
            AllowedClaim(
                claim_id=(
                    f"CLM-{claim_index:03d}"
                ),
                category=ClaimCategory.CODE,
                statement=statement,
                citation_ids=(
                    citation.citation_id,
                ),
            )
        )

        claim_index += 1

    return claims


def build_blocker_claims(
    decision: EngineeringDecisionResult,
    starting_index: int,
) -> list:
    """Build deterministic blocker claims."""

    claims: list[AllowedClaim] = []
    claim_index = starting_index

    for blocker in (
        decision.blocking_items
    ):
        claims.append(
            AllowedClaim(
                claim_id=(
                    f"CLM-{claim_index:03d}"
                ),
                category=ClaimCategory.BLOCKER,
                statement=blocker,
                citation_ids=(),
                deterministic_rule=(
                    decision.assessment
                    .applied_rule
                    .value
                ),
            )
        )

        claim_index += 1

    return claims


def build_closure_claim(
    decision: EngineeringDecisionResult,
    claim_index: int,
) -> AllowedClaim:
    """Build the authoritative closure-status claim."""

    return AllowedClaim(
        claim_id=(
            f"CLM-{claim_index:03d}"
        ),
        category=ClaimCategory.CLOSURE,
        statement=(
            "The deterministic closure status is "
            f"{decision.closure_status.value}."
        ),
        citation_ids=(),
        deterministic_rule=(
            decision.closure_rule.value
        ),
    )


def build_prohibited_claims(
    decision: EngineeringDecisionResult,
) -> tuple[ProhibitedClaim, ...]:
    """Build scenario-specific prohibited claims."""

    prohibited: list[
        ProhibitedClaim
    ] = []

    prohibited.append(
        ProhibitedClaim(
            prohibition_id="PRH-001",
            category=ClaimCategory.ASSESSMENT,
            statement=(
                "Do not change or reinterpret the "
                "deterministic assessment."
            ),
            reason=(
                "Assessment authority belongs to "
                "the deterministic rule engine."
            ),
        )
    )

    prohibited.append(
        ProhibitedClaim(
            prohibition_id="PRH-002",
            category=ClaimCategory.CONFIDENCE,
            statement=(
                "Do not express confidence as a "
                "probability or percentage."
            ),
            reason=(
                "The confidence value is a deterministic "
                "classification, not a calibrated probability."
            ),
        )
    )

    prohibited.append(
        ProhibitedClaim(
            prohibition_id="PRH-003",
            category=ClaimCategory.REQUIREMENT,
            statement=(
                "Do not introduce requirement, observation, "
                "or test IDs outside the engineering-ID allowlist."
            ),
            reason=(
                "Unsupported engineering IDs are not grounded."
            ),
        )
    )

    prohibited.append(
        ProhibitedClaim(
            prohibition_id="PRH-004",
            category=ClaimCategory.CODE,
            statement=(
                "Do not invent source files, functions, "
                "line ranges, or implementation behavior."
            ),
            reason=(
                "Code claims must remain within validated "
                "source-code citations."
            ),
        )
    )

    prohibited.append(
        ProhibitedClaim(
            prohibition_id="PRH-005",
            category=ClaimCategory.TEST,
            statement=(
                "Do not report a PASS, FAIL, or NOT EXECUTED "
                "result without an actual Test citation."
            ),
            reason=(
                "A test-case ID reference is not equivalent "
                "to actual test evidence."
            ),
        )
    )

    if (
        decision.assessment
        .applied_rule
        == AssessmentRule
        .REQUIREMENT_REVISION_CONFLICT
    ):
        prohibited.append(
            ProhibitedClaim(
                prohibition_id="PRH-006",
                category=ClaimCategory.REVISION,
                statement=(
                    "Do not select an applicable "
                    "requirement revision."
                ),
                reason=(
                    "The deterministic decision identifies "
                    "an unresolved revision conflict."
                ),
            )
        )

    if (
        decision.assessment
        .applied_rule
        == AssessmentRule
        .TRACEABILITY_CONCERN
    ):
        prohibited.append(
            ProhibitedClaim(
                prohibition_id="PRH-007",
                category=(
                    ClaimCategory.TRACEABILITY
                ),
                statement=(
                    "Do not replace the explicit source "
                    "requirement reference with a semantically "
                    "similar requirement."
                ),
                reason=(
                    "Traceability must preserve the "
                    "source-recorded requirement reference."
                ),
            )
        )

    if (
        decision.closure_status
        != ClosureStatus.CLOSURE_ESTABLISHED
    ):
        prohibited.append(
            ProhibitedClaim(
                prohibition_id="PRH-008",
                category=ClaimCategory.CLOSURE,
                statement=(
                    "Do not state that the observation is "
                    "resolved, closed, approved for closure, "
                    "or verified as corrected."
                ),
                reason=(
                    "Closure has not been deterministically "
                    "established."
                ),
            )
        )

    prohibited.append(
        ProhibitedClaim(
            prohibition_id="PRH-009",
            category=ClaimCategory.CLOSURE,
            statement=(
                "Do not simulate final engineering, quality, "
                "safety, certification, or release authorization."
            ),
            reason=(
                "Final authorization remains with the "
                "responsible human authority."
            ),
        )
    )

    return tuple(
        prohibited
    )


def narrator_instructions() -> tuple[str, ...]:
    """Return mandatory Phase 14 narrator instructions."""

    return (
        (
            "Preserve the assessment, confidence, "
            "and closure status exactly."
        ),
        (
            "Use only engineering IDs in the "
            "engineering-ID allowlist."
        ),
        (
            "Use only citations in the citation allowlist."
        ),
        (
            "Do not infer facts from missing evidence."
        ),
        (
            "State blockers explicitly when the decision "
            "is inconclusive or closure is not established."
        ),
        (
            "Do not interpret deterministic confidence "
            "as a probability."
        ),
        (
            "Do not claim human approval, certification, "
            "release authorization, or closure authorization."
        ),
    )


def build_answer_contract(
    *,
    decision: EngineeringDecisionResult,
    citations: CitationCollection,
    grounding: GroundingValidationResult,
) -> EngineeringAnswerContract:
    """
    Build the complete Phase 13C answer contract.

    Grounding must already have passed.
    """

    if not grounding.passed:
        raise AnswerContractError(
            "Cannot build answer contract because "
            "grounding validation failed."
        )

    if grounding.error_count != 0:
        raise AnswerContractError(
            "Cannot build answer contract with "
            "grounding errors."
        )

    allowed_claims: list[
        AllowedClaim
    ] = []

    allowed_claims.append(
        build_assessment_claim(
            decision
        )
    )

    allowed_claims.append(
        build_confidence_claim(
            decision
        )
    )

    next_index = 3

    engineering_claims = (
        build_engineering_id_claims(
            decision,
            citations,
            next_index,
        )
    )

    allowed_claims.extend(
        engineering_claims
    )

    next_index += len(
        engineering_claims
    )

    code_claims = build_code_claims(
        citations,
        next_index,
    )

    allowed_claims.extend(
        code_claims
    )

    next_index += len(
        code_claims
    )

    blocker_claims = (
        build_blocker_claims(
            decision,
            next_index,
        )
    )

    allowed_claims.extend(
        blocker_claims
    )

    next_index += len(
        blocker_claims
    )

    allowed_claims.append(
        build_closure_claim(
            decision,
            next_index,
        )
    )

    return EngineeringAnswerContract(
        contract_version="1.0",
        assessment=(
            decision.assessment
            .assessment
        ),
        assessment_rule=(
            decision.assessment
            .applied_rule
        ),
        assessment_reason=(
            decision.assessment.reason
        ),
        confidence=(
            decision.confidence
            .confidence
        ),
        confidence_reason=(
            decision.confidence.reason
        ),
        closure_status=(
            decision.closure_status
        ),
        closure_reason=(
            decision.closure_reason
        ),
        closure_was_assessed=(
            decision.closure_was_assessed
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
        supporting_items=(
            decision.supporting_items
        ),
        blocking_items=(
            decision.blocking_items
        ),
        evidence_summary=(
            build_evidence_summary(
                citations
            )
        ),
        citations=citations,
        allowed_claims=tuple(
            allowed_claims
        ),
        prohibited_claims=(
            build_prohibited_claims(
                decision
            )
        ),
        grounding_passed=(
            grounding.passed
        ),
        grounding_error_count=(
            grounding.error_count
        ),
        grounding_warning_count=(
            grounding.warning_count
        ),
        narrator_instructions=(
            narrator_instructions()
        ),
    )
"""
Deterministic engineering answer-package builder for REQTRACE AI.

Phase 13D converts a validated Phase 13C answer contract and its final
evidence context into a complete, render-ready engineering answer package.

The package includes:

- deterministic assessment
- deterministic confidence
- closure status
- evidence sections
- validated citation references
- source excerpts
- supporting and blocking items
- allowed and prohibited claims
- narrator instructions

This module does not:

- invoke an LLM
- modify the deterministic decision
- create new engineering claims
- invent citations
- invent evidence excerpts
- rewrite traceability
- select an applicable requirement revision
- authorize observation closure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.answer_contract import (
    AllowedClaim,
    EngineeringAnswerContract,
    ProhibitedClaim,
)
from src.citations import (
    EvidenceCitation,
)
from src.context_builder import (
    FinalEvidenceContext,
)
from src.retrieval import (
    RetrievedEvidence,
)


class AnswerPackageError(
    RuntimeError
):
    """Raised when a deterministic answer package cannot be built."""


@dataclass(frozen=True)
class EvidenceExcerpt:
    """One citation-bound evidence excerpt."""

    citation_id: str
    evidence_id: str
    document_type: str

    source_label: str
    provenance_location: str
    excerpt: str

    requirement_id: str | None = None
    observation_id: str | None = None
    test_case_id: str | None = None
    result_status: str | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate one evidence excerpt."""

        if not self.citation_id.strip():
            raise AnswerPackageError(
                "Evidence excerpt requires citation_id."
            )

        if not self.evidence_id.strip():
            raise AnswerPackageError(
                "Evidence excerpt requires evidence_id."
            )

        if not self.source_label.strip():
            raise AnswerPackageError(
                "Evidence excerpt requires source_label."
            )

        if not self.provenance_location.strip():
            raise AnswerPackageError(
                "Evidence excerpt requires provenance_location."
            )

        if not self.excerpt.strip():
            raise AnswerPackageError(
                "Evidence excerpt must not be empty."
            )

    @property
    def inline_citation(
        self,
    ) -> str:
        """Return the approved inline citation marker."""

        return f"[{self.citation_id}]"

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert excerpt into JSON-safe data."""

        return {
            "citation_id": self.citation_id,
            "evidence_id": self.evidence_id,
            "document_type": self.document_type,
            "source_label": self.source_label,
            "provenance_location": (
                self.provenance_location
            ),
            "excerpt": self.excerpt,
            "requirement_id": self.requirement_id,
            "observation_id": self.observation_id,
            "test_case_id": self.test_case_id,
            "result_status": self.result_status,
            "inline_citation": self.inline_citation,
        }


@dataclass(frozen=True)
class EvidenceSection:
    """One grouped engineering-evidence section."""

    section_id: str
    heading: str
    document_type: str

    excerpts: tuple[
        EvidenceExcerpt,
        ...
    ]

    @property
    def citation_ids(
        self,
    ) -> tuple[str, ...]:
        """Return section citation IDs."""

        return tuple(
            excerpt.citation_id
            for excerpt in self.excerpts
        )

    @property
    def is_empty(
        self,
    ) -> bool:
        """Return True when the section has no evidence."""

        return not self.excerpts

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert section into JSON-safe data."""

        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "document_type": self.document_type,
            "citation_ids": list(
                self.citation_ids
            ),
            "excerpts": [
                excerpt.to_dict()
                for excerpt in self.excerpts
            ],
        }


@dataclass(frozen=True)
class DecisionSummary:
    """Deterministic decision summary."""

    assessment: str
    assessment_rule: str
    assessment_reason: str

    confidence: str
    confidence_reason: str

    closure_status: str
    closure_reason: str
    closure_was_assessed: bool

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert decision summary into JSON-safe data."""

        return {
            "assessment": self.assessment,
            "assessment_rule": (
                self.assessment_rule
            ),
            "assessment_reason": (
                self.assessment_reason
            ),
            "confidence": self.confidence,
            "confidence_reason": (
                self.confidence_reason
            ),
            "closure_status": (
                self.closure_status
            ),
            "closure_reason": (
                self.closure_reason
            ),
            "closure_was_assessed": (
                self.closure_was_assessed
            ),
        }


@dataclass(frozen=True)
class EngineeringAnswerPackage:
    """
    Complete deterministic answer package.

    This is the final validated non-LLM output and the controlled input
    for Phase 14 narration.
    """

    package_version: str

    decision: DecisionSummary

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    supporting_items: tuple[str, ...]
    blocking_items: tuple[str, ...]

    evidence_sections: tuple[
        EvidenceSection,
        ...
    ]

    allowed_claims: tuple[
        AllowedClaim,
        ...
    ]

    prohibited_claims: tuple[
        ProhibitedClaim,
        ...
    ]

    citation_allowlist: tuple[str, ...]
    engineering_id_allowlist: tuple[str, ...]

    narrator_instructions: tuple[str, ...]

    grounding_passed: bool
    grounding_error_count: int
    grounding_warning_count: int

    deterministic_text: str

    def __post_init__(
        self,
    ) -> None:
        """Validate the completed answer package."""

        if not self.package_version.strip():
            raise AnswerPackageError(
                "package_version must not be empty."
            )

        if not self.grounding_passed:
            raise AnswerPackageError(
                "Answer package requires passed grounding."
            )

        if self.grounding_error_count != 0:
            raise AnswerPackageError(
                "Answer package requires zero grounding errors."
            )

        if not self.deterministic_text.strip():
            raise AnswerPackageError(
                "deterministic_text must not be empty."
            )

        package_citations = set(
            self.all_evidence_citation_ids
        )

        allowlist = set(
            self.citation_allowlist
        )

        unsupported = (
            package_citations
            - allowlist
        )

        if unsupported:
            raise AnswerPackageError(
                "Answer package contains citations outside "
                "the citation allowlist: "
                + ", ".join(
                    sorted(unsupported)
                )
            )

    @property
    def all_evidence_citation_ids(
        self,
    ) -> tuple[str, ...]:
        """Return ordered unique citation IDs used in evidence sections."""

        values: list[str] = []

        for section in self.evidence_sections:
            values.extend(
                section.citation_ids
            )

        return ordered_unique(
            values
        )

    @property
    def evidence_record_count(
        self,
    ) -> int:
        """Return total evidence excerpts."""

        return sum(
            len(section.excerpts)
            for section in self.evidence_sections
        )

    def section(
        self,
        document_type: str,
    ) -> EvidenceSection | None:
        """Return one evidence section by document type."""

        normalized_type = (
            document_type
            .strip()
            .lower()
        )

        for section in self.evidence_sections:
            if (
                section.document_type
                == normalized_type
            ):
                return section

        return None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert answer package into JSON-safe data."""

        return {
            "package_version": (
                self.package_version
            ),
            "decision": (
                self.decision.to_dict()
            ),
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
            "evidence_sections": [
                section.to_dict()
                for section
                in self.evidence_sections
            ],
            "evidence_record_count": (
                self.evidence_record_count
            ),
            "allowed_claims": [
                allowed_claim_to_dict(
                    claim
                )
                for claim
                in self.allowed_claims
            ],
            "prohibited_claims": [
                prohibited_claim_to_dict(
                    claim
                )
                for claim
                in self.prohibited_claims
            ],
            "citation_allowlist": list(
                self.citation_allowlist
            ),
            "engineering_id_allowlist": list(
                self.engineering_id_allowlist
            ),
            "narrator_instructions": list(
                self.narrator_instructions
            ),
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
            "deterministic_text": (
                self.deterministic_text
            ),
        }


SECTION_CONFIGURATION = (
    (
        "SEC-REQ",
        "Requirement Evidence",
        "frd",
    ),
    (
        "SEC-OBS",
        "Observation Evidence",
        "observation",
    ),
    (
        "SEC-CODE",
        "Source-Code Evidence",
        "code",
    ),
    (
        "SEC-TEST",
        "Test Evidence",
        "test",
    ),
    (
        "SEC-REVIEW",
        "Review Evidence",
        "review",
    ),
)


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


def normalize_excerpt(
    content: str,
    maximum_characters: int = 500,
) -> str:
    """
    Create a deterministic source excerpt.

    The excerpt is built only from retrieved evidence content.
    No paraphrasing or generated language is introduced.
    """

    if maximum_characters < 1:
        raise ValueError(
            "maximum_characters must be positive."
        )

    normalized = " ".join(
        content.split()
    )

    if not normalized:
        raise AnswerPackageError(
            "Evidence content is empty."
        )

    if len(normalized) <= maximum_characters:
        return normalized

    truncated = normalized[
        :maximum_characters
    ]

    last_space = truncated.rfind(
        " "
    )

    if last_space > 0:
        truncated = truncated[
            :last_space
        ]

    return truncated.rstrip() + " ..."


def evidence_by_id(
    context: FinalEvidenceContext,
) -> dict[str, RetrievedEvidence]:
    """Index final-context evidence by stable evidence ID."""

    output: dict[
        str,
        RetrievedEvidence,
    ] = {}

    for evidence in context.evidence:
        if evidence.evidence_id in output:
            raise AnswerPackageError(
                "Duplicate evidence ID in final context: "
                f"{evidence.evidence_id}"
            )

        output[
            evidence.evidence_id
        ] = evidence

    return output


def citation_by_evidence_id(
    contract: EngineeringAnswerContract,
) -> dict[str, EvidenceCitation]:
    """Index contract citations by evidence ID."""

    output: dict[
        str,
        EvidenceCitation,
    ] = {}

    for citation in contract.citations.citations:
        if citation.evidence_id in output:
            raise AnswerPackageError(
                "Duplicate citation evidence ID: "
                f"{citation.evidence_id}"
            )

        output[
            citation.evidence_id
        ] = citation

    return output


def build_evidence_excerpt(
    *,
    evidence: RetrievedEvidence,
    citation: EvidenceCitation,
    maximum_characters: int,
) -> EvidenceExcerpt:
    """Build one citation-bound source excerpt."""

    if (
        evidence.evidence_id
        != citation.evidence_id
    ):
        raise AnswerPackageError(
            "Evidence and citation IDs do not match."
        )

    return EvidenceExcerpt(
        citation_id=(
            citation.citation_id
        ),
        evidence_id=(
            evidence.evidence_id
        ),
        document_type=(
            evidence.document_type
        ),
        source_label=(
            citation.display_label
        ),
        provenance_location=(
            citation.provenance_location
        ),
        excerpt=normalize_excerpt(
            evidence.content,
            maximum_characters=(
                maximum_characters
            ),
        ),
        requirement_id=(
            citation.requirement_id
        ),
        observation_id=(
            citation.observation_id
        ),
        test_case_id=(
            citation.test_case_id
        ),
        result_status=(
            citation.result_status
        ),
    )


def build_evidence_sections(
    *,
    context: FinalEvidenceContext,
    contract: EngineeringAnswerContract,
    maximum_excerpt_characters: int,
) -> tuple[EvidenceSection, ...]:
    """Build deterministic evidence sections."""

    context_index = evidence_by_id(
        context
    )

    citation_index = (
        citation_by_evidence_id(
            contract
        )
    )

    context_ids = set(
        context_index
    )

    citation_ids = set(
        citation_index
    )

    if context_ids != citation_ids:
        missing_citations = (
            context_ids
            - citation_ids
        )

        extra_citations = (
            citation_ids
            - context_ids
        )

        parts: list[str] = []

        if missing_citations:
            parts.append(
                "Missing citations: "
                + ", ".join(
                    sorted(
                        missing_citations
                    )
                )
            )

        if extra_citations:
            parts.append(
                "Extra citations: "
                + ", ".join(
                    sorted(
                        extra_citations
                    )
                )
            )

        raise AnswerPackageError(
            "Context/citation mismatch. "
            + "; ".join(parts)
        )

    sections: list[
        EvidenceSection
    ] = []

    for (
        section_id,
        heading,
        document_type,
    ) in SECTION_CONFIGURATION:

        excerpts: list[
            EvidenceExcerpt
        ] = []

        for evidence in context.evidence:
            if (
                evidence.document_type
                != document_type
            ):
                continue

            citation = citation_index.get(
                evidence.evidence_id
            )

            if citation is None:
                raise AnswerPackageError(
                    "Citation missing for evidence: "
                    f"{evidence.evidence_id}"
                )

            excerpts.append(
                build_evidence_excerpt(
                    evidence=evidence,
                    citation=citation,
                    maximum_characters=(
                        maximum_excerpt_characters
                    ),
                )
            )

        sections.append(
            EvidenceSection(
                section_id=section_id,
                heading=heading,
                document_type=document_type,
                excerpts=tuple(excerpts),
            )
        )

    return tuple(sections)


def make_decision_summary(
    contract: EngineeringAnswerContract,
) -> DecisionSummary:
    """Build the immutable deterministic decision summary."""

    return DecisionSummary(
        assessment=(
            contract.assessment.value
        ),
        assessment_rule=(
            contract.assessment_rule.value
        ),
        assessment_reason=(
            contract.assessment_reason
        ),
        confidence=(
            contract.confidence.value
        ),
        confidence_reason=(
            contract.confidence_reason
        ),
        closure_status=(
            contract.closure_status.value
        ),
        closure_reason=(
            contract.closure_reason
        ),
        closure_was_assessed=(
            contract.closure_was_assessed
        ),
    )


def allowed_claim_to_dict(
    claim: AllowedClaim,
) -> dict[str, Any]:
    """Convert one allowed claim into JSON-safe data."""

    return {
        "claim_id": claim.claim_id,
        "category": claim.category.value,
        "statement": claim.statement,
        "citation_ids": list(
            claim.citation_ids
        ),
        "deterministic_rule": (
            claim.deterministic_rule
        ),
    }


def prohibited_claim_to_dict(
    claim: ProhibitedClaim,
) -> dict[str, Any]:
    """Convert one prohibited claim into JSON-safe data."""

    return {
        "prohibition_id": (
            claim.prohibition_id
        ),
        "category": claim.category.value,
        "statement": claim.statement,
        "reason": claim.reason,
    }


def format_citation_suffix(
    citation_ids: tuple[str, ...],
) -> str:
    """Format deterministic citation markers."""

    if not citation_ids:
        return ""

    markers = " ".join(
        f"[{citation_id}]"
        for citation_id in citation_ids
    )

    return " " + markers


def render_deterministic_text(
    *,
    decision: DecisionSummary,
    requirement_ids: tuple[str, ...],
    observation_ids: tuple[str, ...],
    test_case_ids: tuple[str, ...],
    supporting_items: tuple[str, ...],
    blocking_items: tuple[str, ...],
    sections: tuple[EvidenceSection, ...],
) -> str:
    """
    Render a deterministic engineering answer.

    This renderer uses fixed templates and source excerpts only.
    It is not an LLM-generated answer.
    """

    lines: list[str] = []

    lines.append(
        "REQTRACE AI ENGINEERING DECISION"
    )

    lines.append(
        "=" * 40
    )

    lines.append(
        f"Assessment: {decision.assessment}"
    )

    lines.append(
        f"Assessment rule: "
        f"{decision.assessment_rule}"
    )

    lines.append(
        f"Confidence: {decision.confidence}"
    )

    lines.append(
        f"Closure status: "
        f"{decision.closure_status}"
    )

    lines.append("")

    lines.append(
        "Assessment reason:"
    )

    lines.append(
        decision.assessment_reason
    )

    lines.append("")

    lines.append(
        "Confidence reason:"
    )

    lines.append(
        decision.confidence_reason
    )

    lines.append("")

    lines.append(
        "Closure reason:"
    )

    lines.append(
        decision.closure_reason
    )

    lines.append("")

    lines.append(
        "Engineering IDs:"
    )

    lines.append(
        "  Requirements: "
        + (
            ", ".join(requirement_ids)
            if requirement_ids
            else "None"
        )
    )

    lines.append(
        "  Observations: "
        + (
            ", ".join(observation_ids)
            if observation_ids
            else "None"
        )
    )

    lines.append(
        "  Tests: "
        + (
            ", ".join(test_case_ids)
            if test_case_ids
            else "None"
        )
    )

    lines.append("")

    lines.append(
        "Supporting items:"
    )

    if supporting_items:
        for item in supporting_items:
            lines.append(
                f"  - {item}"
            )
    else:
        lines.append(
            "  - None recorded."
        )

    lines.append("")

    lines.append(
        "Blocking items:"
    )

    if blocking_items:
        for item in blocking_items:
            lines.append(
                f"  - {item}"
            )
    else:
        lines.append(
            "  - None recorded."
        )

    for section in sections:
        if section.is_empty:
            continue

        lines.append("")
        lines.append(
            section.heading + ":"
        )

        for excerpt in section.excerpts:
            citation_suffix = (
                format_citation_suffix(
                    (
                        excerpt.citation_id,
                    )
                )
            )

            lines.append(
                "  - "
                + excerpt.excerpt
                + citation_suffix
            )

            lines.append(
                "    Source: "
                + excerpt.provenance_location
            )

    lines.append("")

    lines.append(
        "Authority limitation:"
    )

    lines.append(
        "REQTRACE AI provides evidence-derived "
        "decision support. Final engineering, quality, "
        "safety, certification, release, and closure "
        "authorization remain with the responsible "
        "human authority."
    )

    return "\n".join(
        lines
    )


def build_answer_package(
    *,
    context: FinalEvidenceContext,
    contract: EngineeringAnswerContract,
    maximum_excerpt_characters: int = 500,
) -> EngineeringAnswerPackage:
    """
    Build the complete Phase 13D deterministic answer package.

    The contract must already be grounded and validated.
    """

    if not contract.grounding_passed:
        raise AnswerPackageError(
            "Cannot build answer package from "
            "an ungrounded contract."
        )

    if contract.grounding_error_count != 0:
        raise AnswerPackageError(
            "Cannot build answer package while "
            "grounding errors exist."
        )

    sections = build_evidence_sections(
        context=context,
        contract=contract,
        maximum_excerpt_characters=(
            maximum_excerpt_characters
        ),
    )

    decision = make_decision_summary(
        contract
    )

    deterministic_text = (
        render_deterministic_text(
            decision=decision,
            requirement_ids=(
                contract.requirement_ids
            ),
            observation_ids=(
                contract.observation_ids
            ),
            test_case_ids=(
                contract.test_case_ids
            ),
            supporting_items=(
                contract.supporting_items
            ),
            blocking_items=(
                contract.blocking_items
            ),
            sections=sections,
        )
    )

    return EngineeringAnswerPackage(
        package_version="1.0",
        decision=decision,
        requirement_ids=(
            contract.requirement_ids
        ),
        observation_ids=(
            contract.observation_ids
        ),
        test_case_ids=(
            contract.test_case_ids
        ),
        supporting_items=(
            contract.supporting_items
        ),
        blocking_items=(
            contract.blocking_items
        ),
        evidence_sections=sections,
        allowed_claims=(
            contract.allowed_claims
        ),
        prohibited_claims=(
            contract.prohibited_claims
        ),
        citation_allowlist=(
            contract.citation_allowlist
        ),
        engineering_id_allowlist=(
            contract.engineering_id_allowlist
        ),
        narrator_instructions=(
            contract.narrator_instructions
        ),
        grounding_passed=(
            contract.grounding_passed
        ),
        grounding_error_count=(
            contract.grounding_error_count
        ),
        grounding_warning_count=(
            contract.grounding_warning_count
        ),
        deterministic_text=(
            deterministic_text
        ),
    )
"""
Structured citation and provenance models for REQTRACE AI.

Phase 13A converts retrieved engineering evidence into validated,
citation-ready provenance records.

Supported provenance forms:

PDF evidence:
    document name
    document revision when available
    one-based page number
    stable evidence ID

Source-code evidence:
    source file
    function name when available
    one-based start and end lines
    stable evidence ID

Citation construction never invents missing provenance.

This module does not:

- generate natural-language answers
- validate engineering claims
- invoke an LLM
- modify deterministic decisions
- rewrite engineering traceability
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.context_builder import (
    FinalEvidenceContext,
)
from src.retrieval import (
    RetrievedEvidence,
)


class CitationValidationError(
    ValueError
):
    """Raised when evidence provenance is invalid."""


class CitationSourceType(
    str,
    Enum,
):
    """Supported citation source categories."""

    PDF = "PDF"
    SOURCE_CODE = "SOURCE_CODE"


CitationSourceType.__test__ = False


PDF_DOCUMENT_TYPES = {
    "frd",
    "observation",
    "test",
    "review",
}

CODE_DOCUMENT_TYPES = {
    "code",
}


@dataclass(frozen=True)
class EvidenceCitation:
    """
    One validated engineering evidence citation.

    PDF evidence uses page_number.

    Source-code evidence uses start_line and end_line.
    """

    citation_id: str
    evidence_id: str

    source_type: CitationSourceType
    document_type: str

    document_name: str | None = None
    document_revision: str | None = None
    page_number: int | None = None

    file_name: str | None = None
    function_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None

    requirement_id: str | None = None
    observation_id: str | None = None
    test_case_id: str | None = None
    result_status: str | None = None

    content_hash: str | None = None
    source_path: str | None = None

    display_label: str = ""

    def __post_init__(
        self,
    ) -> None:
        """Validate citation provenance."""

        if not self.citation_id.strip():
            raise CitationValidationError(
                "citation_id must not be empty."
            )

        if not self.evidence_id.strip():
            raise CitationValidationError(
                "evidence_id must not be empty."
            )

        normalized_type = (
            self.document_type
            .strip()
            .lower()
        )

        if not normalized_type:
            raise CitationValidationError(
                "document_type must not be empty."
            )

        if self.source_type == (
            CitationSourceType.PDF
        ):
            self._validate_pdf_citation()

        elif self.source_type == (
            CitationSourceType
            .SOURCE_CODE
        ):
            self._validate_code_citation()

        else:
            raise CitationValidationError(
                "Unsupported citation source type."
            )

        if not self.display_label.strip():
            raise CitationValidationError(
                "display_label must not be empty."
            )

        if self.content_hash is not None:
            normalized_hash = (
                self.content_hash.strip()
            )

            if len(normalized_hash) != 64:
                raise CitationValidationError(
                    "content_hash must contain "
                    "64 hexadecimal characters."
                )

            try:
                int(
                    normalized_hash,
                    16,
                )
            except ValueError as exc:
                raise CitationValidationError(
                    "content_hash must be hexadecimal."
                ) from exc

    def _validate_pdf_citation(
        self,
    ) -> None:
        """Validate PDF provenance."""

        if not self.document_name:
            raise CitationValidationError(
                "PDF citation requires document_name."
            )

        if self.page_number is None:
            raise CitationValidationError(
                "PDF citation requires page_number."
            )

        if self.page_number < 1:
            raise CitationValidationError(
                "PDF page numbers must be one-based."
            )

        if self.start_line is not None:
            raise CitationValidationError(
                "PDF citation must not contain start_line."
            )

        if self.end_line is not None:
            raise CitationValidationError(
                "PDF citation must not contain end_line."
            )

    def _validate_code_citation(
        self,
    ) -> None:
        """Validate source-code provenance."""

        if not self.file_name:
            raise CitationValidationError(
                "Code citation requires file_name."
            )

        if self.start_line is None:
            raise CitationValidationError(
                "Code citation requires start_line."
            )

        if self.end_line is None:
            raise CitationValidationError(
                "Code citation requires end_line."
            )

        if self.start_line < 1:
            raise CitationValidationError(
                "Code start_line must be one-based."
            )

        if self.end_line < self.start_line:
            raise CitationValidationError(
                "Code end_line must be greater than "
                "or equal to start_line."
            )

        if self.page_number is not None:
            raise CitationValidationError(
                "Code citation must not contain page_number."
            )

    @property
    def engineering_ids(
        self,
    ) -> tuple[str, ...]:
        """Return ordered engineering IDs carried by this citation."""

        values = (
            self.requirement_id,
            self.observation_id,
            self.test_case_id,
        )

        return tuple(
            str(value)
            for value in values
            if value
        )

    @property
    def provenance_location(
        self,
    ) -> str:
        """Return a concise human-readable source location."""

        if self.source_type == (
            CitationSourceType.PDF
        ):
            return (
                f"{self.document_name}, "
                f"page {self.page_number}"
            )

        function_part = ""

        if self.function_name:
            function_part = (
                f", function "
                f"{self.function_name}"
            )

        return (
            f"{self.file_name}"
            f"{function_part}, "
            f"lines {self.start_line}-"
            f"{self.end_line}"
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert citation into a JSON-safe dictionary."""

        return {
            "citation_id": (
                self.citation_id
            ),
            "evidence_id": (
                self.evidence_id
            ),
            "source_type": (
                self.source_type.value
            ),
            "document_type": (
                self.document_type
            ),
            "document_name": (
                self.document_name
            ),
            "document_revision": (
                self.document_revision
            ),
            "page_number": (
                self.page_number
            ),
            "file_name": (
                self.file_name
            ),
            "function_name": (
                self.function_name
            ),
            "start_line": (
                self.start_line
            ),
            "end_line": (
                self.end_line
            ),
            "language": (
                self.language
            ),
            "requirement_id": (
                self.requirement_id
            ),
            "observation_id": (
                self.observation_id
            ),
            "test_case_id": (
                self.test_case_id
            ),
            "result_status": (
                self.result_status
            ),
            "content_hash": (
                self.content_hash
            ),
            "source_path": (
                self.source_path
            ),
            "display_label": (
                self.display_label
            ),
            "provenance_location": (
                self.provenance_location
            ),
        }


@dataclass(frozen=True)
class CitationCollection:
    """Validated citation collection for one final context."""

    citations: tuple[
        EvidenceCitation,
        ...
    ]

    @property
    def citation_ids(
        self,
    ) -> tuple[str, ...]:
        """Return citation IDs in context order."""

        return tuple(
            citation.citation_id
            for citation in self.citations
        )

    @property
    def evidence_ids(
        self,
    ) -> tuple[str, ...]:
        """Return cited evidence IDs."""

        return tuple(
            citation.evidence_id
            for citation in self.citations
        )

    @property
    def record_count(
        self,
    ) -> int:
        """Return number of citations."""

        return len(
            self.citations
        )

    def by_citation_id(
        self,
        citation_id: str,
    ) -> EvidenceCitation | None:
        """Return one citation by citation ID."""

        for citation in self.citations:
            if (
                citation.citation_id
                == citation_id
            ):
                return citation

        return None

    def by_evidence_id(
        self,
        evidence_id: str,
    ) -> EvidenceCitation | None:
        """Return one citation by evidence ID."""

        for citation in self.citations:
            if (
                citation.evidence_id
                == evidence_id
            ):
                return citation

        return None

    def by_document_type(
        self,
        document_type: str,
    ) -> tuple[EvidenceCitation, ...]:
        """Return citations for one evidence type."""

        normalized_type = (
            document_type
            .strip()
            .lower()
        )

        return tuple(
            citation
            for citation in self.citations
            if (
                citation.document_type
                == normalized_type
            )
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert citation collection into JSON-safe data."""

        return {
            "citation_count": (
                self.record_count
            ),
            "citations": [
                citation.to_dict()
                for citation in self.citations
            ],
        }


def optional_string(
    metadata: dict[str, Any],
    field_name: str,
) -> str | None:
    """Return a normalized optional metadata string."""

    value = metadata.get(
        field_name
    )

    if value is None:
        return None

    normalized = str(
        value
    ).strip()

    return (
        normalized
        if normalized
        else None
    )


def optional_integer(
    metadata: dict[str, Any],
    field_name: str,
) -> int | None:
    """Return a validated optional integer metadata value."""

    value = metadata.get(
        field_name
    )

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        raise CitationValidationError(
            f"{field_name} must be an integer."
        )

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise CitationValidationError(
            f"{field_name} must be an integer."
        ) from exc


def citation_source_type(
    evidence: RetrievedEvidence,
) -> CitationSourceType:
    """Determine citation source type from document type."""

    normalized_type = (
        evidence.document_type
        .strip()
        .lower()
    )

    if normalized_type in (
        PDF_DOCUMENT_TYPES
    ):
        return CitationSourceType.PDF

    if normalized_type in (
        CODE_DOCUMENT_TYPES
    ):
        return (
            CitationSourceType
            .SOURCE_CODE
        )

    raise CitationValidationError(
        "Unsupported document type for citation: "
        f"{evidence.document_type}"
    )


def build_pdf_display_label(
    *,
    document_type: str,
    document_name: str,
    page_number: int,
    document_revision: str | None,
    requirement_id: str | None,
    observation_id: str | None,
    test_case_id: str | None,
) -> str:
    """Build a concise PDF citation label."""

    identifier = (
        requirement_id
        or observation_id
        or test_case_id
    )

    parts = [
        document_type.upper(),
        document_name,
    ]

    if document_revision:
        parts.append(
            document_revision
        )

    parts.append(
        f"page {page_number}"
    )

    if identifier:
        parts.append(
            identifier
        )

    return " | ".join(
        parts
    )


def build_code_display_label(
    *,
    file_name: str,
    function_name: str | None,
    start_line: int,
    end_line: int,
) -> str:
    """Build a concise source-code citation label."""

    function_label = (
        function_name
        if function_name
        else "<file scope>"
    )

    return (
        "CODE"
        " | "
        f"{file_name}"
        " | "
        f"{function_label}"
        " | "
        f"lines {start_line}-{end_line}"
    )


def citation_id_for_rank(
    rank: int,
) -> str:
    """Create a deterministic citation label."""

    if rank < 1:
        raise CitationValidationError(
            "Citation rank must be one-based."
        )

    return f"CIT-{rank:03d}"


def build_evidence_citation(
    evidence: RetrievedEvidence,
    *,
    citation_rank: int,
) -> EvidenceCitation:
    """Build one validated citation from retrieved evidence."""

    metadata = dict(
        evidence.metadata
    )

    source_type = citation_source_type(
        evidence
    )

    document_type = (
        evidence.document_type
        .strip()
        .lower()
    )

    evidence_id = (
        evidence.evidence_id
        .strip()
    )

    citation_id = citation_id_for_rank(
        citation_rank
    )

    document_name = optional_string(
        metadata,
        "document_name",
    )

    document_revision = optional_string(
        metadata,
        "document_revision",
    )

    page_number = optional_integer(
        metadata,
        "page_number",
    )

    file_name = optional_string(
        metadata,
        "file_name",
    )

    function_name = optional_string(
        metadata,
        "function_name",
    )

    start_line = optional_integer(
        metadata,
        "start_line",
    )

    end_line = optional_integer(
        metadata,
        "end_line",
    )

    language = optional_string(
        metadata,
        "language",
    )

    requirement_id = optional_string(
        metadata,
        "requirement_id",
    )

    observation_id = optional_string(
        metadata,
        "observation_id",
    )

    test_case_id = optional_string(
        metadata,
        "test_case_id",
    )

    result_status = optional_string(
        metadata,
        "result_status",
    )

    content_hash = optional_string(
        metadata,
        "content_hash",
    )

    source_path = optional_string(
        metadata,
        "file_path",
    )

    if source_type == CitationSourceType.PDF:
        if not document_name:
            raise CitationValidationError(
                "PDF evidence is missing document_name: "
                f"{evidence_id}"
            )

        if page_number is None:
            raise CitationValidationError(
                "PDF evidence is missing page_number: "
                f"{evidence_id}"
            )

        display_label = build_pdf_display_label(
            document_type=document_type,
            document_name=document_name,
            page_number=page_number,
            document_revision=(
                document_revision
            ),
            requirement_id=requirement_id,
            observation_id=observation_id,
            test_case_id=test_case_id,
        )

    else:
        if not file_name:
            raise CitationValidationError(
                "Code evidence is missing file_name: "
                f"{evidence_id}"
            )

        if start_line is None:
            raise CitationValidationError(
                "Code evidence is missing start_line: "
                f"{evidence_id}"
            )

        if end_line is None:
            raise CitationValidationError(
                "Code evidence is missing end_line: "
                f"{evidence_id}"
            )

        display_label = build_code_display_label(
            file_name=file_name,
            function_name=function_name,
            start_line=start_line,
            end_line=end_line,
        )

    return EvidenceCitation(
        citation_id=citation_id,
        evidence_id=evidence_id,
        source_type=source_type,
        document_type=document_type,
        document_name=document_name,
        document_revision=(
            document_revision
        ),
        page_number=page_number,
        file_name=file_name,
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
        language=language,
        requirement_id=requirement_id,
        observation_id=observation_id,
        test_case_id=test_case_id,
        result_status=result_status,
        content_hash=content_hash,
        source_path=source_path,
        display_label=display_label,
    )


def build_citation_collection(
    context: FinalEvidenceContext,
) -> CitationCollection:
    """
    Build validated citations for every final-context record.

    Citation order follows the final evidence-context order.
    """

    citations: list[
        EvidenceCitation
    ] = []

    seen_evidence_ids: set[str] = set()

    for citation_rank, evidence in enumerate(
        context.evidence,
        start=1,
    ):
        if (
            evidence.evidence_id
            in seen_evidence_ids
        ):
            raise CitationValidationError(
                "Duplicate evidence ID in final context: "
                f"{evidence.evidence_id}"
            )

        seen_evidence_ids.add(
            evidence.evidence_id
        )

        citations.append(
            build_evidence_citation(
                evidence,
                citation_rank=(
                    citation_rank
                ),
            )
        )

    return CitationCollection(
        citations=tuple(
            citations
        )
    )
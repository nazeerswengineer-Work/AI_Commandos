"""
Evidence identifiers, hashing, source references, and evidence records.

Evidence IDs are generated deterministically from validated source metadata.
The LLM is never responsible for creating evidence IDs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.metadata import DocumentType, EvidenceMetadata
from src.normalization import (
    normalize_multiline_content,
    normalize_whitespace,
)


UNSAFE_EVIDENCE_COMPONENT = re.compile(r"[:\r\n]+")


def calculate_content_hash(content: str) -> str:
    """Create a SHA-256 hash from normalized evidence content."""

    normalized_content = normalize_multiline_content(content)

    if not normalized_content:
        raise ValueError(
            "Cannot calculate an evidence hash from empty content."
        )

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()


def calculate_file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash from a file's binary content."""

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def sanitize_evidence_component(value: str) -> str:
    """
    Sanitize one evidence-ID component.

    Colons and line breaks are removed because double colons delimit
    evidence-ID components.
    """

    cleaned = normalize_whitespace(value)
    cleaned = UNSAFE_EVIDENCE_COMPONENT.sub("-", cleaned)
    cleaned = cleaned.strip(" -")

    if not cleaned:
        raise ValueError(
            "An evidence-ID component became empty after sanitization."
        )

    return cleaned


def build_frd_evidence_id(
    *,
    document_name: str,
    page_number: int,
    requirement_id: str,
) -> str:
    """Build a stable FRD evidence ID."""

    if page_number < 1:
        raise ValueError("page_number must be one-based.")

    return (
        f"FRD::{sanitize_evidence_component(document_name)}"
        f"::PAGE-{page_number}"
        f"::{sanitize_evidence_component(requirement_id)}"
    )


def build_observation_evidence_id(
    *,
    document_name: str,
    page_number: int,
    observation_id: str,
) -> str:
    """Build a stable observation evidence ID."""

    if page_number < 1:
        raise ValueError("page_number must be one-based.")

    return (
        f"OBS::{sanitize_evidence_component(document_name)}"
        f"::PAGE-{page_number}"
        f"::{sanitize_evidence_component(observation_id)}"
    )


def build_test_evidence_id(
    *,
    document_name: str,
    page_number: int,
    test_case_id: str,
) -> str:
    """Build a stable test-evidence ID."""

    if page_number < 1:
        raise ValueError("page_number must be one-based.")

    return (
        f"TEST::{sanitize_evidence_component(document_name)}"
        f"::PAGE-{page_number}"
        f"::{sanitize_evidence_component(test_case_id)}"
    )


def build_review_evidence_id(
    *,
    document_name: str,
    page_number: int,
    review_anchor: str,
) -> str:
    """Build a stable review-evidence ID."""

    if page_number < 1:
        raise ValueError("page_number must be one-based.")

    return (
        f"REVIEW::{sanitize_evidence_component(document_name)}"
        f"::PAGE-{page_number}"
        f"::{sanitize_evidence_component(review_anchor)}"
    )


def build_code_evidence_id(
    *,
    file_name: str,
    start_line: int,
    end_line: int,
    function_name: str | None = None,
) -> str:
    """Build a stable source-code evidence ID."""

    if start_line < 1:
        raise ValueError("start_line must be one-based.")

    if end_line < start_line:
        raise ValueError(
            "end_line must be greater than or equal to start_line."
        )

    symbol = function_name or "LINE-WINDOW"

    return (
        f"CODE::{sanitize_evidence_component(file_name)}"
        f"::{sanitize_evidence_component(symbol)}"
        f"::{start_line}-{end_line}"
    )


def build_evidence_id(
    metadata: EvidenceMetadata,
    *,
    review_anchor: str | None = None,
) -> str:
    """Build an evidence ID from metadata fields."""

    if metadata.document_type == DocumentType.FRD:
        if metadata.page_number is None:
            raise ValueError("FRD page_number is unavailable.")

        if metadata.requirement_id is None:
            raise ValueError("FRD requirement_id is unavailable.")

        return build_frd_evidence_id(
            document_name=metadata.document_name,
            page_number=metadata.page_number,
            requirement_id=metadata.requirement_id,
        )

    if metadata.document_type == DocumentType.OBSERVATION:
        if metadata.page_number is None:
            raise ValueError("Observation page_number is unavailable.")

        if metadata.observation_id is None:
            raise ValueError("Observation ID is unavailable.")

        return build_observation_evidence_id(
            document_name=metadata.document_name,
            page_number=metadata.page_number,
            observation_id=metadata.observation_id,
        )

    if metadata.document_type == DocumentType.CODE:
        if metadata.file_name is None:
            raise ValueError("Code file_name is unavailable.")

        if metadata.start_line is None or metadata.end_line is None:
            raise ValueError("Code line range is unavailable.")

        return build_code_evidence_id(
            file_name=metadata.file_name,
            function_name=metadata.function_name,
            start_line=metadata.start_line,
            end_line=metadata.end_line,
        )

    if metadata.document_type == DocumentType.TEST:
        if metadata.page_number is None:
            raise ValueError("Test page_number is unavailable.")

        if metadata.test_case_id is None:
            raise ValueError("Test case ID is unavailable.")

        return build_test_evidence_id(
            document_name=metadata.document_name,
            page_number=metadata.page_number,
            test_case_id=metadata.test_case_id,
        )

    if metadata.document_type == DocumentType.REVIEW:
        if metadata.page_number is None:
            raise ValueError("Review page_number is unavailable.")

        anchor = (
            review_anchor
            or metadata.observation_id
            or metadata.requirement_id
            or metadata.test_case_id
            or metadata.chunk_id
        )

        return build_review_evidence_id(
            document_name=metadata.document_name,
            page_number=metadata.page_number,
            review_anchor=anchor,
        )

    raise ValueError(
        f"Unsupported document type: {metadata.document_type}"
    )


class RetrievalScores(BaseModel):
    """Optional retrieval scores associated with evidence."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    retrieval_method: str | None = None
    semantic_score: float | None = None
    keyword_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = Field(default=None, ge=1)

    @field_validator("retrieval_method", mode="before")
    @classmethod
    def normalize_retrieval_method(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        cleaned = normalize_whitespace(str(value))
        return cleaned or None


class EvidenceRecord(BaseModel):
    """One validated evidence chunk and its source metadata."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    content: str = Field(min_length=1)
    metadata: EvidenceMetadata
    scores: RetrievalScores | None = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(
        cls,
        value: Any,
    ) -> str:
        normalized = normalize_multiline_content(str(value))

        if not normalized:
            raise ValueError("Evidence content must not be empty.")

        return normalized

    @model_validator(mode="after")
    def validate_content_hash(
        self,
    ) -> "EvidenceRecord":
        """Ensure metadata hash matches the supplied evidence content."""

        calculated_hash = calculate_content_hash(self.content)

        if calculated_hash != self.metadata.content_hash:
            raise ValueError(
                "metadata.content_hash does not match evidence content."
            )

        expected_evidence_id = build_evidence_id(self.metadata)

        if self.metadata.evidence_id != expected_evidence_id:
            raise ValueError(
                "metadata.evidence_id does not match the deterministic "
                f"evidence ID. Expected: {expected_evidence_id}"
            )

        return self

    def citation_anchor(self) -> str:
        """Return the validated citation anchor."""

        return f"[{self.metadata.evidence_id}]"

    def short_excerpt(self, limit: int = 400) -> str:
        """Return a compact whitespace-normalized excerpt."""

        if limit < 20:
            raise ValueError("Excerpt limit must be at least 20.")

        normalized = normalize_whitespace(self.content)

        if len(normalized) <= limit:
            return normalized

        return normalized[: limit - 3].rstrip() + "..."
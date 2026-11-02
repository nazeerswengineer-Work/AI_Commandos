"""
Validated metadata models for REQTRACE AI evidence.

Metadata contains only source-derived or deterministically generated values.
Unknown metadata remains None and must never be fabricated.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.normalization import (
    normalize_observation_id,
    normalize_optional_observation_id,
    normalize_optional_requirement_id,
    normalize_optional_test_case_id,
    normalize_requirement_id,
    normalize_revision,
    normalize_test_case_id,
    normalize_whitespace,
)


class DocumentType(str, Enum):
    """Supported engineering evidence types."""

    FRD = "frd"
    OBSERVATION = "observation"
    CODE = "code"
    TEST = "test"
    REVIEW = "review"


class ParsingMethod(str, Enum):
    """How source structure was identified."""

    REQUIREMENT = "requirement"
    OBSERVATION = "observation"
    TEST_CASE = "test_case"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    LOGICAL_BLOCK = "logical_block"
    LINE_WINDOW = "line_window"
    PAGE = "page"
    SECTION = "section"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class EvidenceMetadata(BaseModel):
    """
    Consistent metadata attached to one source-evidence chunk.

    Page numbers and source-code line numbers are one-based.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    evidence_id: str = Field(min_length=3)
    document_type: DocumentType
    document_name: str = Field(min_length=1)
    document_revision: str | None = None

    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None

    requirement_id: str | None = None
    observation_id: str | None = None
    test_case_id: str | None = None
    result_status: str | None = None

    file_name: str | None = None
    file_path: str | None = None
    function_name: str | None = None

    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    language: str | None = None

    chunk_id: str = Field(min_length=3)
    content_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )

    ingestion_timestamp: datetime | None = None
    source_modified_timestamp: datetime | None = None

    parsing_method: ParsingMethod = ParsingMethod.UNKNOWN

    @field_validator(
        "evidence_id",
        "document_name",
        "section",
        "result_status",
        "file_name",
        "file_path",
        "function_name",
        "language",
        "chunk_id",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(
        cls,
        value: Any,
    ) -> Any:
        """Trim ordinary metadata strings while preserving None."""

        if value is None:
            return None

        if isinstance(value, str):
            cleaned = normalize_whitespace(value)
            return cleaned or None

        return value

    @field_validator("requirement_id", mode="before")
    @classmethod
    def validate_requirement_id(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return normalize_optional_requirement_id(str(value))

    @field_validator("observation_id", mode="before")
    @classmethod
    def validate_observation_id(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return normalize_optional_observation_id(str(value))

    @field_validator("test_case_id", mode="before")
    @classmethod
    def validate_test_case_id(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return normalize_optional_test_case_id(str(value))

    @field_validator("document_revision", mode="before")
    @classmethod
    def validate_revision(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return normalize_revision(str(value))

    @field_validator("content_hash", mode="before")
    @classmethod
    def normalize_content_hash(
        cls,
        value: Any,
    ) -> str:
        return str(value).strip().lower()

    @field_validator("result_status", mode="after")
    @classmethod
    def normalize_result_status(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return value.upper()

    @field_validator("language", mode="after")
    @classmethod
    def normalize_language(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return value.lower()

    @model_validator(mode="after")
    def validate_metadata_relationships(
        self,
    ) -> "EvidenceMetadata":
        """Validate relationships between metadata fields."""

        if (self.start_line is None) != (self.end_line is None):
            raise ValueError(
                "start_line and end_line must either both be provided "
                "or both be unavailable."
            )

        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError(
                "end_line must be greater than or equal to start_line."
            )

        if self.document_type == DocumentType.CODE:
            if self.page_number is not None:
                raise ValueError(
                    "Code evidence must not contain a PDF page number."
                )

            if self.file_name is None:
                raise ValueError(
                    "Code evidence requires file_name."
                )

            if self.start_line is None or self.end_line is None:
                raise ValueError(
                    "Code evidence requires one-based start_line "
                    "and end_line values."
                )
        else:
            if self.start_line is not None or self.end_line is not None:
                raise ValueError(
                    "Non-code evidence must not contain source-code "
                    "line numbers."
                )

        if self.document_type == DocumentType.FRD:
            if self.requirement_id is None:
                raise ValueError(
                    "FRD evidence requires requirement_id."
                )

            if self.page_number is None:
                raise ValueError(
                    "FRD evidence requires page_number."
                )

        if self.document_type == DocumentType.OBSERVATION:
            if self.observation_id is None:
                raise ValueError(
                    "Observation evidence requires observation_id."
                )

            if self.page_number is None:
                raise ValueError(
                    "Observation evidence requires page_number."
                )

        if self.document_type == DocumentType.TEST:
            if self.test_case_id is None:
                raise ValueError(
                    "Test evidence requires test_case_id."
                )

            if self.page_number is None:
                raise ValueError(
                    "Test evidence requires page_number."
                )

        if self.document_type == DocumentType.REVIEW:
            if self.page_number is None:
                raise ValueError(
                    "Review evidence requires page_number."
                )

        return self

    def to_chroma_metadata(self) -> dict[str, Any]:
        """
        Convert metadata into Chroma-compatible scalar fields.

        None values are removed rather than replaced with fabricated text.
        """

        raw_metadata = self.model_dump(
            mode="json",
            exclude_none=True,
        )

        return {
            key: value
            for key, value in raw_metadata.items()
            if isinstance(value, (str, int, float, bool))
        }


def create_minimal_frd_metadata(
    *,
    evidence_id: str,
    document_name: str,
    requirement_id: str,
    page_number: int,
    chunk_id: str,
    content_hash: str,
    document_revision: str | None = None,
    section: str | None = None,
) -> EvidenceMetadata:
    """Create validated minimum FRD metadata."""

    return EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.FRD,
        document_name=document_name,
        document_revision=document_revision,
        page_number=page_number,
        section=section,
        requirement_id=normalize_requirement_id(requirement_id),
        chunk_id=chunk_id,
        content_hash=content_hash,
        parsing_method=ParsingMethod.REQUIREMENT,
    )


def create_minimal_observation_metadata(
    *,
    evidence_id: str,
    document_name: str,
    observation_id: str,
    page_number: int,
    chunk_id: str,
    content_hash: str,
    requirement_id: str | None = None,
    test_case_id: str | None = None,
) -> EvidenceMetadata:
    """Create validated minimum observation metadata."""

    return EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.OBSERVATION,
        document_name=document_name,
        page_number=page_number,
        observation_id=normalize_observation_id(observation_id),
        requirement_id=requirement_id,
        test_case_id=test_case_id,
        chunk_id=chunk_id,
        content_hash=content_hash,
        parsing_method=ParsingMethod.OBSERVATION,
    )


def create_minimal_test_metadata(
    *,
    evidence_id: str,
    document_name: str,
    test_case_id: str,
    page_number: int,
    chunk_id: str,
    content_hash: str,
    requirement_id: str | None = None,
    result_status: str | None = None,
) -> EvidenceMetadata:
    """Create validated minimum test metadata."""

    return EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.TEST,
        document_name=document_name,
        page_number=page_number,
        test_case_id=normalize_test_case_id(test_case_id),
        requirement_id=requirement_id,
        result_status=result_status,
        chunk_id=chunk_id,
        content_hash=content_hash,
        parsing_method=ParsingMethod.TEST_CASE,
    )


def create_minimal_code_metadata(
    *,
    evidence_id: str,
    file_name: str,
    start_line: int,
    end_line: int,
    chunk_id: str,
    content_hash: str,
    function_name: str | None = None,
    language: str | None = None,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
    file_path: str | Path | None = None,
    parsing_method: ParsingMethod = ParsingMethod.FUNCTION,
) -> EvidenceMetadata:
    """Create validated minimum code metadata."""

    return EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.CODE,
        document_name=file_name,
        file_name=file_name,
        file_path=str(file_path) if file_path is not None else None,
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
        language=language,
        requirement_id=requirement_id,
        observation_id=observation_id,
        test_case_id=test_case_id,
        chunk_id=chunk_id,
        content_hash=content_hash,
        parsing_method=parsing_method,
    )
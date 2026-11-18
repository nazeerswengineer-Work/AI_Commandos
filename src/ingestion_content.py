"""
Normalized content models for dynamic REQTRACE AI ingestion.

These models represent parsed source material before embedding and
ChromaDB insertion.

No metadata value is fabricated. Missing values remain None or empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class IngestionContentError(
    ValueError
):
    """Raised when parsed ingestion content is invalid."""


@dataclass(frozen=True)
class ParsedSection:
    """
    One physically traceable section extracted from a source file.

    For documents:
        page_number and section_name may be available.

    For source code:
        function_name, class_name and line ranges may be available.
    """

    content: str

    section_index: int

    section_name: str | None = None
    page_number: int | None = None

    line_number_start: int | None = None
    line_number_end: int | None = None

    function_name: str | None = None
    class_name: str | None = None

    requirement_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    test_case_ids: tuple[str, ...] = ()

    result_status: str | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate parsed section."""

        if not self.content.strip():
            raise IngestionContentError(
                "Parsed section content must not be empty."
            )

        if self.section_index < 0:
            raise IngestionContentError(
                "section_index must not be negative."
            )

        if (
            self.page_number is not None
            and self.page_number < 1
        ):
            raise IngestionContentError(
                "page_number must be one-based."
            )

        if (
            self.line_number_start is not None
            and self.line_number_start < 1
        ):
            raise IngestionContentError(
                "line_number_start must be one-based."
            )

        if (
            self.line_number_end is not None
            and self.line_number_end < 1
        ):
            raise IngestionContentError(
                "line_number_end must be one-based."
            )

        if (
            self.line_number_start is not None
            and self.line_number_end is not None
            and self.line_number_end
            < self.line_number_start
        ):
            raise IngestionContentError(
                "line_number_end cannot precede "
                "line_number_start."
            )


@dataclass(frozen=True)
class ParsedDocument:
    """Complete normalized parsing result for one source file."""

    document_id: str
    document_name: str
    relative_path: str

    extension: str
    file_hash: str

    document_type: str
    document_category: str

    parsing_method: str

    sections: tuple[
        ParsedSection,
        ...
    ]

    def __post_init__(
        self,
    ) -> None:
        """Validate parsed document."""

        required = {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "relative_path": self.relative_path,
            "extension": self.extension,
            "file_hash": self.file_hash,
            "document_type": self.document_type,
            "document_category": (
                self.document_category
            ),
            "parsing_method": self.parsing_method,
        }

        for field_name, value in required.items():
            if not str(value).strip():
                raise IngestionContentError(
                    f"{field_name} must not be empty."
                )

        if len(self.document_id) != 64:
            raise IngestionContentError(
                "document_id must be SHA-256."
            )

        if len(self.file_hash) != 64:
            raise IngestionContentError(
                "file_hash must be SHA-256."
            )

        if not self.sections:
            raise IngestionContentError(
                "Parsed document must contain "
                "at least one section."
            )


@dataclass(frozen=True)
class IngestionChunk:
    """
    One embedding-ready engineering evidence chunk.

    chunk_id is stable for the document content version and physical
    chunk location.
    """

    chunk_id: str

    document_id: str
    document_name: str
    relative_path: str

    document_type: str
    document_category: str

    file_hash: str

    content: str
    content_hash: str

    chunk_index: int

    section_name: str | None = None
    page_number: int | None = None

    line_number_start: int | None = None
    line_number_end: int | None = None

    function_name: str | None = None
    class_name: str | None = None

    requirement_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    test_case_ids: tuple[str, ...] = ()

    result_status: str | None = None

    parsing_method: str | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate embedding-ready chunk."""

        if not self.chunk_id.strip():
            raise IngestionContentError(
                "chunk_id must not be empty."
            )

        if not self.content.strip():
            raise IngestionContentError(
                "Chunk content must not be empty."
            )

        if len(self.content_hash) != 64:
            raise IngestionContentError(
                "content_hash must be SHA-256."
            )

        if self.chunk_index < 0:
            raise IngestionContentError(
                "chunk_index must not be negative."
            )

    def to_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Convert chunk metadata into a vector-store-safe mapping.

        Lists are serialized as pipe-separated strings because some
        Chroma versions allow only scalar metadata values.
        """

        metadata: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "relative_path": self.relative_path,
            "document_type": self.document_type,
            "document_category": (
                self.document_category
            ),
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "chunk_index": self.chunk_index,
        }

        optional_values = {
            "section_name": self.section_name,
            "page_number": self.page_number,
            "line_number_start": (
                self.line_number_start
            ),
            "line_number_end": (
                self.line_number_end
            ),
            "function_name": self.function_name,
            "class_name": self.class_name,
            "result_status": self.result_status,
            "parsing_method": self.parsing_method,
        }

        for key, value in optional_values.items():
            if value is not None:
                metadata[key] = value

        if self.requirement_ids:
            metadata["requirement_ids"] = "|".join(
                self.requirement_ids
            )

            if len(self.requirement_ids) == 1:
                metadata["requirement_id"] = (
                    self.requirement_ids[0]
                )

        if self.observation_ids:
            metadata["observation_ids"] = "|".join(
                self.observation_ids
            )

            if len(self.observation_ids) == 1:
                metadata["observation_id"] = (
                    self.observation_ids[0]
                )

        if self.test_case_ids:

            metadata[
                "test_case_ids"
            ] = "|".join(
                self.test_case_ids
            )

            # A test-case reference inside an FRD, observation,
            # code file, or review does not constitute actual
            # Test evidence.
            #
            # Only Test evidence receives singular test_case_id,
            # which the grounding layer treats as authoritative
            # test provenance.
            if (
                self.document_type
                .strip()
                .lower()
                == "test"
                and len(
                    self.test_case_ids
                )
                == 1
            ):

                metadata[
                    "test_case_id"
                ] = (
                    self.test_case_ids[0]
                )

        return metadata
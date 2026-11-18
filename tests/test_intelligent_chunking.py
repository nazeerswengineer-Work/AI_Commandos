"""
Tests for embedding-ready intelligent chunks.
"""

from __future__ import annotations

from src.ingestion_content import (
    ParsedDocument,
    ParsedSection,
)

from src.intelligent_chunking import (
    ChunkingConfig,
    chunk_parsed_document,
)


DOCUMENT_ID = "a" * 64
FILE_HASH = "b" * 64


def make_document(
    *,
    document_type: str,
    section: ParsedSection,
) -> ParsedDocument:
    """Create parsed-document fixture."""

    return ParsedDocument(
        document_id=DOCUMENT_ID,
        document_name="sample.md",
        relative_path="sample.md",
        extension=".md",
        file_hash=FILE_HASH,
        document_type=document_type,
        document_category="Other",
        parsing_method="test",
        sections=(
            section,
        ),
    )


def test_short_section_remains_whole() -> None:
    """Small sections should remain one chunk."""

    document = make_document(
        document_type="frd",
        section=ParsedSection(
            content=(
                "REQ-017 requires safe null-input "
                "handling by the software component."
            ),
            section_index=0,
            requirement_ids=(
                "REQ-017",
            ),
        ),
    )

    chunks = chunk_parsed_document(
        document
    )

    assert len(chunks) == 1

    assert chunks[0].requirement_ids == (
        "REQ-017",
    )


def test_result_metadata_is_preserved() -> None:
    """Test result should survive chunk construction."""

    document = make_document(
        document_type="test",
        section=ParsedSection(
            content=(
                "TC-102 Result Status: FAIL. "
                "Null input caused the observed failure."
            ),
            section_index=0,
            test_case_ids=(
                "TC-102",
            ),
            result_status="FAIL",
        ),
    )

    chunks = chunk_parsed_document(
        document
    )

    assert (
        chunks[0].test_case_ids
        == (
            "TC-102",
        )
    )

    assert (
        chunks[0].result_status
        == "FAIL"
    )


def test_long_code_preserves_line_ranges() -> None:
    """Long source functions should split by physical lines."""

    lines = [
        f"    value += {index};"
        for index in range(
            200
        )
    ]

    content = (
        "void process_data(void)\n"
        "{\n"
        + "\n".join(lines)
        + "\n}"
    )

    document = make_document(
        document_type="code",
        section=ParsedSection(
            content=content,
            section_index=0,
            function_name=(
                "process_data"
            ),
            line_number_start=10,
            line_number_end=(
                10
                + len(
                    content.splitlines()
                )
                - 1
            ),
        ),
    )

    chunks = chunk_parsed_document(
        document,
        config=ChunkingConfig(
            maximum_characters=600,
            overlap_characters=0,
        ),
    )

    assert len(chunks) > 1

    assert all(
        chunk.function_name
        == "process_data"
        for chunk in chunks
    )

    assert (
        chunks[0]
        .line_number_start
        == 10
    )

    assert (
        chunks[-1]
        .line_number_end
        == document
        .sections[0]
        .line_number_end
    )


def test_vector_metadata_is_scalar_safe() -> None:
    """
    Vector metadata must remain scalar-safe.

    An Observation may reference a Test Case, but that reference
    must not be promoted to authoritative Test evidence.
    """

    document = make_document(
        document_type="observation",
        section=ParsedSection(
            content=(
                "OBS-023 references REQ-017 "
                "and TC-102."
            ),
            section_index=0,
            requirement_ids=(
                "REQ-017",
            ),
            observation_ids=(
                "OBS-023",
            ),
            test_case_ids=(
                "TC-102",
            ),
        ),
    )

    chunk = (
        chunk_parsed_document(
            document
        )[0]
    )

    metadata = (
        chunk.to_metadata()
    )

    assert (
        metadata[
            "requirement_id"
        ]
        == "REQ-017"
    )

    assert (
        metadata[
            "observation_id"
        ]
        == "OBS-023"
    )

    # TC-102 remains available as traceability metadata.
    assert (
        metadata[
            "test_case_ids"
        ]
        == "TC-102"
    )

    # Observation evidence is NOT actual Test evidence.
    assert (
        "test_case_id"
        not in metadata
    )

    # Chroma-compatible plural metadata remains scalar.
    assert isinstance(
        metadata[
            "requirement_ids"
        ],
        str,
    )

    assert isinstance(
        metadata[
            "observation_ids"
        ],
        str,
    )

    assert isinstance(
        metadata[
            "test_case_ids"
        ],
        str,
    )

def test_frd_test_reference_is_not_actual_test_evidence() -> None:

    document = make_document(
        document_type="frd",
        section=ParsedSection(
            content=(
                "REQ-900 shall reject null input. "
                "Verification shall be performed by TC-900."
            ),
            section_index=0,
            requirement_ids=(
                "REQ-900",
            ),
            test_case_ids=(
                "TC-900",
            ),
        ),
    )

    chunk = (
        chunk_parsed_document(
            document
        )[0]
    )

    metadata = (
        chunk.to_metadata()
    )

    assert (
        metadata[
            "requirement_id"
        ]
        == "REQ-900"
    )

    assert (
        metadata[
            "test_case_ids"
        ]
        == "TC-900"
    )

    assert (
        "test_case_id"
        not in metadata
    )

def test_test_document_exposes_authoritative_test_case_id() -> None:

    document = make_document(
        document_type="test",
        section=ParsedSection(
            content=(
                "TC-900 Result Status: PASS. "
                "Verification of REQ-900 completed."
            ),
            section_index=0,
            requirement_ids=(
                "REQ-900",
            ),
            test_case_ids=(
                "TC-900",
            ),
            result_status="PASS",
        ),
    )

    chunk = (
        chunk_parsed_document(
            document
        )[0]
    )

    metadata = (
        chunk.to_metadata()
    )

    assert (
        metadata[
            "test_case_ids"
        ]
        == "TC-900"
    )

    assert (
        metadata[
            "test_case_id"
        ]
        == "TC-900"
    )

    assert (
        metadata[
            "result_status"
        ]
        == "PASS"
    )
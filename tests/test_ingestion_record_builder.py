"""
Tests for dynamic manifest record construction.
"""

from __future__ import annotations

from src.ingestion_content import (
    IngestionChunk,
)

from src.ingestion_record_builder import (
    dynamic_manifest_record,
    remove_records_for_chunk_ids,
    replace_document_records,
)


def make_chunk() -> IngestionChunk:

    return IngestionChunk(
        chunk_id="dynamic-001",
        document_id="a" * 64,
        document_name="FRD_REQ-900.md",
        relative_path=(
            "dynamic_test/FRD_REQ-900.md"
        ),
        document_type="frd",
        document_category="FRD",
        file_hash="b" * 64,
        content=(
            "REQ-900 requires null-input rejection."
        ),
        content_hash="c" * 64,
        chunk_index=0,
        requirement_ids=(
            "REQ-900",
        ),
        parsing_method=(
            "markdown_heading"
        ),
    )


def test_dynamic_record_contains_retrieval_fields() -> None:

    record = dynamic_manifest_record(
        make_chunk()
    )

    assert (
        record["evidence_id"]
        == "dynamic-001"
    )

    assert (
        record["requirement_id"]
        == "REQ-900"
    )

    assert (
        record["document_type"]
        == "frd"
    )

    assert (
        record["dynamic_ingestion"]
        is True
    )


def test_remove_records_by_chunk_id() -> None:

    records = [
        {
            "evidence_id": "old-001",
        },
        {
            "evidence_id": "keep-001",
        },
    ]

    result = remove_records_for_chunk_ids(
        records=records,
        chunk_ids=[
            "old-001"
        ],
    )

    assert result == [
        {
            "evidence_id": "keep-001",
        }
    ]


def test_replace_document_records() -> None:

    records = [
        {
            "evidence_id": "old-001",
        },
        {
            "evidence_id": "keep-001",
        },
    ]

    result = replace_document_records(
        records=records,
        previous_chunk_ids=[
            "old-001"
        ],
        new_chunks=[
            make_chunk()
        ],
    )

    identifiers = {
        record["evidence_id"]
        for record in result
    }

    assert "old-001" not in identifiers
    assert "keep-001" in identifiers
    assert "dynamic-001" in identifiers
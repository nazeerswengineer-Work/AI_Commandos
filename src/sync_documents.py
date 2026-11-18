"""
Build production manifest records from dynamic ingestion chunks.

records[] remains the evidence-level catalogue.

documents{} remains the source-file synchronization ledger.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.ingestion_content import (
    IngestionChunk,
)


def dynamic_manifest_record(
    chunk: IngestionChunk,
) -> dict[str, Any]:
    """Convert one dynamic chunk into an evidence manifest record."""

    metadata = chunk.to_metadata()

    record: dict[str, Any] = {
        "evidence_id": chunk.chunk_id,
        "chunk_id": chunk.chunk_id,

        "document_id": (
            chunk.document_id
        ),

        "document_name": (
            chunk.document_name
        ),

        "relative_path": (
            chunk.relative_path
        ),

        "document_type": (
            chunk.document_type
        ),

        "document_category": (
            chunk.document_category
        ),

        "file_hash": (
            chunk.file_hash
        ),

        "content_hash": (
            chunk.content_hash
        ),

        "content": chunk.content,

        "chunk_index": (
            chunk.chunk_index
        ),

        "dynamic_ingestion": True,
    }

    optional_values = {
        "section_name": (
            chunk.section_name
        ),

        "page_number": (
            chunk.page_number
        ),

        "line_number_start": (
            chunk.line_number_start
        ),

        "line_number_end": (
            chunk.line_number_end
        ),

        "function_name": (
            chunk.function_name
        ),

        "class_name": (
            chunk.class_name
        ),

        "result_status": (
            chunk.result_status
        ),

        "parsing_method": (
            chunk.parsing_method
        ),
    }

    for key, value in optional_values.items():
        if value is not None:
            record[key] = value

    for key in (
        "requirement_id",
        "observation_id",
        "test_case_id",
        "requirement_ids",
        "observation_ids",
        "test_case_ids",
    ):
        if key in metadata:
            record[key] = metadata[key]

    return record


def dynamic_manifest_records(
    chunks: Iterable[
        IngestionChunk
    ],
) -> tuple[
    dict[str, Any],
    ...
]:
    """Convert dynamic chunks into manifest evidence records."""

    return tuple(
        dynamic_manifest_record(
            chunk
        )
        for chunk in chunks
    )


def record_chunk_id(
    record: dict[str, Any],
) -> str | None:
    """Return the vector identity represented by one manifest record."""

    for field_name in (
        "chunk_id",
        "evidence_id",
    ):
        raw_value = record.get(
            field_name
        )

        if raw_value is None:
            continue

        value = str(
            raw_value
        ).strip()

        if value:
            return value

    return None


def remove_records_for_chunk_ids(
    *,
    records: list[
        dict[str, Any]
    ],
    chunk_ids: Iterable[str],
) -> list[
    dict[str, Any]
]:
    """Remove records associated with specific vector IDs."""

    identifiers = {
        str(value).strip()
        for value in chunk_ids
        if str(value).strip()
    }

    if not identifiers:
        return [
            dict(record)
            for record in records
        ]

    return [
        dict(record)
        for record in records
        if record_chunk_id(
            record
        ) not in identifiers
    ]


def replace_document_records(
    *,
    records: list[
        dict[str, Any]
    ],
    previous_chunk_ids: Iterable[str],
    new_chunks: Iterable[
        IngestionChunk
    ],
) -> list[
    dict[str, Any]
]:
    """Replace one document's old evidence records with new records."""

    retained = remove_records_for_chunk_ids(
        records=records,
        chunk_ids=previous_chunk_ids,
    )

    retained.extend(
        dynamic_manifest_records(
            new_chunks
        )
    )

    return retained


def count_records_by_type(
    records: Iterable[
        dict[str, Any]
    ],
) -> dict[str, int]:
    """Recalculate evidence-record type counts."""

    output: dict[
        str,
        int,
    ] = {}

    for record in records:
        document_type = (
            str(
                record.get(
                    "document_type",
                    "other",
                )
            )
            .strip()
            .lower()
            or "other"
        )

        output[document_type] = (
            output.get(
                document_type,
                0,
            )
            + 1
        )

    return dict(
        sorted(
            output.items()
        )
    )

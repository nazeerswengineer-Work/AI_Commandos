"""
Tests for Phase 16H.2 baseline-ledger bootstrap.
"""

from __future__ import annotations

from pathlib import Path

from src.document_discovery import (
    discover_documents,
)
from src.ingestion_manifest import (
    IngestionManifest,
)
from src.manifest_bootstrap import (
    bootstrap_document_ledger,
)


def test_existing_record_bootstraps_document(
    tmp_path,
) -> None:
    """Existing indexed evidence should create an INDEXED ledger entry."""

    documents_root = (
        tmp_path
        / "documents"
    )

    documents_root.mkdir()

    document_path = (
        documents_root
        / "OBS-080.md"
    )

    document_path.write_text(
        "# OBS-080\nObserved behavior.",
        encoding="utf-8",
    )

    manifest = (
        IngestionManifest.empty()
    )

    manifest.records.append(
        {
            "evidence_id": (
                "OBS::OBS-080"
            ),
            "chunk_id": (
                "obs-080-chunk-001"
            ),
            "document_type": (
                "observation"
            ),
            "document_name": (
                "OBS-080.md"
            ),
            "observation_id": (
                "OBS-080"
            ),
        }
    )

    original_records = list(
        manifest.records
    )

    result = bootstrap_document_ledger(
        manifest=manifest,
        documents_root=documents_root,
    )

    assert result.passed is True
    assert result.matched_documents == 1

    assert (
        "OBS-080.md"
        in manifest.documents
    )

    entry = manifest.documents[
        "OBS-080.md"
    ]

    assert entry.chunk_ids == (
        "obs-080-chunk-001",
    )

    assert entry.document_type == (
        "observation"
    )

    assert (
        manifest.records
        == original_records
    )


def test_unindexed_file_remains_unmatched(
    tmp_path,
) -> None:
    """A physical file without existing evidence must not be marked indexed."""

    documents_root = (
        tmp_path
        / "documents"
    )

    documents_root.mkdir()

    (
        documents_root
        / "NEW-FRD.md"
    ).write_text(
        "# REQ-900",
        encoding="utf-8",
    )

    manifest = (
        IngestionManifest.empty()
    )

    result = bootstrap_document_ledger(
        manifest=manifest,
        documents_root=documents_root,
    )

    assert result.passed is False
    assert result.unmatched_documents == 1
    assert manifest.documents == {}


def test_multiple_chunks_are_associated(
    tmp_path,
) -> None:
    """All existing chunk IDs should be attached to the source file."""

    documents_root = (
        tmp_path
        / "documents"
    )

    documents_root.mkdir()

    (
        documents_root
        / "FRD.md"
    ).write_text(
        "# REQ-001\n# REQ-002",
        encoding="utf-8",
    )

    manifest = (
        IngestionManifest.empty()
    )

    manifest.records.extend(
        [
            {
                "evidence_id": "FRD-001",
                "chunk_id": "chunk-001",
                "document_type": "frd",
                "document_name": "FRD.md",
            },
            {
                "evidence_id": "FRD-002",
                "chunk_id": "chunk-002",
                "document_type": "frd",
                "document_name": "FRD.md",
            },
        ]
    )

    result = bootstrap_document_ledger(
        manifest=manifest,
        documents_root=documents_root,
    )

    assert result.passed is True

    assert (
        manifest.documents[
            "FRD.md"
        ].chunk_ids
        == (
            "chunk-001",
            "chunk-002",
        )
    )


def test_existing_ledger_entry_is_preserved(
    tmp_path,
) -> None:
    """Bootstrap should not overwrite existing entries by default."""

    documents_root = (
        tmp_path
        / "documents"
    )

    documents_root.mkdir()

    (
        documents_root
        / "OBS.md"
    ).write_text(
        "# OBS-001",
        encoding="utf-8",
    )

    discovered = discover_documents(
        documents_root
    )[0]

    manifest = (
        IngestionManifest.empty()
    )

    from src.ingestion_models import (
        ManifestDocument,
    )

    existing = ManifestDocument(
        document_id=(
            discovered.document_id
        ),
        relative_path=(
            discovered.relative_path
        ),
        document_name=(
            discovered.document_name
        ),
        extension=(
            discovered.extension
        ),
        file_hash=(
            discovered.file_hash
        ),
        file_size=(
            discovered.file_size
        ),
        modified_time_ns=(
            discovered.modified_time_ns
        ),
        chunk_ids=(
            "existing-chunk",
        ),
    )

    manifest.upsert(
        existing
    )

    result = bootstrap_document_ledger(
        manifest=manifest,
        documents_root=documents_root,
    )

    assert (
        manifest.documents[
            "OBS.md"
        ]
        == existing
    )

    assert (
        result.ledger_entries_created
        == 0
    )


def test_records_are_never_modified(
    tmp_path,
) -> None:
    """Bootstrap must preserve engineering evidence records exactly."""

    documents_root = (
        tmp_path
        / "documents"
    )

    documents_root.mkdir()

    (
        documents_root
        / "test.txt"
    ).write_text(
        "TC-001 PASS",
        encoding="utf-8",
    )

    manifest = (
        IngestionManifest.empty()
    )

    record = {
        "evidence_id": "TEST-001",
        "document_type": "test",
        "document_name": "test.txt",
        "test_case_id": "TC-001",
        "result_status": "PASS",
    }

    manifest.records.append(
        dict(record)
    )

    bootstrap_document_ledger(
        manifest=manifest,
        documents_root=documents_root,
    )

    assert manifest.records == [
        record
    ]

def test_observation_file_name_is_not_source_candidate() -> None:
    """
    Observation file_name may reference affected source code.

    It must not cause the observation evidence record to map
    to that code file.
    """

    from src.manifest_bootstrap import (
        record_file_candidates,
    )

    record = {
        "evidence_id": (
            "OBS::Observation_Report_OBS-023.pdf"
            "::PAGE-1::OBS-023"
        ),

        "document_type": (
            "observation"
        ),

        "document_name": (
            "Observation_Report_OBS-023.pdf"
        ),

        "file_path": (
            "D:/project/documents/observations/"
            "Observation_Report_OBS-023.pdf"
        ),

        # Referenced implementation file.
        # This is NOT the source document.
        "file_name": (
            "data_processor.c"
        ),

        "observation_id": (
            "OBS-023"
        ),
    }

    candidates = (
        record_file_candidates(
            record
        )
    )

    assert (
        "observation_report_obs-023.pdf"
        in candidates
    )

    assert any(
        candidate.endswith(
            (
                "/observations/"
                "observation_report_obs-023.pdf"
            )
        )
        for candidate in candidates
    )

    assert (
        "data_processor.c"
        not in candidates
    )

def test_code_file_name_is_source_candidate() -> None:
    """
    Code evidence should use file_name as physical source provenance.
    """

    from src.manifest_bootstrap import (
        record_file_candidates,
    )

    record = {
        "evidence_id": (
            "CODE::data_processor.c"
            "::process_data"
        ),

        "document_type": (
            "code"
        ),

        "file_name": (
            "data_processor.c"
        ),

        "function_name": (
            "process_data"
        ),

        "start_line": 26,

        "end_line": 47,
    }

    candidates = (
        record_file_candidates(
            record
        )
    )

    assert (
        "data_processor.c"
        in candidates
    )
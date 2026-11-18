"""
Tests for document discovery and change detection.
"""

from __future__ import annotations

from src.document_discovery import (
    calculate_document_id,
    calculate_file_hash,
    compare_documents_with_manifest,
    discover_documents,
)
from src.ingestion_manifest import (
    IngestionManifest,
)
from src.ingestion_models import (
    DocumentChangeType,
    ManifestDocument,
)


def manifest_record(
    discovered,
) -> ManifestDocument:
    """Convert discovered file to an indexed manifest fixture."""

    return ManifestDocument(
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
            "chunk-001",
        ),
    )


def test_hash_is_content_based(
    tmp_path,
) -> None:
    """Equal file content should produce equal SHA-256 hashes."""

    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"

    first.write_text(
        "REQ-017",
        encoding="utf-8",
    )

    second.write_text(
        "REQ-017",
        encoding="utf-8",
    )

    assert (
        calculate_file_hash(first)
        == calculate_file_hash(second)
    )


def test_document_id_is_path_based() -> None:
    """Logical document ID should depend on relative path."""

    first = calculate_document_id(
        "frd/FRD-001.pdf"
    )

    second = calculate_document_id(
        "frd/FRD-001.pdf"
    )

    different = calculate_document_id(
        "archive/FRD-001.pdf"
    )

    assert first == second
    assert first != different


def test_supported_documents_are_discovered(
    tmp_path,
) -> None:
    """Supported files should be found recursively."""

    nested = tmp_path / "observations"

    nested.mkdir()

    (
        nested
        / "OBS-080.md"
    ).write_text(
        "# OBS-080",
        encoding="utf-8",
    )

    (
        tmp_path
        / "controller.c"
    ).write_text(
        "void control(void) {}",
        encoding="utf-8",
    )

    (
        tmp_path
        / "ignored.bin"
    ).write_bytes(
        b"binary"
    )

    discovered = discover_documents(
        tmp_path
    )

    paths = {
        item.relative_path
        for item in discovered
    }

    assert (
        "observations/OBS-080.md"
        in paths
    )

    assert (
        "controller.c"
        in paths
    )

    assert (
        "ignored.bin"
        not in paths
    )


def test_temporary_files_are_ignored(
    tmp_path,
) -> None:
    """Word lock files must not be indexed."""

    (
        tmp_path
        / "~$FRD.docx"
    ).write_bytes(
        b"temporary"
    )

    assert (
        discover_documents(
            tmp_path
        )
        == ()
    )


def test_new_document_detection(
    tmp_path,
) -> None:
    """File absent from manifest should be NEW."""

    (
        tmp_path
        / "FRD.md"
    ).write_text(
        "# REQ-017",
        encoding="utf-8",
    )

    discovered = discover_documents(
        tmp_path
    )

    changes = (
        compare_documents_with_manifest(
            discovered_documents=(
                discovered
            ),
            manifest=(
                IngestionManifest.empty()
            ),
        )
    )

    assert len(
        changes.new_documents
    ) == 1


def test_unchanged_document_detection(
    tmp_path,
) -> None:
    """Matching hash should classify as UNCHANGED."""

    path = tmp_path / "OBS.txt"

    path.write_text(
        "OBS-023",
        encoding="utf-8",
    )

    discovered = discover_documents(
        tmp_path
    )

    manifest = (
        IngestionManifest.empty()
    )

    manifest.upsert(
        manifest_record(
            discovered[0]
        )
    )

    changes = (
        compare_documents_with_manifest(
            discovered_documents=(
                discovered
            ),
            manifest=manifest,
        )
    )

    assert (
        changes.changes[0]
        .change_type
        == DocumentChangeType.UNCHANGED
    )


def test_changed_document_detection(
    tmp_path,
) -> None:
    """Modified content at the same path should be CHANGED."""

    path = tmp_path / "OBS.txt"

    path.write_text(
        "OBS-023 original",
        encoding="utf-8",
    )

    first_discovery = (
        discover_documents(
            tmp_path
        )
    )

    manifest = (
        IngestionManifest.empty()
    )

    manifest.upsert(
        manifest_record(
            first_discovery[0]
        )
    )

    path.write_text(
        "OBS-023 modified",
        encoding="utf-8",
    )

    second_discovery = (
        discover_documents(
            tmp_path
        )
    )

    changes = (
        compare_documents_with_manifest(
            discovered_documents=(
                second_discovery
            ),
            manifest=manifest,
        )
    )

    assert (
        changes.changes[0]
        .change_type
        == DocumentChangeType.CHANGED
    )


def test_deleted_document_detection(
    tmp_path,
) -> None:
    """Manifest file absent from disk should be DELETED."""

    path = tmp_path / "TC.txt"

    path.write_text(
        "TC-102 FAIL",
        encoding="utf-8",
    )

    discovered = discover_documents(
        tmp_path
    )

    manifest = (
        IngestionManifest.empty()
    )

    manifest.upsert(
        manifest_record(
            discovered[0]
        )
    )

    path.unlink()

    changes = (
        compare_documents_with_manifest(
            discovered_documents=(),
            manifest=manifest,
        )
    )

    assert len(
        changes.deleted_documents
    ) == 1
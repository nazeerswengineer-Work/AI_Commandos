"""
Tests for REQTRACE AI production ingestion manifest.
"""

from __future__ import annotations

import json

import pytest

from src.ingestion_manifest import (
    IngestionManifest,
    IngestionManifestError,
    load_ingestion_manifest,
    normalize_relative_path,
    save_ingestion_manifest,
)

from src.ingestion_models import (
    IngestionDocumentStatus,
    ManifestDocument,
)


VALID_HASH = (
    "a" * 64
)

VALID_DOCUMENT_ID = (
    "b" * 64
)


def make_record(
    relative_path: str = (
        "frd/FRD_001.pdf"
    ),
) -> ManifestDocument:

    return ManifestDocument(
        document_id=(
            VALID_DOCUMENT_ID
        ),

        relative_path=(
            relative_path
        ),

        document_name=(
            "FRD_001.pdf"
        ),

        extension=".pdf",

        file_hash=(
            VALID_HASH
        ),

        file_size=1234,

        modified_time_ns=5678,

        document_type="frd",

        document_category="FRD",

        chunk_ids=(
            "chunk-001",
            "chunk-002",
        ),

        status=(
            IngestionDocumentStatus
            .INDEXED
        ),

        indexed_at=(
            "2026-11-17T12:00:00Z"
        ),
    )


def test_missing_manifest_returns_empty(
    tmp_path,
) -> None:

    manifest = (
        load_ingestion_manifest(
            tmp_path
            / "manifest.json"
        )
    )

    assert (
        manifest.manifest_version
        == "1.1"
    )

    assert manifest.documents == {}
    assert manifest.records == []


def test_manifest_round_trip(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "manifest.json"
    )

    manifest = (
        IngestionManifest.empty()
    )

    manifest.upsert(
        make_record()
    )

    manifest.records.append(
        {
            "evidence_id": "FRD-001",
            "document_type": "frd",
        }
    )

    save_ingestion_manifest(
        manifest,
        path,
    )

    loaded = (
        load_ingestion_manifest(
            path
        )
    )

    assert (
        loaded.to_dict()
        == manifest.to_dict()
    )


def test_version_10_is_backward_compatible(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "legacy.json"
    )

    path.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",

                "generated_at_utc": (
                    "2026-11-01T00:00:00Z"
                ),

                "record_count": 1,

                "document_count": 1,

                "record_counts_by_type": {
                    "frd": 1,
                },

                "duplicate_content_groups": [],

                "records": [
                    {
                        "evidence_id": (
                            "FRD-001"
                        ),

                        "document_type": (
                            "frd"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = (
        load_ingestion_manifest(
            path
        )
    )

    assert (
        manifest.manifest_version
        == "1.1"
    )

    assert len(
        manifest.records
    ) == 1

    assert manifest.documents == {}


def test_saved_manifest_uses_version_11(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "manifest.json"
    )

    save_ingestion_manifest(
        IngestionManifest.empty(),
        path,
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "manifest_version"
        ]
        == "1.1"
    )

    assert (
        "schema_version"
        not in payload
    )


def test_path_normalization() -> None:

    assert (
        normalize_relative_path(
            r"observations\OBS-080.pdf"
        )
        == (
            "observations/OBS-080.pdf"
        )
    )


def test_dot_prefix_normalization() -> None:

    assert (
        normalize_relative_path(
            "./frd/FRD-125.pdf"
        )
        == "frd/FRD-125.pdf"
    )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../secret.pdf",
        "../../secret.pdf",
        "frd/../secret.pdf",
        "frd/../../secret.pdf",
        r"..\secret.pdf",
        r"frd\..\secret.pdf",
        "/absolute/document.pdf",
        "C:/documents/FRD.pdf",
        r"C:\documents\FRD.pdf",
    ],
)
def test_unsafe_paths_are_rejected(
    unsafe_path: str,
) -> None:

    with pytest.raises(
        IngestionManifestError
    ):

        normalize_relative_path(
            unsafe_path
        )


def test_invalid_manifest_version_is_rejected(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "manifest.json"
    )

    path.write_text(
        json.dumps(
            {
                "manifest_version": (
                    "99.0"
                ),

                "documents": {},

                "records": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        IngestionManifestError
    ):

        load_ingestion_manifest(
            path
        )


def test_accidental_manifest_inside_records_is_rejected(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "manifest.json"
    )

    path.write_text(
        json.dumps(
            {
                "manifest_version": "1.1",

                "documents": {},

                "records": [
                    {
                        "schema_version": "1.0",
                        "documents": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        IngestionManifestError
    ):

        load_ingestion_manifest(
            path
        )


def test_remove_sync_record() -> None:

    manifest = (
        IngestionManifest.empty()
    )

    record = (
        make_record()
    )

    manifest.upsert(
        record
    )

    removed = (
        manifest.remove(
            record.relative_path
        )
    )

    assert removed == record

    assert (
        record.relative_path
        not in manifest.documents
    )
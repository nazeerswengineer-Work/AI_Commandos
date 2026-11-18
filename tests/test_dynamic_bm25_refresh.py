"""
Tests for Phase 16H.6 dynamic BM25 retrieval.
"""

from __future__ import annotations

import json

from src.dynamic_bm25 import (
    DynamicBM25Index,
    load_dynamic_lexical_records,
)


def dynamic_record(
    *,
    evidence_id: str = "DYN::REQ-900",
    content: str = (
        "REQ-900 requires null-input rejection."
    ),
) -> dict:

    return {
        "evidence_id": evidence_id,
        "chunk_id": evidence_id,

        "document_id": "a" * 64,

        "document_name": (
            "FRD_REQ-900.md"
        ),

        "relative_path": (
            "dynamic_test/FRD_REQ-900.md"
        ),

        "document_type": "frd",
        "document_category": "FRD",

        "file_hash": "b" * 64,
        "content_hash": "c" * 64,

        "content": content,

        "chunk_index": 0,

        "requirement_id": "REQ-900",

        "requirement_ids": (
            "REQ-900"
        ),

        "test_case_ids": (
            "TC-900"
        ),

        "dynamic_ingestion": True,
    }


def write_manifest(
    path,
    records,
) -> None:

    path.write_text(
        json.dumps(
            {
                "manifest_version": (
                    "1.1"
                ),

                "generated_at_utc": None,

                "record_count": len(
                    records
                ),

                "document_count": 1,

                "record_counts_by_type": {
                    "frd": len(
                        records
                    ),
                },

                "duplicate_content_groups": [],

                "documents": {},

                "records": records,
            }
        ),
        encoding="utf-8",
    )


def test_dynamic_record_is_loaded(
    tmp_path,
) -> None:

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    write_manifest(
        manifest_path,
        [
            dynamic_record()
        ],
    )

    records = (
        load_dynamic_lexical_records(
            manifest_path
        )
    )

    assert len(records) == 1

    assert (
        records[0].evidence_id
        == "DYN::REQ-900"
    )


def test_baseline_record_is_not_duplicated(
    tmp_path,
) -> None:

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    write_manifest(
        manifest_path,
        [
            {
                "evidence_id": (
                    "FRD::BASELINE"
                ),

                "document_type": "frd",

                "content": (
                    "REQ-017 baseline evidence."
                ),

                "dynamic_ingestion": False,
            },

            dynamic_record(),
        ],
    )

    records = (
        load_dynamic_lexical_records(
            manifest_path
        )
    )

    assert (
        tuple(
            record.evidence_id
            for record in records
        )
        == (
            "DYN::REQ-900",
        )
    )


def test_dynamic_requirement_is_bm25_searchable(
    tmp_path,
) -> None:

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    write_manifest(
        manifest_path,
        [
            dynamic_record()
        ],
    )

    index = (
        DynamicBM25Index
        .from_manifest(
            manifest_path
        )
    )

    results = index.retrieve(
        "REQ-900",
        top_k=5,
    )
    # results = index.retrieve(
    #         "REQ-910",
    #         top_k=10,
    #     )

    assert results

    assert (
        results[0].evidence_id
        == "DYN::REQ-900"
    )

    assert (
        results[0].retrieval_method
        == "bm25_dynamic"
    )

    assert (
        results[0].metadata[
            "requirement_id"
        ]
        == "REQ-900"
    )


def test_test_reference_is_searchable_lexically(
    tmp_path,
) -> None:

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    write_manifest(
        manifest_path,
        [
            dynamic_record()
        ],
    )

    index = (
        DynamicBM25Index
        .from_manifest(
            manifest_path
        )
    )

    results = index.retrieve(
        "TC-900",
        top_k=5,
    )

    assert results

    assert (
        results[0].evidence_id
        == "DYN::REQ-900"
    )


def test_deleted_dynamic_record_disappears_after_refresh(
    tmp_path,
) -> None:

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    write_manifest(
        manifest_path,
        [
            dynamic_record()
        ],
    )

    before = (
        DynamicBM25Index
        .from_manifest(
            manifest_path
        )
    )

    assert before.retrieve(
        "REQ-900"
    )

    # Simulate successful Phase 16H deletion:
    # vectors removed and records[] updated.
    write_manifest(
        manifest_path,
        [],
    )

    after = (
        DynamicBM25Index
        .from_manifest(
            manifest_path
        )
    )

    assert (
        after.retrieve(
            "REQ-900"
        )
        == []
    )


def test_changed_content_is_visible_after_refresh(
    tmp_path,
) -> None:

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    write_manifest(
        manifest_path,
        [
            dynamic_record(
                content=(
                    "REQ-900 requires "
                    "null-input rejection."
                )
            )
        ],
    )

    before = (
        DynamicBM25Index
        .from_manifest(
            manifest_path
        )
    )

    assert not before.retrieve(
        "watchdog monitoring"
    )

    write_manifest(
        manifest_path,
        [
            dynamic_record(
                content=(
                    "REQ-900 requires watchdog "
                    "monitoring and null-input rejection."
                )
            )
        ],
    )

    after = (
        DynamicBM25Index
        .from_manifest(
            manifest_path
        )
    )

    results = after.retrieve(
        "watchdog monitoring"
    )

    assert results

    assert (
        results[0].evidence_id
        == "DYN::REQ-900"
    )
"""
Tests for Streamlit-facing ingestion integration.
"""

from __future__ import annotations

from src.streamlit_ingestion import (
    DocumentIndexStatus,
)


def test_synchronized_index_status() -> None:

    status = DocumentIndexStatus(
        discovered=28,
        indexed=28,
        new=0,
        changed=0,
        unchanged=28,
        deleted=0,
        failed=0,
        manifest_version="1.1",
    )

    assert status.new == 0
    assert status.changed == 0
    assert status.deleted == 0

    assert (
        status.discovered
        == status.indexed
    )


def test_pending_index_status() -> None:

    status = DocumentIndexStatus(
        discovered=29,
        indexed=28,
        new=1,
        changed=0,
        unchanged=28,
        deleted=0,
        failed=0,
        manifest_version="1.1",
    )

    pending = (
        status.new
        + status.changed
        + status.deleted
    )

    assert pending == 1


def test_changed_document_status() -> None:

    status = DocumentIndexStatus(
        discovered=28,
        indexed=28,
        new=0,
        changed=1,
        unchanged=27,
        deleted=0,
        failed=0,
        manifest_version="1.1",
    )

    assert status.changed == 1


def test_deleted_document_status() -> None:

    status = DocumentIndexStatus(
        discovered=27,
        indexed=28,
        new=0,
        changed=0,
        unchanged=27,
        deleted=1,
        failed=0,
        manifest_version="1.1",
    )

    assert status.deleted == 1
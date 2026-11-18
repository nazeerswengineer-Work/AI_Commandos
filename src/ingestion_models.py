"""
Data models for incremental REQTRACE AI document ingestion.

The Phase 16H ingestion architecture distinguishes:

1. Existing engineering evidence records
2. New document-level synchronization records

Document synchronization classifies source files as:

- NEW
- CHANGED
- UNCHANGED
- DELETED
- FAILED
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class IngestionModelError(
    ValueError
):
    """Raised when an ingestion model is invalid."""


class DocumentChangeType(
    str,
    Enum,
):
    """Filesystem change classification."""

    NEW = "NEW"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"
    DELETED = "DELETED"
    FAILED = "FAILED"


DocumentChangeType.__test__ = False


class IngestionDocumentStatus(
    str,
    Enum,
):
    """Persistent indexing status."""

    DISCOVERED = "DISCOVERED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    DELETED = "DELETED"


IngestionDocumentStatus.__test__ = False


@dataclass(frozen=True)
class DiscoveredDocument:
    """One supported file discovered under documents/."""

    document_id: str

    absolute_path: Path
    relative_path: str

    document_name: str
    extension: str

    file_hash: str
    file_size: int
    modified_time_ns: int

    def __post_init__(
        self,
    ) -> None:

        if not self.document_id.strip():
            raise IngestionModelError(
                "document_id must not be empty."
            )

        if len(self.document_id) != 64:
            raise IngestionModelError(
                "document_id must be SHA-256."
            )

        if not self.absolute_path.is_absolute():
            raise IngestionModelError(
                "absolute_path must be absolute."
            )

        if not self.relative_path.strip():
            raise IngestionModelError(
                "relative_path must not be empty."
            )

        if not self.document_name.strip():
            raise IngestionModelError(
                "document_name must not be empty."
            )

        if not self.extension.strip():
            raise IngestionModelError(
                "extension must not be empty."
            )

        if len(self.file_hash) != 64:
            raise IngestionModelError(
                "file_hash must be SHA-256."
            )

        if self.file_size < 0:
            raise IngestionModelError(
                "file_size must not be negative."
            )

        if self.modified_time_ns < 0:
            raise IngestionModelError(
                "modified_time_ns must not be negative."
            )


@dataclass(frozen=True)
class ManifestDocument:
    """
    Document-level synchronization record.

    file_hash represents the SHA-256 of the complete source file.

    This is different from the content_hash values stored on
    existing engineering evidence records.
    """

    document_id: str
    relative_path: str

    document_name: str
    extension: str

    file_hash: str
    file_size: int
    modified_time_ns: int

    document_type: str | None = None
    document_category: str | None = None

    chunk_ids: tuple[str, ...] = ()

    status: IngestionDocumentStatus = (
        IngestionDocumentStatus.INDEXED
    )

    indexed_at: str | None = None
    last_error: str | None = None

    def __post_init__(
        self,
    ) -> None:

        if len(self.document_id) != 64:
            raise IngestionModelError(
                "document_id must be SHA-256."
            )

        if not self.relative_path.strip():
            raise IngestionModelError(
                "relative_path must not be empty."
            )

        if not self.document_name.strip():
            raise IngestionModelError(
                "document_name must not be empty."
            )

        if len(self.file_hash) != 64:
            raise IngestionModelError(
                "file_hash must be SHA-256."
            )

        if self.file_size < 0:
            raise IngestionModelError(
                "file_size must not be negative."
            )

        if self.modified_time_ns < 0:
            raise IngestionModelError(
                "modified_time_ns must not be negative."
            )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "ManifestDocument":

        return cls(
            document_id=str(
                payload["document_id"]
            ),

            relative_path=str(
                payload["relative_path"]
            ),

            document_name=str(
                payload["document_name"]
            ),

            extension=str(
                payload.get(
                    "extension",
                    "",
                )
            ),

            file_hash=str(
                payload["file_hash"]
            ),

            file_size=int(
                payload.get(
                    "file_size",
                    0,
                )
            ),

            modified_time_ns=int(
                payload.get(
                    "modified_time_ns",
                    0,
                )
            ),

            document_type=(
                str(
                    payload[
                        "document_type"
                    ]
                )
                if payload.get(
                    "document_type"
                )
                else None
            ),

            document_category=(
                str(
                    payload[
                        "document_category"
                    ]
                )
                if payload.get(
                    "document_category"
                )
                else None
            ),

            chunk_ids=tuple(
                str(value)
                for value
                in payload.get(
                    "chunk_ids",
                    [],
                )
            ),

            status=(
                IngestionDocumentStatus(
                    payload.get(
                        "status",
                        (
                            IngestionDocumentStatus
                            .INDEXED
                            .value
                        ),
                    )
                )
            ),

            indexed_at=(
                str(
                    payload[
                        "indexed_at"
                    ]
                )
                if payload.get(
                    "indexed_at"
                )
                else None
            ),

            last_error=(
                str(
                    payload[
                        "last_error"
                    ]
                )
                if payload.get(
                    "last_error"
                )
                else None
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "document_id": (
                self.document_id
            ),

            "relative_path": (
                self.relative_path
            ),

            "document_name": (
                self.document_name
            ),

            "extension": (
                self.extension
            ),

            "file_hash": (
                self.file_hash
            ),

            "file_size": (
                self.file_size
            ),

            "modified_time_ns": (
                self.modified_time_ns
            ),

            "document_type": (
                self.document_type
            ),

            "document_category": (
                self.document_category
            ),

            "chunk_ids": list(
                self.chunk_ids
            ),

            "status": (
                self.status.value
            ),

            "indexed_at": (
                self.indexed_at
            ),

            "last_error": (
                self.last_error
            ),
        }


@dataclass(frozen=True)
class DocumentChange:
    """One filesystem/manifest comparison result."""

    change_type: DocumentChangeType
    relative_path: str

    discovered: (
        DiscoveredDocument
        | None
    ) = None

    manifest: (
        ManifestDocument
        | None
    ) = None

    reason: str = ""

    def __post_init__(
        self,
    ) -> None:

        if not self.relative_path.strip():
            raise IngestionModelError(
                "relative_path must not be empty."
            )

        if (
            self.change_type
            in {
                DocumentChangeType.NEW,
                DocumentChangeType.CHANGED,
                DocumentChangeType.UNCHANGED,
            }
            and self.discovered is None
        ):
            raise IngestionModelError(
                f"{self.change_type.value} "
                "requires discovered metadata."
            )

        if (
            self.change_type
            == DocumentChangeType.DELETED
            and self.manifest is None
        ):
            raise IngestionModelError(
                "DELETED requires a manifest record."
            )


@dataclass(frozen=True)
class DocumentChangeSet:
    """Complete synchronization comparison."""

    changes: tuple[
        DocumentChange,
        ...
    ] = ()

    def by_type(
        self,
        change_type: DocumentChangeType,
    ) -> tuple[
        DocumentChange,
        ...
    ]:

        return tuple(
            change
            for change
            in self.changes
            if (
                change.change_type
                == change_type
            )
        )

    @property
    def new_documents(
        self,
    ):

        return self.by_type(
            DocumentChangeType.NEW
        )

    @property
    def changed_documents(
        self,
    ):

        return self.by_type(
            DocumentChangeType.CHANGED
        )

    @property
    def unchanged_documents(
        self,
    ):

        return self.by_type(
            DocumentChangeType.UNCHANGED
        )

    @property
    def deleted_documents(
        self,
    ):

        return self.by_type(
            DocumentChangeType.DELETED
        )

    @property
    def failed_documents(
        self,
    ):

        return self.by_type(
            DocumentChangeType.FAILED
        )

    @property
    def discovered_count(
        self,
    ) -> int:

        return sum(
            1
            for change
            in self.changes
            if (
                change.change_type
                != DocumentChangeType.DELETED
            )
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "discovered": (
                self.discovered_count
            ),

            "new": len(
                self.new_documents
            ),

            "changed": len(
                self.changed_documents
            ),

            "unchanged": len(
                self.unchanged_documents
            ),

            "deleted": len(
                self.deleted_documents
            ),

            "failed": len(
                self.failed_documents
            ),

            "changes": [
                {
                    "relative_path": (
                        change.relative_path
                    ),
                    "change_type": (
                        change.change_type.value
                    ),
                    "reason": (
                        change.reason
                    ),
                }
                for change
                in self.changes
            ],
        }
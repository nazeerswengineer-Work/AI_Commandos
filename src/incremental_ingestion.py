"""
REQTRACE AI incremental ingestion transaction coordinator.

The coordinator processes:

- NEW documents
- CHANGED documents
- DELETED documents

UNCHANGED documents are skipped.

Transaction boundary:

1. Parse and chunk the source.
2. Generate all new embeddings.
3. Mutate and verify the existing Chroma collection.
4. Update records[] and documents{} in memory.
5. Save the manifest atomically.

The manifest is never marked INDEXED before vector verification passes.
"""

from __future__ import annotations

import copy
import shutil

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.document_discovery import (
    detect_document_changes,
)

from src.dynamic_document_parser import (
    parse_discovered_document,
)

from src.dynamic_vector_adapter import (
    DynamicVectorAdapter,
    DynamicVectorError,
)

from src.ingestion_manifest import (
    IngestionManifest,
    save_ingestion_manifest,
)

from src.ingestion_models import (
    DocumentChange,
    DocumentChangeType,
    IngestionDocumentStatus,
    ManifestDocument,
)

from src.ingestion_record_builder import (
    count_records_by_type,
    remove_records_for_chunk_ids,
    replace_document_records,
)

from src.intelligent_chunking import (
    ChunkingConfig,
    chunk_parsed_document,
)


class IncrementalIngestionError(
    RuntimeError
):
    """Raised when incremental synchronization cannot continue safely."""


def utc_now_iso() -> str:
    """Return current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


@dataclass(frozen=True)
class DocumentSyncResult:
    """Result for one document synchronization action."""

    relative_path: str
    change_type: str
    status: str

    document_type: str | None = None
    collection_name: str | None = None

    old_chunk_count: int = 0
    new_chunk_count: int = 0

    message: str = ""

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert result into JSON-safe data."""

        return {
            "relative_path": (
                self.relative_path
            ),

            "change_type": (
                self.change_type
            ),

            "status": self.status,

            "document_type": (
                self.document_type
            ),

            "collection_name": (
                self.collection_name
            ),

            "old_chunk_count": (
                self.old_chunk_count
            ),

            "new_chunk_count": (
                self.new_chunk_count
            ),

            "message": self.message,
        }


@dataclass(frozen=True)
class IncrementalSyncReport:
    """Complete incremental synchronization report."""

    dry_run: bool

    discovered: int
    new: int
    changed: int
    unchanged: int
    deleted: int

    indexed: int
    removed: int
    skipped: int
    failed: int

    manifest_saved: bool

    results: tuple[
        DocumentSyncResult,
        ...
    ] = field(
        default_factory=tuple
    )

    @property
    def passed(
        self,
    ) -> bool:
        """Return whether no document failed."""

        return self.failed == 0

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert report into JSON-safe data."""

        return {
            "passed": self.passed,
            "dry_run": self.dry_run,

            "discovered": (
                self.discovered
            ),

            "new": self.new,
            "changed": self.changed,

            "unchanged": (
                self.unchanged
            ),

            "deleted": self.deleted,

            "indexed": self.indexed,
            "removed": self.removed,
            "skipped": self.skipped,
            "failed": self.failed,

            "manifest_saved": (
                self.manifest_saved
            ),

            "results": [
                result.to_dict()
                for result in self.results
            ],
        }


def clone_manifest(
    manifest: IngestionManifest,
) -> IngestionManifest:
    """Create an isolated manifest working copy."""

    return copy.deepcopy(
        manifest
    )


def refresh_manifest_summaries(
    manifest: IngestionManifest,
) -> None:
    """Recalculate mutable production-manifest summaries."""

    manifest.manifest_version = "1.1"

    manifest.generated_at_utc = (
        utc_now_iso()
    )

    manifest.record_count = len(
        manifest.records
    )

    manifest.document_count = len(
        manifest.documents
    )

    manifest.record_counts_by_type = (
        count_records_by_type(
            manifest.records
        )
    )


def manifest_document_from_chunks(
    *,
    change: DocumentChange,
    chunks,
    indexed_at: str,
) -> ManifestDocument:
    """Build the persisted synchronization record."""

    discovered = change.discovered

    if discovered is None:
        raise IncrementalIngestionError(
            "Cannot create an indexed manifest record "
            "without discovered source metadata."
        )

    prepared_chunks = tuple(
        chunks
    )

    if not prepared_chunks:
        raise IncrementalIngestionError(
            "Cannot create an indexed manifest record "
            "without chunks."
        )

    document_types = {
        chunk.document_type
        for chunk in prepared_chunks
    }

    document_categories = {
        chunk.document_category
        for chunk in prepared_chunks
    }

    if len(document_types) != 1:
        raise IncrementalIngestionError(
            "Parsed chunks have inconsistent document types."
        )
    if len(document_categories) != 1:
        raise IncrementalIngestionError(
            "Parsed chunks have inconsistent categories."
        )

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

        document_type=next(
            iter(
                document_types
            )
        ),

        document_category=next(
            iter(
                document_categories
            )
        ),

        chunk_ids=tuple(
            chunk.chunk_id
            for chunk in prepared_chunks
        ),

        status=(
            IngestionDocumentStatus
            .INDEXED
        ),

        indexed_at=(
            indexed_at
        ),

        last_error=None,
    )


class IncrementalIngestionCoordinator:
    """Coordinate parsing, chunking, vectors, and manifest state."""

    def __init__(
        self,
        *,
        vector_adapter: DynamicVectorAdapter,
        chunking_config: (
            ChunkingConfig
            | None
        ) = None,
        continue_on_error: bool = False,
    ) -> None:
        """Create the coordinator."""

        self.vector_adapter = (
            vector_adapter
        )

        self.chunking_config = (
            chunking_config
            or ChunkingConfig()
        )

        self.continue_on_error = (
            continue_on_error
        )

    def _prepare_chunks(
        self,
        change: DocumentChange,
    ):
        """Parse and chunk one NEW or CHANGED source."""

        discovered = change.discovered

        if discovered is None:
            raise IncrementalIngestionError(
                f"{change.change_type.value} "
                "change has no discovered document."
            )

        parsed = parse_discovered_document(
            discovered
        )

        return chunk_parsed_document(
            parsed,
            config=self.chunking_config,
        )

    def synchronize(
        self,
        *,
        documents_root: str | Path,
        manifest: IngestionManifest,
        manifest_path: str | Path,
        apply: bool,
        backup_manifest: bool = True,
    ) -> IncrementalSyncReport:
        """
        Synchronize current source files with vectors and manifest.

        In dry-run mode parsing and chunking are performed, but no
        embedding, Chroma mutation, manifest mutation, or save occurs.
        """

        changes = detect_document_changes(
            documents_root=documents_root,
            manifest=manifest,
        )

        working_manifest = (
            clone_manifest(
                manifest
            )
        )

        results: list[
            DocumentSyncResult
        ] = []

        indexed = 0
        removed = 0
        skipped = 0
        failed = 0

        successful_vector_operations: list[
            tuple[
                str,
                str,
                tuple[str, ...],
            ]
        ] = []

        for change in changes.changes:
            if (
                change.change_type
                == DocumentChangeType.UNCHANGED
            ):
                skipped += 1

                results.append(
                    DocumentSyncResult(
                        relative_path=(
                            change.relative_path
                        ),
                        change_type="UNCHANGED",
                        status="SKIPPED",
                        message=(
                            "Complete-file SHA-256 "
                            "matches the manifest."
                        ),
                    )
                )

                continue

            if (
                change.change_type
                in {
                    DocumentChangeType.NEW,
                    DocumentChangeType.CHANGED,
                }
            ):
                try:
                    chunks = (
                        self._prepare_chunks(
                            change
                        )
                    )

                    previous = (
                        change.manifest
                    )

                    previous_chunk_ids = (
                        previous.chunk_ids
                        if previous is not None
                        else ()
                    )

                    previous_document_type = (
                        previous.document_type
                        if previous is not None
                        else None
                    )

                    document_type = (
                        chunks[0]
                        .document_type
                    )

                    if not apply:
                        results.append(
                            DocumentSyncResult(
                                relative_path=(
                                    change.relative_path
                                ),

                                change_type=(
                                    change
                                    .change_type
                                    .value
                                ),

                                status="DRY_RUN",

                                document_type=(
                                    document_type
                                ),

                                old_chunk_count=len(
                                    previous_chunk_ids
                                ),

                                new_chunk_count=len(
                                    chunks
                                ),

                                message=(
                                    "Document parsed and "
                                    "chunked successfully. "
                                    "No vectors were written."
                                ),
                            )
                        )

                        continue

                    vector_result = (
                        self.vector_adapter
                        .upsert_document(
                            chunks=chunks,

                            previous_chunk_ids=(
                                previous_chunk_ids
                            ),

                            previous_document_type=(
                                previous_document_type
                            ),
                        )
                    )

                    working_manifest.records = (
                        replace_document_records(
                            records=(
                                working_manifest
                                .records
                            ),

                            previous_chunk_ids=(
                                previous_chunk_ids
                            ),

                            new_chunks=chunks,
                        )
                    )

                    document_record = (
                        manifest_document_from_chunks(
                            change=change,
                            chunks=chunks,
                            indexed_at=(
                                utc_now_iso()
                            ),
                        )
                    )

                    working_manifest.upsert(
                        document_record
                    )

                    successful_vector_operations.append(
                        (
                            document_type,
                            vector_result.collection_name,
                            vector_result.inserted_ids,
                        )
                    )

                    indexed += 1

                    results.append(
                        DocumentSyncResult(
                            relative_path=(
                                change.relative_path
                            ),

                            change_type=(
                                change
                                .change_type
                                .value
                            ),

                            status="INDEXED",

                            document_type=(
                                document_type
                            ),

                            collection_name=(
                                vector_result
                                .collection_name
                            ),

                            old_chunk_count=len(
                                previous_chunk_ids
                            ),

                            new_chunk_count=len(
                                chunks
                            ),

                            message=(
                                "Vectors were written and "
                                "verified successfully."
                            ),
                        )
                    )

                except Exception as exc:
                    failed += 1

                    results.append(
                        DocumentSyncResult(
                            relative_path=(
                                change.relative_path
                            ),

                            change_type=(
                                change
                                .change_type
                                .value
                            ),

                            status="FAILED",

                            message=(
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        )
                    )

                    if not self.continue_on_error:
                        break

                continue

            if (
                change.change_type
                == DocumentChangeType.DELETED
            ):
                previous = change.manifest

                if previous is None:
                    failed += 1

                    results.append(
                        DocumentSyncResult(
                            relative_path=(
                                change.relative_path
                            ),

                            change_type="DELETED",
                            status="FAILED",

                            message=(
                                "Deleted change has no "
                                "manifest synchronization record."
                            ),
                        )
                    )

                    if not self.continue_on_error:
                        break

                    continue

                if not apply:
                    results.append(
                        DocumentSyncResult(
                            relative_path=(
                                change.relative_path
                            ),

                            change_type="DELETED",
                            status="DRY_RUN",

                            document_type=(
                                previous.document_type
                            ),

                            old_chunk_count=len(
                                previous.chunk_ids
                            ),

                            new_chunk_count=0,

                            message=(
                                "Document deletion detected. "
                                "No vectors were removed."
                            ),
                        )
                    )

                    continue

                try:
                    if not previous.document_type:
                        raise IncrementalIngestionError(
                            "Deleted document has no "
                            "document_type in the manifest."
                        )

                    deleted_ids = ( 
                        self.vector_adapter
                        .delete_document(
                            document_type=(
                                previous
                                .document_type
                            ),

                            chunk_ids=(
                                previous
                                .chunk_ids
                            ),
                        )
                    )

                    working_manifest.records = ( 
                        remove_records_for_chunk_ids(
                            records=(
                                working_manifest
                                .records
                            ),

                            chunk_ids=(
                                previous
                                .chunk_ids
                            ),
                        )
                    )

                    working_manifest.remove(
                        change.relative_path
                    )

                    removed += 1

                    results.append(
                        DocumentSyncResult(
                            relative_path=(
                                change.relative_path
                            ),

                            change_type="DELETED",
                            status="REMOVED",

                            document_type=(
                                previous.document_type
                            ),

                            old_chunk_count=len(
                                deleted_ids
                            ),

                            new_chunk_count=0,

                            message=(
                                "Vectors and manifest records "
                                "were removed."
                            ),
                        )
                    )

                except Exception as exc:
                    failed += 1

                    results.append(
                        DocumentSyncResult(
                            relative_path=(
                                change.relative_path
                            ),

                            change_type="DELETED",
                            status="FAILED",

                            document_type=( 
                                previous.document_type
                            ),

                            old_chunk_count=len(
                                previous.chunk_ids
                            ),

                            message=( 
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        )
                    )

                    if not self.continue_on_error:
                        break

        manifest_saved = False

        if apply and failed == 0:
            refresh_manifest_summaries(
                working_manifest
            )

            resolved_manifest_path = (
                Path(
                    manifest_path
                )
                .expanduser()
                .resolve()
            )

            if (
                backup_manifest
                and resolved_manifest_path.exists()
            ):
                backup_path = (
                    resolved_manifest_path
                    .with_suffix(
                        resolved_manifest_path
                        .suffix
                        + ".pre-sync.bak"
                    )
                )

                shutil.copy2(
                    resolved_manifest_path,
                    backup_path,
                )

            save_ingestion_manifest(
                working_manifest,
                resolved_manifest_path,
            )

            manifest_saved = True

            # Update the caller's in-memory object only after save.
            manifest.manifest_version = (
                working_manifest
                .manifest_version
            ) 

            manifest.generated_at_utc = (
                working_manifest
                .generated_at_utc
            )

            manifest.record_count = (
                working_manifest
                .record_count
            )

            manifest.document_count = (
                working_manifest
                .document_count
            )

            manifest.record_counts_by_type = (
                dict(
                    working_manifest
                    .record_counts_by_type
                )
            )

            manifest.duplicate_content_groups = (
                list(
                    working_manifest
                    .duplicate_content_groups
                )
            )

            manifest.records = [
                dict(record)
                for record
                in working_manifest.records
            ]

            manifest.documents = dict(
                working_manifest.documents
            )

        return IncrementalSyncReport(
            dry_run=not apply,

            discovered=(
                changes.discovered_count
            ),

            new=len(
                changes.new_documents
            ),

            changed=len(
                changes.changed_documents
            ),

            unchanged=len(
                changes.unchanged_documents
            ),

            deleted=len(
                changes.deleted_documents
            ),

            indexed=indexed,
            removed=removed,
            skipped=skipped,
            failed=failed,

            manifest_saved=(
                manifest_saved
            ),

            results=tuple(
                results
            ),
        )

"""
Bootstrap the Phase 16H document synchronization ledger.

The existing REQTRACE AI manifest already contains engineering evidence
records in records[].

This module builds documents{} from:

- physical files discovered under documents/
- existing evidence records
- complete-file SHA-256 hashes
- existing evidence/chunk identifiers

No parsing, embedding, or vector-database modification occurs here.

Safety:

- existing records[] are preserved;
- ambiguous file-to-record mappings are reported;
- missing source files are reported;
- the caller decides whether to save the result;
- document records are marked INDEXED only when they can be associated
  with existing evidence records.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.document_discovery import (
    discover_documents,
)
from src.ingestion_manifest import (
    IngestionManifest,
    normalize_relative_path,
)
from src.ingestion_models import (
    DiscoveredDocument,
    IngestionDocumentStatus,
    ManifestDocument,
)


class ManifestBootstrapError(
    RuntimeError
):
    """Raised when baseline-ledger bootstrap cannot proceed safely."""


@dataclass(frozen=True)
class BootstrapIssue:
    """One non-fatal bootstrap issue."""

    relative_path: str | None
    issue_type: str
    message: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "issue_type": self.issue_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Result of one baseline manifest bootstrap."""

    discovered_count: int
    ledger_entries_created: int
    matched_documents: int
    unmatched_documents: int
    unmatched_records: int
    ambiguous_documents: int

    issues: tuple[
        BootstrapIssue,
        ...
    ] = field(
        default_factory=tuple
    )

    @property
    def passed(
        self,
    ) -> bool:
        """
        Return whether every discovered document was mapped safely.

        Unmatched evidence records do not automatically fail bootstrap
        because one historical record may reference a source that is no
        longer distributed with the current package.
        """

        return bool(
            self.unmatched_documents == 0
            and self.ambiguous_documents == 0
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "discovered_count": self.discovered_count,
            "ledger_entries_created": (
                self.ledger_entries_created
            ),
            "matched_documents": (
                self.matched_documents
            ),
            "unmatched_documents": (
                self.unmatched_documents
            ),
            "unmatched_records": (
                self.unmatched_records
            ),
            "ambiguous_documents": (
                self.ambiguous_documents
            ),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def normalized_text(
    value: object,
) -> str:
    """Normalize a metadata value for comparison."""

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .replace("\\", "/")
        .lower()
    )


def record_file_candidates(
    record: dict[str, Any],
) -> tuple[str, ...]:
    """
    Return authoritative source-file references for one evidence record.

    Candidate fields depend on document type.

    Important:

    Observation records may contain file_name to reference affected
    source code. That file is engineering traceability metadata, not
    the physical source document that produced the observation record.

    Therefore file_name must not be treated as an observation-document
    source candidate.
    """

    document_type = (
        str(
            record.get(
                "document_type",
                "",
            )
        )
        .strip()
        .lower()
    )

    # ---------------------------------------------------------
    # Evidence records sourced from engineering documents.
    # ---------------------------------------------------------

    if document_type in {
        "frd",
        "observation",
        "test",
        "review",
    }:

        field_names = (
            "relative_path",
            "file_path",
            "source_path",
            "document_path",
            "document_name",
            "source_file",
        )

    # ---------------------------------------------------------
    # Code records.
    #
    # file_name is authoritative here because the evidence
    # itself originates from the source-code file.
    # ---------------------------------------------------------

    elif document_type == "code":

        field_names = (
            "relative_path",
            "file_path",
            "source_path",
            "document_path",
            "file_name",
            "document_name",
            "source_file",
        )

    # ---------------------------------------------------------
    # Unknown historical record type.
    #
    # Prefer document provenance. Keep file_name last for
    # backward compatibility.
    # ---------------------------------------------------------

    else:

        field_names = (
            "relative_path",
            "file_path",
            "source_path",
            "document_path",
            "document_name",
            "source_file",
            "file_name",
        )

    values: list[str] = []

    seen: set[str] = set()

    for field_name in field_names:

        raw_value = record.get(
            field_name
        )

        if raw_value is None:
            continue

        normalized = normalized_text(
            raw_value
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        values.append(
            normalized
        )

    return tuple(
        values
    )


def record_chunk_identifier(
    record: dict[str, Any],
) -> str | None:
    """
    Return the existing vector/evidence identifier for one record.

    chunk_id is preferred because it is most likely to correspond to the
    vector-store record. evidence_id is used as a safe fallback.
    """

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


def infer_document_type(
    records: tuple[
        dict[str, Any],
        ...
    ],
) -> str | None:
    """Return a single document type when records agree."""

    values = {
        str(
            record.get(
                "document_type",
                "",
            )
        ).strip()
        for record in records
        if str(
            record.get(
                "document_type",
                "",
            )
        ).strip()
    }

    if len(values) == 1:
        return next(
            iter(values)
        )

    return None


def infer_document_category(
    records: tuple[
        dict[str, Any],
        ...
    ],
) -> str | None:
    """Map existing document types to UI-friendly categories."""

    document_type = infer_document_type(
        records
    )

    if document_type is None:
        return None

    categories = {
        "frd": "FRD",
        "observation": "Observation Report",
        "code": "Source Code",
        "test": "Test Report",
        "review": "Engineering Review",
    }

    return categories.get(
        document_type.lower(),
        "Other",
    )


def candidate_matches_document(
    *,
    candidate: str,
    document: DiscoveredDocument,
) -> bool:
    """
    Check whether a record source reference matches a discovered file.

    Matching order:

    1. Normalized relative path
    2. Absolute path ending with the relative path
    3. Exact basename

    Basename matching is accepted only after ambiguity is checked by the
    caller.
    """

    relative_path = normalized_text(
        document.relative_path
    )

    document_name = normalized_text(
        document.document_name
    )

    if candidate == relative_path:
        return True

    if candidate.endswith(
        "/" + relative_path
    ):
        return True

    candidate_name = (
        candidate
        .rsplit(
            "/",
            1,
        )[-1]
    )

    return bool(
        candidate_name == document_name
    )


def build_record_matches(
    *,
    discovered_documents: tuple[
        DiscoveredDocument,
        ...
    ],
    evidence_records: list[
        dict[str, Any]
    ],
) -> tuple[
    dict[
        str,
        tuple[
            dict[str, Any],
            ...
        ],
    ],
    set[int],
    list[BootstrapIssue],
]:
    """
    Match existing evidence records to discovered source files.

    Returns:

    - records grouped by discovered relative path;
    - indexes of matched evidence records;
    - bootstrap issues.
    """

    records_by_document: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    matched_record_indexes: set[
        int
    ] = set()

    issues: list[
        BootstrapIssue
    ] = []

    discovered_by_name: dict[
        str,
        list[
            DiscoveredDocument
        ],
    ] = defaultdict(
        list
    )

    for document in discovered_documents:
        discovered_by_name[
            document.document_name.lower()
        ].append(
            document
        )
    for index, record in enumerate(
        evidence_records
    ):
        candidates = record_file_candidates(
            record
        )

        matching_paths: set[str] = set()

        for document in discovered_documents:
            if any(
                candidate_matches_document(
                    candidate=candidate,
                    document=document,
                )
                for candidate in candidates
            ):
                matching_paths.add(
                    document.relative_path
                )

        if len(
            matching_paths
        ) == 1:
            relative_path = next(
                iter(
                    matching_paths
                )
            )

            records_by_document[
                relative_path
            ].append(
                record
            )

            matched_record_indexes.add(
                index
            )

            continue

        if len(
            matching_paths
        ) > 1:
            issues.append(
                BootstrapIssue(
                    relative_path=None,
                    issue_type=(
                        "AMBIGUOUS_RECORD_SOURCE"
                    ),
                    message=(
                        "Evidence record maps to multiple "
                        "physical files: "
                        + ", ".join(
                            sorted(
                                matching_paths
                            )
                        )
                    ),
                )
            )

    return (
        {
            relative_path: tuple(
                records
            )
            for relative_path, records
            in records_by_document.items()
        },
        matched_record_indexes,
        issues,
    )


def bootstrap_document_ledger(
    *,
    manifest: IngestionManifest,
    documents_root: str | Path,
    overwrite_existing: bool = False,
) -> BootstrapResult:
    """
    Populate manifest.documents from the current source files.

    This function modifies the supplied IngestionManifest in memory.

    Existing documents{} entries are preserved unless
    overwrite_existing=True.

    No ChromaDB operation occurs.
    """

    discovered_documents = discover_documents(
        documents_root
    )

    (
        records_by_document,
        matched_record_indexes,
        issues,
    ) = build_record_matches(
        discovered_documents=(
            discovered_documents
        ),
        evidence_records=(
            manifest.records
        ),
    )

    created = 0
    matched_documents = 0
    unmatched_documents = 0
    ambiguous_documents = 0

    name_counts: dict[
        str,
        int,
    ] = defaultdict(
        int
    )

    for document in discovered_documents:
        name_counts[
            document.document_name.lower()
        ] += 1

    for document in discovered_documents:
        relative_path = (
            document.relative_path
        )

        if (
            relative_path
            in manifest.documents
            and not overwrite_existing
        ):
            continue

        matched_records = (
            records_by_document.get(
                relative_path,
                (),
            )
        )

        if not matched_records:
            unmatched_documents += 1

            issues.append(
                BootstrapIssue(
                    relative_path=(
                        relative_path
                    ),
                    issue_type=(
                        "NO_EXISTING_EVIDENCE_RECORD"
                    ),
                    message=(
                        "The physical file has no matching "
                        "existing evidence record. It remains "
                        "NEW and must be processed by the "
                        "incremental ingestion pipeline."
                    ),
                )
            )

            continue

        # Duplicate basenames are acceptable only when existing records
        # provided an unambiguous relative or absolute path match.
        if (
            name_counts[
                document.document_name.lower()
            ] > 1
        ):
            exact_path_match = any(
                normalized_text(
                    candidate
                )
                == normalized_text(
                    relative_path
                )
                or normalized_text(
                    candidate
                ).endswith(
                    "/"
                    + normalized_text(
                        relative_path
                    )
                )
                for record in matched_records
                for candidate
                in record_file_candidates(
                    record
                )
            )

            if not exact_path_match:
                ambiguous_documents += 1

                issues.append(
                    BootstrapIssue(
                        relative_path=(
                            relative_path
                        ),
                        issue_type=(
                        "AMBIGUOUS_DOCUMENT_NAME"
                        ),
                        message=(
                            "Multiple source files share this "
                            "basename and the existing records "
                            "do not preserve a unique path."
                        ),
                    )
                )

                continue

        chunk_ids = tuple(
            dict.fromkeys(
                chunk_id
                for chunk_id in (
                    record_chunk_identifier(
                        record
                    )
                    for record in matched_records
                )
                if chunk_id
            )
        )

        if not chunk_ids:
            issues.append(
                BootstrapIssue(
                    relative_path=(
                        relative_path
                    ),
                    issue_type=(
                        "MISSING_CHUNK_IDENTIFIERS"
                    ),
                    message=(
                        "Matching evidence records contain "
                        "neither chunk_id nor evidence_id."
                    ),
                )
            )

            unmatched_documents += 1
            continue

        manifest.upsert(
            ManifestDocument(
                document_id=(
                    document.document_id
                ),
                relative_path=(
                    relative_path
                ),
                document_name=(
                    document.document_name
                ),
                extension=(
                    document.extension
                ),
                file_hash=(
                    document.file_hash
                ),
                file_size=(
                    document.file_size
                ),
                modified_time_ns=(
                    document.modified_time_ns
                ),
                document_type=(
                    infer_document_type(
                        matched_records
                    )
                ),
                document_category=(
                    infer_document_category(
                        matched_records
                    )
                ),
                chunk_ids=(
                    chunk_ids
                ),
                status=(
                    IngestionDocumentStatus
                    .INDEXED
                ),
                indexed_at=(
                    manifest.generated_at_utc
                    or utc_now_iso()
                ),
                last_error=None,
            )
        )

        created += 1
        matched_documents += 1

    unmatched_records = (
        len(
            manifest.records
        )
        - len(
            matched_record_indexes
        )
    )

    if unmatched_records:
        issues.append(
            BootstrapIssue(
                relative_path=None,
                issue_type=(
                    "UNMATCHED_EVIDENCE_RECORDS"
                ),
                message=(
                    f"{unmatched_records} existing evidence "
                    "record(s) could not be associated with "
                    "a current source file."
                ),
            )
        )

    manifest.manifest_version = "1.1"

    return BootstrapResult(
        discovered_count=len(
            discovered_documents
        ),
        ledger_entries_created=(
            created
        ),
        matched_documents=(
            matched_documents
        ),
        unmatched_documents=(
            unmatched_documents
        ),
        unmatched_records=(
            unmatched_records
        ),
        ambiguous_documents=(
            ambiguous_documents
        ),
        issues=tuple(
            issues
        ),
    )

def unmatched_record_details(
    *,
    evidence_records: list[
        dict[str, Any]
    ],
    matched_record_indexes: set[int],
) -> tuple[
    dict[str, Any],
    ...
]:
    """Return source metadata for unmatched historical records."""

    output: list[
        dict[str, Any]
    ] = []

    for index, record in enumerate(
        evidence_records
    ):

        if index in matched_record_indexes:
            continue

        output.append(
            {
                "index": index,

                "evidence_id": (
                    record.get(
                        "evidence_id"
                    )
                ),

                "document_type": (
                    record.get(
                        "document_type"
                    )
                ),

                "document_name": (
                    record.get(
                        "document_name"
                    )
                ),

                "file_name": (
                    record.get(
                        "file_name"
                    )
                ),

                "file_path": (
                    record.get(
                        "file_path"
                    )
                ),

                "source_path": (
                    record.get(
                        "source_path"
                    )
                ),
            }
        )

    return tuple(
        output
    )
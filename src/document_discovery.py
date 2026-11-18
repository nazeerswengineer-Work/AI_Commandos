"""
REQTRACE AI document discovery and change detection.

This module:

- scans documents/
- ignores temporary files
- supports engineering document/source formats
- calculates complete-file SHA-256 hashes
- generates path-based logical document IDs
- compares source files against manifest documents{}
- classifies NEW / CHANGED / UNCHANGED / DELETED

No parsing, embedding, or Chroma modification occurs here.
"""

from __future__ import annotations

import hashlib
import logging

from pathlib import Path

from src.ingestion_manifest import (
    IngestionManifest,
    normalize_relative_path,
)

from src.ingestion_models import (
    DiscoveredDocument,
    DocumentChange,
    DocumentChangeSet,
    DocumentChangeType,
)


LOGGER = logging.getLogger(
    __name__
)


SUPPORTED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".py",
    }
)


IGNORED_FILE_NAMES = frozenset(
    {
        "thumbs.db",
        "desktop.ini",
        ".ds_store",
    }
)


class DocumentDiscoveryError(
    RuntimeError
):
    """Raised when document discovery cannot proceed."""


def is_temporary_file(
    path: Path,
) -> bool:
    """Identify files that should never enter ingestion."""

    name = (
        path.name
        .strip()
        .lower()
    )

    if name in (
        IGNORED_FILE_NAMES
    ):
        return True

    if name.startswith(
        "~$"
    ):
        return True

    if name.startswith(
        "."
    ):
        return True

    if (
        path.suffix.lower()
        in {
            ".tmp",
            ".temp",
            ".part",
            ".crdownload",
        }
    ):
        return True

    return False


def is_supported_document(
    path: Path,
) -> bool:
    """Return whether a path is ingestion-compatible."""

    return bool(
        path.is_file()
        and not is_temporary_file(
            path
        )
        and (
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


def calculate_file_hash(
    file_path: str | Path,
    *,
    block_size: int = (
        1024
        * 1024
    ),
) -> str:
    """Calculate complete-file SHA-256."""

    if block_size < 1:

        raise ValueError(
            "block_size must be positive."
        )

    path = (
        Path(
            file_path
        )
        .expanduser()
        .resolve()
    )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:

            block = stream.read(
                block_size
            )

            if not block:

                break

            digest.update(
                block
            )

    return digest.hexdigest()


def calculate_document_id(
    relative_path: str,
) -> str:
    """
    Create stable document identity from normalized relative path.

    Content changes do not change the logical document ID.
    """

    normalized = (
        normalize_relative_path(
            relative_path
        )
    )

    return (
        hashlib
        .sha256(
            normalized.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def discover_document(
    *,
    path: Path,
    documents_root: Path,
) -> DiscoveredDocument:
    """Inspect one source file."""

    resolved_path = (
        path.resolve()
    )

    resolved_root = (
        documents_root.resolve()
    )

    try:

        relative = (
            resolved_path
            .relative_to(
                resolved_root
            )
        )

    except ValueError as exc:

        raise DocumentDiscoveryError(
            "Document is outside "
            "the configured documents root: "
            f"{resolved_path}"
        ) from exc

    relative_path = (
        normalize_relative_path(
            relative
        )
    )

    stat = (
        resolved_path.stat()
    )

    return DiscoveredDocument(
        document_id=(
            calculate_document_id(
                relative_path
            )
        ),

        absolute_path=(
            resolved_path
        ),

        relative_path=(
            relative_path
        ),

        document_name=(
            resolved_path.name
        ),

        extension=(
            resolved_path
            .suffix
            .lower()
        ),

        file_hash=(
            calculate_file_hash(
                resolved_path
            )
        ),

        file_size=(
            stat.st_size
        ),

        modified_time_ns=(
            stat.st_mtime_ns
        ),
    )


def discover_documents(
    documents_root: str | Path,
) -> tuple[
    DiscoveredDocument,
    ...
]:
    """Recursively discover supported source documents."""

    root = (
        Path(
            documents_root
        )
        .expanduser()
        .resolve()
    )

    if not root.exists():

        raise DocumentDiscoveryError(
            "Documents directory "
            "does not exist: "
            f"{root}"
        )

    if not root.is_dir():

        raise DocumentDiscoveryError(
            "Configured documents path "
            "is not a directory: "
            f"{root}"
        )

    candidates = sorted(
        (
            path
            for path
            in root.rglob(
                "*"
            )
            if is_supported_document(
                path
            )
        ),

        key=lambda path: (
            str(
                path
            ).lower()
        ),
    )

    discovered: list[
        DiscoveredDocument
    ] = []

    for path in candidates:

        try:

            discovered.append(
                discover_document(
                    path=path,
                    documents_root=root,
                )
            )

        except OSError as exc:

            # Do not log document content.
            LOGGER.error(
                (
                    "Unable to inspect "
                    "document=%s error_type=%s"
                ),
                path.name,
                type(exc).__name__,
            )

    return tuple(
        discovered
    )


def compare_documents_with_manifest(
    *,
    discovered_documents: tuple[
        DiscoveredDocument,
        ...
    ],
    manifest: IngestionManifest,
) -> DocumentChangeSet:
    """Compare current files against manifest documents{}."""

    discovered_by_path = {
        document.relative_path: (
            document
        )
        for document
        in discovered_documents
    }

    changes: list[
        DocumentChange
    ] = []

    # ---------------------------------------------------------
    # NEW / CHANGED / UNCHANGED
    # ---------------------------------------------------------

    for relative_path in sorted(
        discovered_by_path
    ):

        discovered = (
            discovered_by_path[
                relative_path
            ]
        )

        existing = (
            manifest.get(
                relative_path
            )
        )

        if existing is None:

            changes.append(
                DocumentChange(
                    change_type=(
                        DocumentChangeType.NEW
                    ),

                    relative_path=(
                        relative_path
                    ),

                    discovered=(
                        discovered
                    ),

                    manifest=None,

                    reason=(
                        "Document has no "
                        "synchronization record."
                    ),
                )
            )

            continue

        if (
            existing.document_id
            != discovered.document_id
        ):

            changes.append(
                DocumentChange(
                    change_type=(
                        DocumentChangeType.CHANGED
                    ),

                    relative_path=(
                        relative_path
                    ),

                    discovered=(
                        discovered
                    ),

                    manifest=(
                        existing
                    ),

                    reason=(
                        "Logical document ID "
                        "differs from manifest."
                    ),
                )
            )

            continue

        if (
            existing.file_hash
            != discovered.file_hash
        ):

            changes.append(
                DocumentChange(
                    change_type=(
                        DocumentChangeType.CHANGED
                    ),

                    relative_path=(
                        relative_path
                    ),

                    discovered=(
                        discovered
                    ),

                    manifest=(
                        existing
                    ),

                    reason=(
                        "Complete-file SHA-256 "
                        "changed."
                    ),
                )
            )

            continue

        changes.append(
            DocumentChange(
                change_type=(
                    DocumentChangeType.UNCHANGED
                ),

                relative_path=(
                    relative_path
                ),

                discovered=(
                    discovered
                ),

                manifest=(
                    existing
                ),

                reason=(
                    "Complete-file SHA-256 "
                    "matches manifest."
                ),
            )
        )

    # ---------------------------------------------------------
    # DELETED
    # ---------------------------------------------------------

    discovered_paths = set(
        discovered_by_path
    )

    manifest_paths = set(
        manifest.documents
    )

    for relative_path in sorted(
        manifest_paths
        - discovered_paths
    ):

        changes.append(
            DocumentChange(
                change_type=(
                    DocumentChangeType.DELETED
                ),

                relative_path=(
                    relative_path
                ),

                discovered=None,

                manifest=(
                    manifest.documents[
                        relative_path
                    ]
                ),

                reason=(
                    "Manifest document no longer "
                    "exists under documents/."
                ),
            )
        )

    return DocumentChangeSet(
        changes=tuple(
            sorted(
                changes,
                key=lambda item: (
                    item.relative_path.lower()
                ),
            )
        )
    )


def detect_document_changes(
    *,
    documents_root: str | Path,
    manifest: IngestionManifest,
) -> DocumentChangeSet:
    """Run discovery plus change detection."""

    discovered = (
        discover_documents(
            documents_root
        )
    )

    return (
        compare_documents_with_manifest(
            discovered_documents=(
                discovered
            ),
            manifest=(
                manifest
            ),
        )
    )
"""
REQTRACE AI combined engineering evidence and document-sync manifest.

Existing production manifest:

    manifest_version
    generated_at_utc
    record_count
    document_count
    record_counts_by_type
    duplicate_content_groups
    records[]

Phase 16H extends the same manifest with:

    documents{}

The records[] collection remains authoritative engineering evidence
metadata.

The documents{} collection manages source-file synchronization.

The two hash types have different purposes:

    documents[*].file_hash
        SHA-256 of the complete source file

    records[*].content_hash
        SHA-256 of one evidence/chunk content block
"""

from __future__ import annotations

import json
import os
import re

from dataclasses import (
    dataclass,
    field,
)

from pathlib import Path
from typing import Any

from src.ingestion_models import (
    ManifestDocument,
)


class IngestionManifestError(
    RuntimeError
):
    """Raised for ingestion-manifest failures."""


MANIFEST_VERSION = "1.1"

SUPPORTED_MANIFEST_VERSIONS = (
    "1.0",
    "1.1",
)


def normalize_relative_path(
    value: str | Path,
) -> str:
    """
    Normalize a portable manifest-relative path.

    Accepted:

        observations/OBS-080.pdf
        observations\\OBS-080.pdf
        ./frd/FRD-125.pdf

    Rejected:

        ../secret.pdf
        frd/../../secret.pdf
        C:\\documents\\FRD.pdf
        /absolute/document.pdf
    """

    text = (
        str(value)
        .strip()
    )

    if not text:
        raise IngestionManifestError(
            "Relative path must not be empty."
        )

    normalized = text.replace(
        "\\",
        "/",
    )

    # POSIX/UNC-like absolute path.
    if normalized.startswith(
        "/"
    ):
        raise IngestionManifestError(
            "Manifest paths must be relative."
        )

    # Windows drive path.
    if re.match(
        r"^[A-Za-z]:/",
        normalized,
    ):
        raise IngestionManifestError(
            "Manifest paths must be relative."
        )

    # SECURITY:
    # Check traversal before modifying "./".
    components = tuple(
        component
        for component
        in normalized.split(
            "/"
        )
        if component not in {
            "",
            ".",
        }
    )

    if ".." in components:
        raise IngestionManifestError(
            "Manifest paths must not contain '..'."
        )

    while normalized.startswith(
        "./"
    ):
        normalized = normalized[
            2:
        ]

    while "//" in normalized:
        normalized = normalized.replace(
            "//",
            "/",
        )

    path = Path(
        normalized
    )

    if path.is_absolute():
        raise IngestionManifestError(
            "Manifest paths must be relative."
        )

    if ".." in path.parts:
        raise IngestionManifestError(
            "Manifest paths must not contain '..'."
        )

    normalized_parts = tuple(
        part
        for part
        in path.parts
        if part not in {
            "",
            ".",
        }
    )

    if not normalized_parts:
        raise IngestionManifestError(
            "Relative path must contain "
            "a file or directory name."
        )

    return "/".join(
        normalized_parts
    )


@dataclass
class IngestionManifest:
    """
    Combined REQTRACE AI manifest.

    records
        Existing evidence-level metadata.

    documents
        Phase 16H file synchronization metadata.
    """

    manifest_version: str = (
        MANIFEST_VERSION
    )

    generated_at_utc: (
        str
        | None
    ) = None

    record_count: int = 0

    document_count: int = 0

    record_counts_by_type: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    duplicate_content_groups: list[
        Any
    ] = field(
        default_factory=list
    )

    documents: dict[
        str,
        ManifestDocument,
    ] = field(
        default_factory=dict
    )

    records: list[
        dict[
            str,
            Any,
        ]
    ] = field(
        default_factory=list
    )

    @classmethod
    def empty(
        cls,
    ) -> "IngestionManifest":
        """Create a new empty version-1.1 manifest."""

        return cls(
            manifest_version=(
                MANIFEST_VERSION
            ),
            generated_at_utc=None,
            record_count=0,
            document_count=0,
            record_counts_by_type={},
            duplicate_content_groups=[],
            documents={},
            records=[],
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "IngestionManifest":
        """Load a version 1.0 or version 1.1 production manifest."""

        manifest_version = (
            str(
                payload.get(
                    "manifest_version",
                    "",
                )
            )
            .strip()
        )

        # Compatibility with the short-lived prototype.
        if not manifest_version:

            manifest_version = (
                str(
                    payload.get(
                        "schema_version",
                        "",
                    )
                )
                .strip()
            )

        if (
            manifest_version
            not in (
                SUPPORTED_MANIFEST_VERSIONS
            )
        ):

            raise IngestionManifestError(
                "Unsupported ingestion manifest "
                f"version: {manifest_version!r}. "
                "Supported versions: "
                + ", ".join(
                    SUPPORTED_MANIFEST_VERSIONS
                )
            )

        # ---------------------------------------------------------
        # Evidence records
        # ---------------------------------------------------------

        raw_records = payload.get(
            "records",
            [],
        )

        if raw_records is None:

            raw_records = []

        if not isinstance(
            raw_records,
            list,
        ):

            raise IngestionManifestError(
                "Manifest 'records' must "
                "be an array."
            )

        records: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for (
            index,
            raw_record,
        ) in enumerate(
            raw_records
        ):

            if not isinstance(
                raw_record,
                dict,
            ):

                raise IngestionManifestError(
                    "Evidence record must "
                    "be an object at index "
                    f"{index}."
                )

            # Detect the accidental prototype object
            # that was previously inserted inside records[].
            if (
                "schema_version"
                in raw_record
                and "documents"
                in raw_record
                and "evidence_id"
                not in raw_record
            ):

                raise IngestionManifestError(
                    "A manifest object was found "
                    "inside records[] at index "
                    f"{index}."
                )

            records.append(
                dict(
                    raw_record
                )
            )

        # ---------------------------------------------------------
        # Document synchronization records
        # ---------------------------------------------------------

        raw_documents = (
            payload.get(
                "documents",
                {},
            )
        )

        if raw_documents is None:

            raw_documents = {}

        if not isinstance(
            raw_documents,
            dict,
        ):

            raise IngestionManifestError(
                "Manifest 'documents' "
                "must be an object."
            )

        documents: dict[
            str,
            ManifestDocument,
        ] = {}

        for (
            raw_relative_path,
            raw_document,
        ) in raw_documents.items():

            if not isinstance(
                raw_document,
                dict,
            ):

                raise IngestionManifestError(
                    "Manifest document record "
                    "must be an object: "
                    f"{raw_relative_path}"
                )

            relative_path = (
                normalize_relative_path(
                    raw_relative_path
                )
            )

            document = (
                ManifestDocument
                .from_dict(
                    raw_document
                )
            )

            document_relative_path = (
                normalize_relative_path(
                    document.relative_path
                )
            )

            if (
                document_relative_path
                != relative_path
            ):

                raise IngestionManifestError(
                    "Manifest document key does "
                    "not match document relative_path: "
                    f"{raw_relative_path}"
                )

            documents[
                relative_path
            ] = document

        # ---------------------------------------------------------
        # Metadata summaries
        # ---------------------------------------------------------

        raw_type_counts = (
            payload.get(
                "record_counts_by_type",
                {},
            )
        )

        if not isinstance(
            raw_type_counts,
            dict,
        ):

            raise IngestionManifestError(
                "record_counts_by_type "
                "must be an object."
            )

        record_counts_by_type = {
            str(key): int(value)
            for (
                key,
                value,
            )
            in raw_type_counts.items()
        }

        duplicate_groups = (
            payload.get(
                "duplicate_content_groups",
                [],
            )
        )

        if not isinstance(
            duplicate_groups,
            list,
        ):

            raise IngestionManifestError(
                "duplicate_content_groups "
                "must be an array."
            )

        # ---------------------------------------------------------
        # Build current in-memory representation.
        # ---------------------------------------------------------

        return cls(
            # Version 1.0 is normalized to 1.1 in memory.
            manifest_version=(
                MANIFEST_VERSION
            ),

            generated_at_utc=(
                str(
                    payload[
                        "generated_at_utc"
                    ]
                )
                if payload.get(
                    "generated_at_utc"
                )
                else None
            ),

            record_count=len(
                records
            ),

            document_count=int(
                payload.get(
                    "document_count",
                    0,
                )
            ),

            record_counts_by_type=(
                record_counts_by_type
            ),

            duplicate_content_groups=list(
                duplicate_groups
            ),

            documents=(
                documents
            ),

            records=(
                records
            ),
        )

    def get(
        self,
        relative_path: str,
    ) -> (
        ManifestDocument
        | None
    ):
        """Return one document-sync record."""

        normalized = (
            normalize_relative_path(
                relative_path
            )
        )

        return self.documents.get(
            normalized
        )

    def upsert(
        self,
        document: ManifestDocument,
    ) -> None:
        """Add or update one document-sync record."""

        normalized = (
            normalize_relative_path(
                document.relative_path
            )
        )

        if (
            normalized
            != document.relative_path
        ):

            raise IngestionManifestError(
                "ManifestDocument relative_path "
                "must already be normalized."
            )

        self.documents[
            normalized
        ] = document

    def remove(
        self,
        relative_path: str,
    ) -> (
        ManifestDocument
        | None
    ):
        """Remove one document-sync record."""

        return self.documents.pop(
            normalize_relative_path(
                relative_path
            ),
            None,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the production manifest to JSON-safe data."""

        return {
            "manifest_version": (
                MANIFEST_VERSION
            ),

            "generated_at_utc": (
                self.generated_at_utc
            ),

            "record_count": len(
                self.records
            ),

            "document_count": len(
                self.documents
            ),

            "record_counts_by_type": dict(
                self.record_counts_by_type
            ),

            "duplicate_content_groups": list(
                self.duplicate_content_groups
            ),

            "documents": {
                relative_path: (
                    self.documents[
                        relative_path
                    ].to_dict()
                )
                for relative_path
                in sorted(
                    self.documents
                )
            },

            "records": [
                dict(
                    record
                )
                for record
                in self.records
            ],
        }


def load_ingestion_manifest(
    manifest_path: str | Path,
) -> IngestionManifest:
    """
    Load the ingestion manifest.

    A missing manifest creates a new empty manifest.
    """

    path = (
        Path(
            manifest_path
        )
        .expanduser()
        .resolve()
    )

    if not path.exists():

        return (
            IngestionManifest
            .empty()
        )

    try:

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:

        raise IngestionManifestError(
            "Ingestion manifest contains "
            "invalid JSON: "
            f"{path}"
        ) from exc

    except OSError as exc:

        raise IngestionManifestErroror(
            "Unable to read ingestion manifest: "
            f"{path}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):

        raise IngestionManifestError(
            "Ingestion manifest root "
            "must be an object."
        )

    return (
        IngestionManifest
        .from_dict(
            payload
        )
    )


def save_ingestion_manifest(
    manifest: IngestionManifest,
    manifest_path: str | Path,
) -> None:
    """
    Atomically save the combined oduction manifest.
    """

    path = (
        Path(
            manifest_path
        )
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    serialized = json.dumps(
        manifest.to_dict(),
        indent=2,
        ensure_ascii=False,
    )

    try:

        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:

            stream.write(
                serialized
            )

            stream.write(
                "\n"
            )

            stream.flush()

            os.fsync(
                stream.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

    except OSError as exc:

        try:

            temporary_path.unlink(
                missing_ok=True
            )

        except OSError:

            pass

        raise IngestionManifestError(
            "Unable to save ingestion manifest: "
            f"{path}"
        ) from exc
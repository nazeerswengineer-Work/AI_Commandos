"""
REQTRACE AI Phase 16H document-change diagnostic.

This command performs only:

    filesystem discovery
        ↓
    full-file SHA-256
        ↓
    manifest comparison
        ↓
    NEW / CHANGED / UNCHANGED / DELETED

No ChromaDB records are changed.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from src.document_discovery import (
    detect_document_changes,
)

from src.ingestion_manifest import (
    load_ingestion_manifest,
)


def parse_arguments() -> (
    argparse.Namespace
):
    """Parse diagnostic arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI incremental "
            "document change inspection."
        )
    )

    parser.add_argument(
        "--documents",
        default="documents",
        help=(
            "Documents directory relative "
            "to the project root."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/ingestion_manifest.json"
        ),
        help=(
            "Ingestion manifest relative "
            "to the project root."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Inspect source-document changes."""

    args = parse_arguments()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    documents_root = (
        base_directory
        / args.documents
    )

    manifest_path = (
        base_directory
        / args.manifest
    )

    manifest = (
        load_ingestion_manifest(
            manifest_path
        )
    )

    changes = (
        detect_document_changes(
            documents_root=(
                documents_root
            ),
            manifest=(
                manifest
            ),
        )
    )

    print()

    print("=" * 78)

    print(
        "REQTRACE AI - DOCUMENT CHANGE INSPECTION"
    )

    print("=" * 78)

    print(
        f"Manifest version : "
        f"{manifest.manifest_version}"
    )

    print(
        f"Evidence records : "
        f"{len(manifest.records)}"
    )

    print(
        f"Sync documents   : "
        f"{len(manifest.documents)}"
    )

    print()

    print(
        json.dumps(
            changes.to_dict(),
            indent=2,
            ensure_ascii=True,
        )
    )

    print()

    print(
        "This command is read-only. "
        "No embeddings or ChromaDB records "
        "were modified."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
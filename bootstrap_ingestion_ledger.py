"""
Bootstrap REQTRACE AI's Phase 16H document synchronization ledger.

Default behavior is dry-run.

Use --apply to update ingestion_manifest.json.

This command does not:

- parse source documents;
- create embeddings;
- modify ChromaDB;
- alter existing records[] evidence metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil

from datetime import datetime, timezone
from pathlib import Path

from src.ingestion_manifest import (
    load_ingestion_manifest,
    save_ingestion_manifest,
)
from src.manifest_bootstrap import (
    bootstrap_document_ledger,
)


def parse_arguments() -> argparse.Namespace:
    """Parse bootstrap arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap REQTRACE AI document-sync "
            "metadata from the existing indexed corpus."
        )
    )

    parser.add_argument(
        "--documents",
        default="documents",
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/ingestion_manifest.json"
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the bootstrapped documents{} "
            "ledger to the manifest."
        ),
    )

    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Rebuild existing documents{} entries."
        ),
    )

    return parser.parse_args()


def timestamp_for_filename() -> str:
    """Return a Windows-safe timestamp."""

    return (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )


def main() -> int:
    """Bootstrap the baseline synchronization ledger."""

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

    manifest = load_ingestion_manifest(
        manifest_path
    )

    original_record_count = len(
        manifest.records
    )

    result = bootstrap_document_ledger(
        manifest=manifest,
        documents_root=documents_root,
        overwrite_existing=(
            args.overwrite_existing
        ),
    )

    if (
        len(
            manifest.records
        )
        != original_record_count
    ):
        raise RuntimeError(
            "Bootstrap unexpectedly changed "
            "the existing evidence records."
        )

    print()

    print("=" * 78)

    print(
        "REQTRACE AI - INGESTION LEDGER BOOTSTRAP"
    )

    print("=" * 78)

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=True,
        )
    )

    if result.issues:

        print()
        print("BOOTSTRAP ISSUES")
        print("-" * 78)

        for issue in result.issues:

            path_text = (
                issue.relative_path
                if issue.relative_path
                else "<manifest records>"
            )

            print(
                f"[{issue.issue_type}] "
                f"{path_text}"
            )

            print(
                f"  {issue.message}"
            )

    if not args.apply:
        print()

        print(
            "DRY RUN ONLY"
        )

        print(
            "No manifest changes were written."
        )

        print()

        print(
            "Use --apply after reviewing "
            "the bootstrap result."
        )

        return (
            0
            if result.passed
            else 1
        )

    if not result.passed:
        print()

        print(
            "Manifest was not updated because "
            "one or more source documents could "
            "not be mapped safely."
        )

        return 1

    backup_path = (
        manifest_path.parent
        / (
            manifest_path.name
            + "."
            + timestamp_for_filename()
            + ".bak"
        )
    )

    shutil.copy2(
        manifest_path,
        backup_path,
    )

    save_ingestion_manifest(
        manifest,
        manifest_path,
    )

    print()

    print(
        f"Backup written : {backup_path}"
    )

    print(
        f"Manifest saved : {manifest_path}"
    )

    print(
        f"Sync documents : "
        f"{len(manifest.documents)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
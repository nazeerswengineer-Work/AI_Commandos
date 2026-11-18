"""
Inspect unmatched document-to-evidence mappings.
"""

from __future__ import annotations

from pathlib import Path

from src.document_discovery import (
    discover_documents,
)

from src.ingestion_manifest import (
    load_ingestion_manifest,
)

from src.manifest_bootstrap import (
    build_record_matches,
    record_file_candidates,
)


def main() -> int:

    base = (
        Path(__file__)
        .resolve()
        .parent
    )

    documents = discover_documents(
        base / "documents"
    )

    manifest = load_ingestion_manifest(
        base
        / "data"
        / "ingestion_manifest.json"
    )

    (
        records_by_document,
        matched_indexes,
        issues,
    ) = build_record_matches(
        discovered_documents=documents,
        evidence_records=manifest.records,
    )

    print()
    print("UNMATCHED PHYSICAL DOCUMENTS")
    print("=" * 78)

    for document in documents:

        if (
            document.relative_path
            in records_by_document
        ):
            continue

        print(
            document.relative_path
        )

    print()
    print("UNMATCHED MANIFEST RECORDS")
    print("=" * 78)

    for index, record in enumerate(
        manifest.records
    ):

        if index in matched_indexes:
            continue

        print(
            f"Index         : {index}"
        )

        print(
            f"Evidence ID   : "
            f"{record.get('evidence_id')}"
        )

        print(
            f"Document type : "
            f"{record.get('document_type')}"
        )

        print(
            f"Document name : "
            f"{record.get('document_name')}"
        )

        print(
            f"File name     : "
            f"{record.get('file_name')}"
        )

        print(
            f"File path     : "
            f"{record.get('file_path')}"
        )

        print(
            "Candidates    : "
            f"{record_file_candidates(record)}"
        )

        print("-" * 78)

    if issues:

        print()
        print("MAPPING ISSUES")
        print("=" * 78)

        for issue in issues:

            print(
                f"{issue.issue_type}: "
                f"{issue.message}"
            )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
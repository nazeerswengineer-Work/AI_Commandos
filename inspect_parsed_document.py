"""
Inspect dynamic parsing and intelligent chunking without embedding.

Usage:

    python inspect_parsed_document.py documents/frd/New_FRD.pdf
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from src.document_discovery import (
    discover_document,
)

from src.dynamic_document_parser import (
    parse_discovered_document,
)

from src.intelligent_chunking import (
    ChunkingConfig,
    chunk_parsed_document,
)


def parse_arguments() -> argparse.Namespace:
    """Parse diagnostic arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect REQTRACE AI dynamic parsing and chunking."
        )
    )

    parser.add_argument(
        "file",
        help=(
            "Source file path relative to "
            "the project directory."
        ),
    )

    parser.add_argument(
        "--documents-root",
        default="documents",
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=4000,
    )

    return parser.parse_args()


def main() -> int:
    """Parse and inspect one source file."""

    args = parse_arguments()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    documents_root = (
        base_directory
        / args.documents_root
    ).resolve()

    file_path = (
        base_directory
        / args.file
    ).resolve()

    document = discover_document(
        path=file_path,
        documents_root=documents_root,
    )

    parsed = parse_discovered_document(
        document
    )

    chunks = chunk_parsed_document(
        parsed,
        config=ChunkingConfig(
            maximum_characters=(
                args.max_characters
            ),
        ),
    )

    payload = {
        "document": {
            "document_id": (
                parsed.document_id
            ),
            "document_name": (
                parsed.document_name
            ),
            "relative_path": (
                parsed.relative_path
            ),
            "document_type": (
                parsed.document_type
            ),
            "document_category": (
                parsed.document_category
            ),
            "parsing_method": (
                parsed.parsing_method
            ),
            "section_count": len(
                parsed.sections
            ),
            "chunk_count": len(
                chunks
            ),
        },
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": (
                    chunk.chunk_index
                ),
                "section_name": (
                    chunk.section_name
                ),
                "page_number": (
                    chunk.page_number
                ),
                "line_number_start": (
                    chunk.line_number_start
                ),
                "line_number_end": (
                    chunk.line_number_end
                ),
                "function_name": (
                    chunk.function_name
                ),
                "class_name": (
                    chunk.class_name
                ),
                "requirement_ids": list(
                    chunk.requirement_ids
                ),
                "observation_ids": list(
                    chunk.observation_ids
                ),
                "test_case_ids": list(
                    chunk.test_case_ids
                ),
                "result_status": (
                    chunk.result_status
                ),
                "content_characters": len(
                    chunk.content
                ),
                "preview": (
                    chunk.content[:300]
                ),
            }
            for chunk in chunks
        ],
    }

    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
        )
    )

    print()
    print(
        "No embeddings or vector records were created."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
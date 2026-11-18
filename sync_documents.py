"""
REQTRACE AI incremental document synchronization.

Default behavior is dry-run.

Use:

    python sync_documents.py --apply

to embed NEW/CHANGED documents, remove DELETED document vectors, and
save ingestion_manifest.json.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import chromadb

from config import (
    get_settings,
)

from src.dynamic_vector_adapter import (
    DynamicVectorAdapter,
)

from src.embeddings import (
    EmbeddingClientConfig,
    OllamaEmbeddingClient,
)

from src.incremental_ingestion import (
    IncrementalIngestionCoordinator,
)

from src.ingestion_lock import (
    IngestionFileLock,
)

from src.ingestion_manifest import (
    load_ingestion_manifest,
)

from src.intelligent_chunking import (
    ChunkingConfig,
)

from src.vector_store import (
    CollectionNames,
)


def parse_arguments() -> argparse.Namespace:
    """Parse synchronization arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Synchronize REQTRACE AI source documents "
            "with the existing persistent ChromaDB index."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply ChromaDB and manifest changes. "
            "Without this option the command is read-only."
        ),
    )

    parser.add_argument(
        "--documents",
        default=None,
        help=(
            "Optional documents directory override."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/ingestion_manifest.json"
        ),
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--overlap-characters",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    """Run incremental synchronization."""

    args = parse_arguments()

    project_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    settings = get_settings()

    documents_root = (
        Path(
            args.documents
        ).expanduser().resolve()
        if args.documents
        else settings.documents_path
    )

    manifest_path = (
        project_root
        / args.manifest
    ).resolve()

    lock_path = (
        manifest_path.parent
        / "incremental_ingestion.lock"
    )

    embedding_config = (
        EmbeddingClientConfig(
            base_url=(
                settings
                .ollama_base_url
            ),
            model=(
                settings
                .ollama_embed_model
            ),
        )
    )

    collection_names = (
        CollectionNames(
            frd=(
                settings
                .frd_collection_name
            ),

            observation=(
                settings
                .observation_collection_name
            ),

            code=(
                settings
                .code_collection_name
            ),

            test=(
                settings
                .test_collection_name
            ),

            review=(
                settings
                .reviews_collection_name
            ),
        )
    )

    with IngestionFileLock(
        lock_path,
        timeout_seconds=10.0,
        stale_after_seconds=3600.0,
    ):
        manifest = (
            load_ingestion_manifest(
                manifest_path
            )
        )

        chroma_client = (
            chromadb.PersistentClient(
                path=str(
                    settings
                    .chroma_db_path
                )
            )
        )

        with OllamaEmbeddingClient(
            embedding_config
        ) as embedding_client:
            vector_adapter = (
                DynamicVectorAdapter(
                    chroma_client=(
                        chroma_client
                    ),

                    collection_names=(
                        collection_names
                    ),

                    embedding_client=(
                        embedding_client
                    ),
                )
            )

            coordinator = (
                IncrementalIngestionCoordinator(
                    vector_adapter=(
                        vector_adapter
                    ),

                    chunking_config=(
                        ChunkingConfig(
                            maximum_characters=(
                                args
                                .max_characters
                            ),

                            overlap_characters=(
                                args
                                .overlap_characters
                            ),
                        )
                    ),

                    continue_on_error=(
                        args
                        .continue_on_error
                    ),
                )
            )

            report = (
                coordinator.synchronize(
                    documents_root=(
                        documents_root
                    ),

                    manifest=manifest,

                    manifest_path=(
                        manifest_path
                    ),

                    apply=args.apply,
                )
            )

    print()

    print("=" * 78)

    print(
        "REQTRACE AI - INCREMENTAL DOCUMENT SYNCHRONIZATION"
    )

    print("=" * 78)

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=True,
        )
    )

    if not args.apply:
        print()
        print(
            "DRY RUN ONLY"
        )

        print(
            "No embeddings, ChromaDB records, "
            "or manifest data were changed."
        )

    return (
        0
        if report.passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
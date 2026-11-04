"""
REQTRACE AI Phase 9 indexing command.

Builds the validated Phase 8 corpus, generates Ollama embeddings, and
synchronizes the persistent ChromaDB collections.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import get_settings
from src.corpus import build_project_corpus
from src.embeddings import (
    EmbeddingClientConfig,
    OllamaEmbeddingClient,
)
from src.logging_config import configure_logging
from src.vector_store import (
    ChromaEvidenceStore,
    CollectionNames,
    VectorStoreError,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index the validated REQTRACE AI corpus "
            "into persistent ChromaDB collections."
        )
    )

    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help=(
            "Do not remove indexed records that are "
            "absent from the current corpus."
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify existing collections without "
            "generating embeddings or updating data."
        ),
    )

    return parser.parse_args()


def print_report(report: object) -> None:
    print()
    print("=" * 78)
    print("PHASE 9 INDEXING REPORT")
    print("=" * 78)

    for evidence_type, stats in (
        report.collections.items()
    ):
        print()
        print(f"[{evidence_type.upper()}]")
        print(
            f"  Collection          : "
            f"{stats.collection_name}"
        )
        print(
            f"  Corpus records      : "
            f"{stats.corpus_records}"
        )
        print(
            f"  Existing records    : "
            f"{stats.existing_records}"
        )
        print(
            f"  Inserted or updated : "
            f"{stats.inserted_or_updated}"
        )
        print(
            f"  Unchanged           : "
            f"{stats.unchanged}"
        )
        print(
            f"  Stale deleted       : "
            f"{stats.stale_deleted}"
        )
        print(
            f"  Embedding batches   : "
            f"{stats.batches_embedded}"
        )
        print(
            f"  Embedding dimension : "
            f"{stats.embedding_dimension}"
        )
        print(
            f"  Final count         : "
            f"{stats.final_collection_count}"
        )

    print()
    print("-" * 78)
    print(
        f"Total corpus records : "
        f"{report.total_corpus_records}"
    )
    print(
        f"Total upserted       : "
        f"{report.total_upserted}"
    )
    print(
        f"Total unchanged      : "
        f"{report.total_unchanged}"
    )
    print(
        f"Total stale deleted  : "
        f"{report.total_stale_deleted}"
    )


def main() -> int:
    args = parse_arguments()
    settings = get_settings()

    logger = configure_logging(
        log_level=settings.log_level,
        log_file_path=settings.log_file_path,
    )

    base_directory = Path(__file__).resolve().parent

    print("=" * 78)
    print("REQTRACE AI - PHASE 9 PERSISTENT INDEXING")
    print("=" * 78)
    print(
        "Only validated Phase 8 evidence records "
        "will be indexed."
    )
    print(
        "No LLM generation, assessment, retrieval, "
        "or closure decision will occur."
    )

    corpus = build_project_corpus(
        base_directory
    )

    print()
    print(f"Corpus records : {corpus.record_count}")
    print(
        f"Embedding model: "
        f"{settings.ollama_embed_model}"
    )
    print(
        f"ChromaDB path  : "
        f"{settings.chroma_db_path}"
    )
    print(
        f"Batch size     : "
        f"{settings.embedding_batch_size}"
    )

    embedding_config = EmbeddingClientConfig(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        connect_timeout_seconds=(
            settings.ollama_connect_timeout_seconds
        ),
        read_timeout_seconds=(
            settings.llm_timeout_seconds
        ),
        max_retries=2,
        retry_delay_seconds=1.0,
        truncate=True,
        keep_alive="5m",
    )

    collection_names = CollectionNames(
        frd=settings.frd_collection_name,
        observation=(
            settings.observation_collection_name
        ),
        code=settings.code_collection_name,
        test=settings.test_collection_name,
        review=settings.reviews_collection_name,
    )

    try:
        with OllamaEmbeddingClient(
            embedding_config
        ) as embedding_client:
            store = ChromaEvidenceStore(
                database_path=(
                    settings.chroma_db_path
                ),
                collection_names=(
                    collection_names
                ),
                embedding_client=(
                    embedding_client
                ),
                batch_size=(
                    settings.embedding_batch_size
                ),
            )

            if args.verify_only:
                verification = store.verify_corpus(
                    corpus
                )

                print()
                print("Verification results:")

                for evidence_type, passed in (
                    verification.items()
                ):
                    status = (
                        "PASS"
                        if passed
                        else "FAIL"
                    )

                    print(
                        f"  {evidence_type:<12}: "
                        f"{status}"
                    )

                return (
                    0
                    if all(verification.values())
                    else 1
                )

            report = store.index_corpus(
                corpus,
                delete_stale=(
                    not args.keep_stale
                ),
            )

            print_report(report)

            verification = store.verify_corpus(
                corpus
            )

            print()
            print("Post-index verification:")

            for evidence_type, passed in (
                verification.items()
            ):
                status = (
                    "PASS"
                    if passed
                    else "FAIL"
                )

                print(
                    f"  {evidence_type:<12}: "
                    f"{status}"
                )

            if not all(
                verification.values()
            ):
                logger.error(
                    "Phase 9 corpus verification failed."
                )
                return 1

    except VectorStoreError as exc:
        logger.exception(
            "Phase 9 indexing failed."
        )
        print()
        print(f"INDEXING FAILED: {exc}")
        return 1

    except Exception as exc:
        logger.exception(
            "Unexpected Phase 9 error."
        )
        print()
        print(f"INDEXING FAILED: {exc}")
        return 1

    print()
    print("Phase 9 indexing completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
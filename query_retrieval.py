"""
REQTRACE AI Phase 10 retrieval CLI.

Retrieval diagnostics only.
No LLM answer generation occurs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from config import get_settings
from src.embeddings import (
    EmbeddingClientConfig,
    OllamaEmbeddingClient,
)
from src.retrieval import (
    EvidenceRetriever,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact-ID-first retrieval "
            "against indexed REQTRACE AI evidence."
        )
    )

    parser.add_argument(
        "query",
        help="Engineering retrieval query.",
    )

    parser.add_argument(
        "--exact-only",
        action="store_true",
        help=(
            "Disable semantic retrieval "
            "and use explicit IDs only."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    settings = get_settings()

    embedding_config = (
        EmbeddingClientConfig(
            base_url=(
                settings.ollama_base_url
            ),
            model=(
                settings.ollama_embed_model
            ),
            connect_timeout_seconds=(
                settings
                .ollama_connect_timeout_seconds
            ),
            read_timeout_seconds=(
                settings.llm_timeout_seconds
            ),
        )
    )

    client = chromadb.PersistentClient(
        path=str(
            Path(
                settings.chroma_db_path
            ).resolve()
        )
    )

    collection_names = {
        "frd": (
            settings.frd_collection_name
        ),
        "observation": (
            settings
            .observation_collection_name
        ),
        "code": (
            settings.code_collection_name
        ),
        "test": (
            settings.test_collection_name
        ),
        "review": (
            settings.reviews_collection_name
        ),
    }

    with OllamaEmbeddingClient(
        embedding_config
    ) as embedding_client:
        retriever = EvidenceRetriever(
            client=client,
            collection_names=(
                collection_names
            ),
            embedding_client=(
                embedding_client
            ),
            vector_top_k=(
                settings.vector_top_k
            ),
            final_context_limit=(
                settings.final_context_limit
            ),
            min_relevance_score=(
                settings.min_relevance_score
            ),
        )

        result = retriever.retrieve(
            args.query,
            include_semantic=(
                not args.exact_only
            ),
        )

    print()
    print("=" * 78)
    print("REQTRACE AI - PHASE 10 RETRIEVAL")
    print("=" * 78)

    print(
        f"Query              : "
        f"{result.query}"
    )

    print(
        f"Requirement IDs    : "
        f"{result.identifiers.requirement_ids}"
    )

    print(
        f"Observation IDs    : "
        f"{result.identifiers.observation_ids}"
    )

    print(
        f"Test IDs           : "
        f"{result.identifiers.test_case_ids}"
    )

    print(
        f"Exact matches      : "
        f"{result.exact_match_count}"
    )

    print(
        f"Semantic candidates: "
        f"{result.semantic_match_count}"
    )

    print()
    print("Ranked Evidence")
    print("-" * 78)

    if not result.evidence:
        print("No evidence retrieved.")
        return 0

    for item in result.evidence:
        metadata = item.metadata

        print()
        print(
            f"[{item.rank}] "
            f"{item.evidence_id}"
        )
        print(
            f"    Type   : "
            f"{item.document_type}"
        )
        print(
            f"    Method : "
            f"{item.retrieval_method}"
        )
        print(
            f"    Score  : "
            f"{item.score:.4f}"
        )

        if metadata.get(
            "requirement_id"
        ):
            print(
                "    REQ    : "
                f"{metadata['requirement_id']}"
            )

        if metadata.get(
            "observation_id"
        ):
            print(
                "    OBS    : "
                f"{metadata['observation_id']}"
            )

        if metadata.get(
            "test_case_id"
        ):
            print(
                "    TEST   : "
                f"{metadata['test_case_id']}"
            )

        if metadata.get(
            "page_number"
        ):
            print(
                "    Page   : "
                f"{metadata['page_number']}"
            )

        if metadata.get(
            "start_line"
        ):
            print(
                "    Lines  : "
                f"{metadata['start_line']}"
                "-"
                f"{metadata.get('end_line')}"
            )

        preview = " ".join(
            item.content.split()
        )

        print(
            "    Preview: "
            f"{preview[:220]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
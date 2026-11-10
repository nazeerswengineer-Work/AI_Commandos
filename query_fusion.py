"""
REQTRACE AI Phase 11B RRF retrieval diagnostic.

Combines:
- Phase 11A BM25
- Phase 10 dense semantic retrieval

No traceability expansion.
No evidence-type diversification.
No LLM generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from config import get_settings

from src.bm25_retrieval import (
    BM25EvidenceIndex,
)
from src.corpus import (
    build_project_corpus,
)
from src.embeddings import (
    EmbeddingClientConfig,
    OllamaEmbeddingClient,
)
from src.fusion import (
    normalize_fused_results,
    reciprocal_rank_fusion,
)
from src.retrieval import (
    EvidenceRetriever,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 11B "
            "BM25 + semantic RRF retrieval."
        )
    )

    parser.add_argument(
        "query",
        help=(
            "Engineering retrieval query."
        ),
    )

    parser.add_argument(
        "--bm25-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--vector-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fused-k",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    settings = get_settings()

    base_directory = (
        Path(__file__)
        .resolve()
        .parent
    )

    corpus = build_project_corpus(
        base_directory
    )

    bm25_index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    bm25_k = (
        args.bm25_k
        if args.bm25_k is not None
        else settings.bm25_top_k
    )

    vector_k = (
        args.vector_k
        if args.vector_k is not None
        else settings.vector_top_k
    )

    fused_k = (
        args.fused_k
        if args.fused_k is not None
        else settings.fused_top_k
    )

    bm25_results = (
        bm25_index.retrieve(
            args.query,
            top_k=bm25_k,
        )
    )

    chroma_client = (
        chromadb.PersistentClient(
            path=str(
                settings.chroma_db_path
            )
        )
    )

    collection_names = {
        "frd": (
            settings
            .frd_collection_name
        ),
        "observation": (
            settings
            .observation_collection_name
        ),
        "code": (
            settings
            .code_collection_name
        ),
        "test": (
            settings
            .test_collection_name
        ),
        "review": (
            settings
            .reviews_collection_name
        ),
    }

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
            connect_timeout_seconds=(
                settings
                .ollama_connect_timeout_seconds
            ),
            read_timeout_seconds=(
                settings
                .llm_timeout_seconds
            ),
        )
    )

    with OllamaEmbeddingClient(
        embedding_config
    ) as embedding_client:

        dense_retriever = (
            EvidenceRetriever(
                client=(
                    chroma_client
                ),
                collection_names=(
                    collection_names
                ),
                embedding_client=(
                    embedding_client
                ),
                vector_top_k=(
                    vector_k
                ),
                final_context_limit=(
                    vector_k
                ),
                min_relevance_score=(
                    settings
                    .min_relevance_score
                ),
            )
        )

        semantic_results = (
            dense_retriever
            .semantic_retrieval(
                args.query
            )
        )

    ranked_lists = {
        "bm25": (
            bm25_results
        ),
        "semantic": (
            semantic_results
        ),
    }

    fused = reciprocal_rank_fusion(
        ranked_lists,
        rank_constant=60,
    )

    results = (
        normalize_fused_results(
            fused,
            top_k=fused_k,
        )
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 11B RRF FUSION"
    )
    print("=" * 78)

    print(
        f"Query      : {args.query}"
    )

    print(
        f"BM25       : "
        f"{len(bm25_results)}"
    )

    print(
        f"Semantic   : "
        f"{len(semantic_results)}"
    )

    print(
        f"Fused Top-K: "
        f"{len(results)}"
    )

    print()

    for item in results:

        metadata = item.metadata

        print(
            f"[{item.rank}] "
            f"{item.evidence_id}"
        )

        print(
            f"    Type   : "
            f"{item.document_type}"
        )

        print(
            f"    Score  : "
            f"{item.score:.4f}"
        )

        print(
            f"    Votes  : "
            f"{metadata.get('fusion_contributions')}"
        )

        if metadata.get(
            "requirement_id"
        ):
            print(
                f"    REQ    : "
                f"{metadata['requirement_id']}"
            )

        if metadata.get(
            "observation_id"
        ):
            print(
                f"    OBS    : "
                f"{metadata['observation_id']}"
            )

        if metadata.get(
            "test_case_id"
        ):
            print(
                f"    TEST   : "
                f"{metadata['test_case_id']}"
            )

        if metadata.get(
            "function_name"
        ):
            print(
                f"    Func   : "
                f"{metadata['function_name']}"
            )

        preview = " ".join(
            item.content.split()
        )

        print(
            f"    Preview: "
            f"{preview[:220]}"
        )

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
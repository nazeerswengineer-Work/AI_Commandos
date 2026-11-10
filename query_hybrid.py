"""
REQTRACE AI Phase 11 hybrid retrieval CLI.

No answer generation or engineering assessment occurs.
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
from src.hybrid_retrieval import (
    HybridEvidenceRetriever,
)
from src.retrieval import EvidenceRetriever


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run REQTRACE AI Phase 11 "
            "hybrid evidence retrieval."
        )
    )

    parser.add_argument(
        "query",
        help="Engineering retrieval query.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    settings = get_settings()

    base_directory = (
        Path(__file__).resolve().parent
    )

    corpus = build_project_corpus(
        base_directory
    )

    bm25_index = (
        BM25EvidenceIndex.from_corpus(
            corpus
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

    with OllamaEmbeddingClient(
        embedding_config
    ) as embedding_client:
        dense_retriever = EvidenceRetriever(
            client=chroma_client,
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
                settings.fused_top_k
            ),
            min_relevance_score=(
                settings.min_relevance_score
            ),
        )

        hybrid = HybridEvidenceRetriever(
            corpus=corpus,
            dense_retriever=(
                dense_retriever
            ),
            bm25_index=bm25_index,
            bm25_top_k=(
                settings.bm25_top_k
            ),
            fused_top_k=(
                settings.fused_top_k
            ),
            final_context_limit=(
                settings.final_context_limit
            ),
            rrf_rank_constant=60,
            traceability_limit=20,
            diversify_types=True,
        )

        result = hybrid.retrieve(
            args.query
        )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 11 HYBRID RETRIEVAL"
    )
    print("=" * 78)

    print(f"Query          : {result.query}")
    print(
        f"Exact results  : "
        f"{len(result.exact_results)}"
    )
    print(
        f"BM25 results   : "
        f"{len(result.bm25_results)}"
    )
    print(
        f"Semantic       : "
        f"{len(result.semantic_results)}"
    )
    print(
        f"Traceability   : "
        f"{len(result.traceability_results)}"
    )
    print(
        f"Final context  : "
        f"{len(result.evidence)}"
    )

    print()
    print("Final Ranked Evidence")
    print("-" * 78)

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

        for label, field_name in (
            ("REQ", "requirement_id"),
            ("OBS", "observation_id"),
            ("TEST", "test_case_id"),
        ):
            if metadata.get(field_name):
                print(
                    f"    {label:<6}: "
                    f"{metadata[field_name]}"
                )

        preview = " ".join(
            item.content.split()
        )

        print(
            f"    Preview: {preview[:240]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
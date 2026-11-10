"""
REQTRACE AI Phase 11D final-context diagnostic.

Pipeline:

Exact-ID retrieval
BM25 retrieval
Dense retrieval
RRF fusion
Controlled traceability expansion
Final evidence-context assembly

No LLM generation occurs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from config import get_settings
from src.bm25_retrieval import (
    BM25EvidenceIndex,
)
from src.context_builder import (
    build_final_context,
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
    analyze_query_ids,
)
from src.traceability import (
    expand_engineering_family,
)


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 11D "
            "final evidence context."
        )
    )

    parser.add_argument(
        "query",
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

    bm25_results = (
        bm25_index.retrieve(
            args.query,
            top_k=(
                settings.bm25_top_k
            ),
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

    identifiers = analyze_query_ids(
        args.query
    )

    with OllamaEmbeddingClient(
        embedding_config
    ) as embedding_client:

        dense = EvidenceRetriever(
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
                settings
                .min_relevance_score
            ),
        )

        exact_results = (
            dense.exact_id_retrieval(
                identifiers
            )
        )

        semantic_results = (
            dense.semantic_retrieval(
                args.query
            )
        )

    fused = reciprocal_rank_fusion(
        {
            "bm25": bm25_results,
            "semantic": semantic_results,
        },
        rank_constant=60,
    )

    rrf_results = (
        normalize_fused_results(
            fused,
            top_k=(
                settings.fused_top_k
            ),
        )
    )

    if exact_results:

        anchors = list(
            exact_results
        )

    elif rrf_results:

        anchors = [
            rrf_results[0]
        ]

    else:

        anchors = []

    expansion = expand_engineering_family(
        corpus,
        anchors,
        maximum_iterations=3,
        maximum_results=20,
    )

    final_context = (
        build_final_context(
            exact_results=list(
                exact_results
            ),
            traceability_results=list(
                expansion.expanded
            ),
            rrf_results=list(
                rrf_results
            ),
            context_limit=(
                settings
                .final_context_limit
            ),
        )
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 11D FINAL CONTEXT"
    )
    print("=" * 78)

    print(
        f"Query        : {args.query}"
    )

    print(
        f"Exact        : "
        f"{final_context.exact_count}"
    )

    print(
        f"Traceability : "
        f"{final_context.traceability_count}"
    )

    print(
        f"RRF          : "
        f"{final_context.rrf_count}"
    )

    print(
        f"Final records: "
        f"{len(final_context.evidence)}"
    )

    print()
    print("Final Evidence Context")
    print("-" * 78)

    for item in final_context.evidence:

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
            "document_revision"
        ):
            print(
                f"    Rev    : "
                f"{metadata['document_revision']}"
            )

        if metadata.get(
            "function_name"
        ):
            print(
                f"    Func   : "
                f"{metadata['function_name']}"
            )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
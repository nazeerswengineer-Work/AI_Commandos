"""
Reusable Phase 11 retrieval pipeline.

Combines validated retrieval components but performs no LLM generation.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from config import Settings

from src.bm25_retrieval import BM25EvidenceIndex
from src.context_builder import (
    FinalEvidenceContext,
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


def retrieve_final_context(
    query: str,
    settings: Settings,
    base_directory: str | Path,
) -> FinalEvidenceContext:
    """Run the validated Phase 11 retrieval pipeline."""

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "Query must not be empty."
        )

    base_path = (
        Path(base_directory)
        .expanduser()
        .resolve()
    )

    corpus = build_project_corpus(
        base_path
    )

    bm25_index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    bm25_results = (
        bm25_index.retrieve(
            clean_query,
            top_k=(
                settings.bm25_top_k
            ),
        )
    )

    client = chromadb.PersistentClient(
        path=str(
            settings.chroma_db_path
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

    identifiers = analyze_query_ids(
        clean_query
    )

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

        dense = EvidenceRetriever(
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
                settings.fused_top_k
            ),
            min_relevance_score=(
                settings.min_relevance_score
            ),
        )

        exact_results = (
            dense.exact_id_retrieval(
                identifiers
            )
        )

        semantic_results = (
            dense.semantic_retrieval(
                clean_query
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

    return build_final_context(
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
            settings.final_context_limit
        ),
    )
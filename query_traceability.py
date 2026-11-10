"""
REQTRACE AI Phase 11C controlled traceability diagnostic.

Pipeline:

- Phase 10 exact-ID retrieval
- Phase 11A BM25 retrieval
- Phase 10 semantic retrieval
- Phase 11B RRF fusion
- controlled anchor selection
- explicit REQ / OBS / TC family expansion

No final-context diversification.
No LLM generation.
No assessment or closure decision.
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
    RetrievedEvidence,
    analyze_query_ids,
)
from src.traceability import (
    expand_engineering_family,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 11C "
            "controlled engineering-family expansion."
        )
    )

    parser.add_argument(
        "query",
        help="Engineering retrieval query.",
    )

    parser.add_argument(
        "--anchor-count",
        type=int,
        default=1,
        help=(
            "Number of fused anchors for a query "
            "without explicit engineering IDs."
        ),
    )

    parser.add_argument(
        "--bm25-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--vector-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--fused-k",
        type=int,
        default=8,
    )

    return parser.parse_args()


def select_anchors(
    *,
    exact_results: list[RetrievedEvidence],
    fused_results: list[RetrievedEvidence],
    anchor_count: int,
) -> list[RetrievedEvidence]:
    """
    Select controlled expansion anchors.

    Explicit-ID query:
        Use all exact results.

    Natural-language query:
        Use only the strongest fused result by default.
    """

    if anchor_count < 1:
        raise ValueError(
            "anchor_count must be positive."
        )

    if exact_results:
        return list(
            exact_results
        )

    return list(
        fused_results[
            :anchor_count
        ]
    )


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
            top_k=args.bm25_k,
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
        dense_retriever = (
            EvidenceRetriever(
                client=chroma_client,
                collection_names=(
                    collection_names
                ),
                embedding_client=(
                    embedding_client
                ),
                vector_top_k=(
                    args.vector_k
                ),
                final_context_limit=(
                    args.fused_k
                ),
                min_relevance_score=(
                    settings
                    .min_relevance_score
                ),
            )
        )

        exact_results = (
            dense_retriever
            .exact_id_retrieval(
                identifiers
            )
        )

        semantic_results = (
            dense_retriever
            .semantic_retrieval(
                args.query
            )
        )

    ranked_lists = {
        "bm25": bm25_results,
        "semantic": semantic_results,
    }

    fused = reciprocal_rank_fusion(
        ranked_lists,
        rank_constant=60,
    )

    fused_results = (
        normalize_fused_results(
            fused,
            top_k=args.fused_k,
        )
    )

    anchors = select_anchors(
        exact_results=exact_results,
        fused_results=fused_results,
        anchor_count=(
            args.anchor_count
        ),
    )

    expansion = expand_engineering_family(
        corpus,
        anchors,
        maximum_iterations=3,
        maximum_results=20,
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 11C "
        "CONTROLLED TRACEABILITY"
    )
    print("=" * 78)

    print(
        f"Query         : {args.query}"
    )

    print(
        f"Exact results : "
        f"{len(exact_results)}"
    )

    print(
        f"BM25 results  : "
        f"{len(bm25_results)}"
    )

    print(
        f"Semantic      : "
        f"{len(semantic_results)}"
    )

    print(
        f"Fused results : "
        f"{len(fused_results)}"
    )

    print(
        f"Anchors       : "
        f"{len(expansion.anchors)}"
    )

    print(
        f"Expanded      : "
        f"{len(expansion.expanded)}"
    )

    print(
        f"Iterations    : "
        f"{expansion.iterations}"
    )

    print()
    print("Final Traceability Keys")
    print("-" * 78)

    print(
        f"Requirements : "
        f"{expansion.keys.requirement_ids}"
    )

    print(
        f"Observations : "
        f"{expansion.keys.observation_ids}"
    )

    print(
        f"Tests        : "
        f"{expansion.keys.test_case_ids}"
    )

    print()
    print("Anchor Evidence")
    print("-" * 78)

    for item in expansion.anchors:
        print(
            f"{item.evidence_id}"
        )

    print()
    print("Expanded Evidence Family")
    print("-" * 78)

    for item in expansion.expanded:
        metadata = item.metadata

        print()
        print(
            f"[{item.rank}] "
            f"{item.evidence_id}"
        )

        print(
            f"    Type : "
            f"{item.document_type}"
        )

        if metadata.get(
            "requirement_id"
        ):
            print(
                f"    REQ  : "
                f"{metadata['requirement_id']}"
            )

        if metadata.get(
            "observation_id"
        ):
            print(
                f"    OBS  : "
                f"{metadata['observation_id']}"
            )

        if metadata.get(
            "test_case_id"
        ):
            print(
                f"    TEST : "
                f"{metadata['test_case_id']}"
            )

        if metadata.get(
            "function_name"
        ):
            print(
                f"    Func : "
                f"{metadata['function_name']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
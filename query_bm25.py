"""
REQTRACE AI Phase 11A BM25 retrieval diagnostic.

This command performs lexical retrieval only.

No semantic retrieval.
No ChromaDB query.
No traceability expansion.
No LLM generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_settings
from src.bm25_retrieval import (
    BM25EvidenceIndex,
)
from src.corpus import (
    build_project_corpus,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REQTRACE AI Phase 11A "
            "BM25 lexical retrieval."
        )
    )

    parser.add_argument(
        "query",
        help="Engineering retrieval query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Number of BM25 results. "
            "Defaults to BM25_TOP_K."
        ),
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

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    top_k = (
        args.top_k
        if args.top_k is not None
        else settings.bm25_top_k
    )

    results = index.retrieve(
        args.query,
        top_k=top_k,
    )

    print()
    print("=" * 78)
    print(
        "REQTRACE AI - PHASE 11A BM25 RETRIEVAL"
    )
    print("=" * 78)

    print(
        f"Query       : {args.query}"
    )

    print(
        f"Corpus size : {corpus.record_count}"
    )

    print(
        f"BM25 Top-K  : {top_k}"
    )

    print(
        f"Results     : {len(results)}"
    )

    print()
    print("Ranked Lexical Evidence")
    print("-" * 78)

    if not results:
        print(
            "No positive BM25 evidence found."
        )
        return 0

    for item in results:
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

        requirement_id = (
            metadata.get(
                "requirement_id"
            )
        )

        observation_id = (
            metadata.get(
                "observation_id"
            )
        )

        test_case_id = (
            metadata.get(
                "test_case_id"
            )
        )

        function_name = (
            metadata.get(
                "function_name"
            )
        )

        if requirement_id:
            print(
                f"    REQ    : "
                f"{requirement_id}"
            )

        if observation_id:
            print(
                f"    OBS    : "
                f"{observation_id}"
            )

        if test_case_id:
            print(
                f"    TEST   : "
                f"{test_case_id}"
            )

        if function_name:
            print(
                f"    Func   : "
                f"{function_name}"
            )

        preview = " ".join(
            item.content.split()
        )

        print(
            f"    Preview: "
            f"{preview[:240]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

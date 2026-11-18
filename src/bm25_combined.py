"""
Combined lexical retrieval for REQTRACE AI.

Merges:

- original EvidenceCorpus BM25 results;
- Phase 16H dynamic-manifest BM25 results.

Duplicate evidence IDs are collapsed deterministically.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.retrieval import (
    RetrievedEvidence,
)


def merge_bm25_results(
    *result_groups: Iterable[
        RetrievedEvidence
    ],
    top_k: int,
) -> list[
    RetrievedEvidence
]:
    """Merge lexical result groups by evidence ID."""

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    merged: dict[
        str,
        RetrievedEvidence,
    ] = {}

    for group in result_groups:

        for item in group:

            existing = merged.get(
                item.evidence_id
            )

            if (
                existing is None
                or item.score > existing.score
            ):
                merged[
                    item.evidence_id
                ] = item

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            -item.score,
            item.evidence_id,
        ),
    )

    return [
        RetrievedEvidence(
            evidence_id=(
                item.evidence_id
            ),
            document_type=(
                item.document_type
            ),
            content=(
                item.content
            ),
            metadata=dict(
                item.metadata
            ),
            retrieval_method=(
                item.retrieval_method
            ),
            score=(
                item.score
            ),
            rank=rank,
        )
        for rank, item
        in enumerate(
            ordered[
                :top_k
            ],
            start=1,
        )
    ]
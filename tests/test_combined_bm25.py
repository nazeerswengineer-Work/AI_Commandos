"""
Tests for combined baseline and dynamic BM25 results.
"""

from __future__ import annotations

from src.bm25_combined import (
    merge_bm25_results,
)

from src.retrieval import (
    RetrievedEvidence,
)


def evidence(
    evidence_id: str,
    score: float,
    method: str,
) -> RetrievedEvidence:

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type="frd",
        content="Synthetic content.",
        metadata={
            "document_type": "frd",
        },
        retrieval_method=method,
        score=score,
        rank=1,
    )


def test_baseline_and_dynamic_results_merge() -> None:

    result = merge_bm25_results(
        [
            evidence(
                "BASE",
                0.8,
                "bm25",
            )
        ],

        [
            evidence(
                "DYNAMIC",
                1.0,
                "bm25_dynamic",
            )
        ],

        top_k=5,
    )

    assert [
        item.evidence_id
        for item in result
    ] == [
        "DYNAMIC",
        "BASE",
    ]


def test_duplicate_id_is_collapsed() -> None:

    result = merge_bm25_results(
        [
            evidence(
                "SAME",
                0.4,
                "bm25",
            )
        ],

        [
            evidence(
                "SAME",
                0.9,
                "bm25_dynamic",
            )
        ],

        top_k=5,
    )

    assert len(result) == 1

    assert (
        result[0].score
        == 0.9
    )
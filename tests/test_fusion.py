"""
Tests for Phase 11B Reciprocal Rank Fusion.
"""

from __future__ import annotations

import pytest

from src.fusion import (
    normalize_fused_results,
    reciprocal_rank_fusion,
    reciprocal_rank_score,
)
from src.retrieval import (
    RetrievedEvidence,
)


def make_result(
    evidence_id: str,
    method: str,
    rank: int,
    score: float = 1.0,
) -> RetrievedEvidence:

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type="frd",
        content="Evidence content",
        metadata={
            "document_type": "frd"
        },
        retrieval_method=method,
        score=score,
        rank=rank,
    )


def test_rrf_formula() -> None:

    score = reciprocal_rank_score(
        1,
        60,
    )

    assert score == pytest.approx(
        1.0 / 61.0
    )


def test_invalid_rank_is_rejected() -> None:

    with pytest.raises(
        ValueError
    ):
        reciprocal_rank_score(
            0,
            60,
        )


def test_invalid_rank_constant_is_rejected() -> None:

    with pytest.raises(
        ValueError
    ):
        reciprocal_rank_score(
            1,
            0,
        )


def test_record_present_in_both_lists_is_boosted() -> None:

    bm25 = [
        make_result(
            "A",
            "bm25",
            1,
        ),
        make_result(
            "B",
            "bm25",
            2,
        ),
    ]

    semantic = [
        make_result(
            "B",
            "semantic",
            1,
        ),
        make_result(
            "C",
            "semantic",
            2,
        ),
    ]

    fused = reciprocal_rank_fusion(
        {
            "bm25": bm25,
            "semantic": semantic,
        }
    )

    assert (
        fused[0]
        .evidence
        .evidence_id
        == "B"
    )

    assert len(
        fused[0].contributions
    ) == 2


def test_duplicate_inside_one_channel_counts_once() -> None:

    bm25 = [
        make_result(
            "A",
            "bm25",
            1,
        ),
        make_result(
            "A",
            "bm25",
            2,
        ),
    ]

    fused = reciprocal_rank_fusion(
        {
            "bm25": bm25
        }
    )

    assert len(fused) == 1

    assert len(
        fused[0].contributions
    ) == 1


def test_normalized_fused_results() -> None:

    bm25 = [
        make_result(
            "A",
            "bm25",
            1,
        ),
        make_result(
            "B",
            "bm25",
            2,
        ),
    ]

    semantic = [
        make_result(
            "A",
            "semantic",
            1,
        )
    ]

    fused = reciprocal_rank_fusion(
        {
            "bm25": bm25,
            "semantic": semantic,
        }
    )

    results = (
        normalize_fused_results(
            fused,
            top_k=2,
        )
    )

    assert len(results) == 2

    assert results[0].rank == 1

    assert (
        results[0]
        .retrieval_method
        == "rrf"
    )

    assert (
        results[0].score
        == pytest.approx(1.0)
    )

    assert (
        "bm25:"
        in results[0]
        .metadata[
            "fusion_contributions"
        ]
    )

    assert (
        "semantic:"
        in results[0]
        .metadata[
            "fusion_contributions"
        ]
    )
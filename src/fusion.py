"""
Reciprocal Rank Fusion for REQTRACE AI.

Phase 11B combines independently validated:

1. BM25 lexical retrieval
2. Dense semantic retrieval

RRF operates on ranking positions rather than trying to compare
BM25 scores directly with cosine-similarity scores.

This module does not:
- perform traceability expansion
- access an LLM
- make engineering decisions
- determine closure readiness
"""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval import RetrievedEvidence


@dataclass(frozen=True)
class FusionContribution:
    """One retrieval channel's contribution to a fused result."""

    retrieval_method: str
    source_rank: int
    contribution: float


@dataclass(frozen=True)
class FusedEvidence:
    """One evidence record after Reciprocal Rank Fusion."""

    evidence: RetrievedEvidence
    rrf_score: float

    contributions: tuple[
        FusionContribution,
        ...
    ]


def reciprocal_rank_score(
    rank: int,
    rank_constant: int = 60,
) -> float:
    """
    Calculate one RRF contribution.

    Formula:

        1 / (rank_constant + rank)

    RRF score is a retrieval-ranking quantity.

    It is NOT:
    - engineering confidence
    - compliance probability
    - observation-validity probability
    """

    if rank < 1:
        raise ValueError(
            "rank must be one-based."
        )

    if rank_constant < 1:
        raise ValueError(
            "rank_constant must be positive."
        )

    return 1.0 / (
        rank_constant + rank
    )


def reciprocal_rank_fusion(
    ranked_lists: dict[
        str,
        list[RetrievedEvidence],
    ],
    rank_constant: int = 60,
) -> tuple[FusedEvidence, ...]:
    """
    Fuse multiple ranked evidence lists.

    Evidence appearing in both BM25 and semantic results receives
    contributions from both channels.

    Duplicate evidence IDs inside the same retrieval channel are counted
    only once.
    """

    if rank_constant < 1:
        raise ValueError(
            "rank_constant must be positive."
        )

    scores: dict[
        str,
        float,
    ] = {}

    evidence_by_id: dict[
        str,
        RetrievedEvidence,
    ] = {}

    contributions_by_id: dict[
        str,
        list[FusionContribution],
    ] = {}

    for retrieval_method, items in ranked_lists.items():

        seen_in_channel: set[str] = set()

        for fallback_rank, item in enumerate(
            items,
            start=1,
        ):
            evidence_id = (
                item.evidence_id
            )

            if evidence_id in seen_in_channel:
                continue

            seen_in_channel.add(
                evidence_id
            )

            if item.rank >= 1:
                source_rank = (
                    item.rank
                )
            else:
                source_rank = (
                    fallback_rank
                )

            contribution = reciprocal_rank_score(
                source_rank,
                rank_constant,
            )

            previous_score = scores.get(
                evidence_id,
                0.0,
            )

            scores[
                evidence_id
            ] = (
                previous_score
                + contribution
            )

            if evidence_id not in contributions_by_id:
                contributions_by_id[
                    evidence_id
                ] = []

            contributions_by_id[
                evidence_id
            ].append(
                FusionContribution(
                    retrieval_method=(
                        retrieval_method
                    ),
                    source_rank=(
                        source_rank
                    ),
                    contribution=(
                        contribution
                    ),
                )
            )

            existing = evidence_by_id.get(
                evidence_id
            )

            if existing is None:
                evidence_by_id[
                    evidence_id
                ] = item

            elif item.score > existing.score:
                evidence_by_id[
                    evidence_id
                ] = item

    fused_results: list[
        FusedEvidence
    ] = []

    for evidence_id, score in scores.items():

        fused_results.append(
            FusedEvidence(
                evidence=(
                    evidence_by_id[
                        evidence_id
                    ]
                ),
                rrf_score=score,
                contributions=tuple(
                    contributions_by_id[
                        evidence_id
                    ]
                ),
            )
        )

    fused_results.sort(
        key=lambda result: (
            -result.rrf_score,
            result.evidence.evidence_id,
        )
    )

    return tuple(
        fused_results
    )


def normalize_fused_results(
    fused_results: tuple[
        FusedEvidence,
        ...
    ],
    top_k: int = 6,
) -> list[RetrievedEvidence]:
    """
    Convert fused results to the common RetrievedEvidence structure.

    The resulting score is normalized for diagnostics only.

    It must NOT be interpreted as engineering confidence.
    """

    if top_k < 1:
        raise ValueError(
            "top_k must be positive."
        )

    selected = fused_results[
        :top_k
    ]

    if not selected:
        return []

    maximum_score = max(
        result.rrf_score
        for result in selected
    )

    retrieved: list[
        RetrievedEvidence
    ] = []

    for rank, result in enumerate(
        selected,
        start=1,
    ):
        if maximum_score > 0.0:
            normalized_score = (
                result.rrf_score
                / maximum_score
            )
        else:
            normalized_score = 0.0

        metadata = dict(
            result.evidence.metadata
        )

        contribution_text: list[
            str
        ] = []

        for contribution in (
            result.contributions
        ):
            contribution_text.append(
                (
                    contribution
                    .retrieval_method
                    + ":"
                    + str(
                        contribution
                        .source_rank
                    )
                )
            )

        metadata[
            "fusion_contributions"
        ] = ";".join(
            contribution_text
        )

        retrieved.append(
            RetrievedEvidence(
                evidence_id=(
                    result.evidence
                    .evidence_id
                ),
                document_type=(
                    result.evidence
                    .document_type
                ),
                content=(
                    result.evidence
                    .content
                ),
                metadata=metadata,
                retrieval_method=(
                    "rrf"
                ),
                score=(
                    normalized_score
                ),
                rank=rank,
            )
        )

    return retrieved
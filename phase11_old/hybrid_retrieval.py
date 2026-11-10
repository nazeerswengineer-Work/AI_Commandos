"""
Hybrid retrieval orchestration for REQTRACE AI.

Phase 11 combines:

- exact engineering-ID retrieval
- BM25 lexical retrieval
- dense semantic retrieval
- reciprocal rank fusion
- explicit traceability expansion
- evidence-type diversity
- protection of authoritative exact-ID results

Engineering rules:

1. Exact-ID evidence has highest retrieval priority.
2. Natural-language queries expand traceability only from the strongest
   fused evidence anchor.
3. Explicit-ID queries expand from all exact matches.
4. Exact-ID evidence is never removed merely to improve evidence-type
   diversity.
5. Retrieval does not make engineering assessments.
6. Retrieval does not determine closure readiness.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bm25_retrieval import BM25EvidenceIndex
from src.corpus import EvidenceCorpus
from src.fusion import (
    fused_to_retrieved,
    reciprocal_rank_fusion,
)
from src.retrieval import (
    EvidenceRetriever,
    QueryIdentifiers,
    RetrievedEvidence,
    analyze_query_ids,
)
from src.traceability import expand_traceability


@dataclass(frozen=True)
class HybridRetrievalResult:
    """Complete Phase 11 hybrid retrieval result."""

    query: str
    identifiers: QueryIdentifiers

    exact_results: tuple[RetrievedEvidence, ...]
    bm25_results: tuple[RetrievedEvidence, ...]
    semantic_results: tuple[RetrievedEvidence, ...]
    traceability_results: tuple[RetrievedEvidence, ...]

    evidence: tuple[RetrievedEvidence, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return final evidence IDs in ranked order."""

        return tuple(
            item.evidence_id
            for item in self.evidence
        )

    @property
    def evidence_types(self) -> tuple[str, ...]:
        """Return ordered unique evidence types."""

        return tuple(
            dict.fromkeys(
                item.document_type
                for item in self.evidence
            )
        )


def deduplicate_results(
    results: list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    """
    Remove duplicate evidence IDs while preserving existing order.

    The first occurrence wins because the caller has already established
    retrieval priority.
    """

    deduplicated: list[RetrievedEvidence] = []
    seen: set[str] = set()

    for item in results:
        if item.evidence_id in seen:
            continue

        seen.add(item.evidence_id)
        deduplicated.append(item)

    return deduplicated


def select_traceability_seeds(
    *,
    exact_results: list[RetrievedEvidence],
    fused_results: list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    """
    Select a coherent evidence family for traceability expansion.

    Explicit-ID query:
        Expand from all exact-ID records.

    Natural-language query:
        Expand only from the strongest fused record.

    This prevents lower-ranked unrelated candidate families from
    contaminating traceability expansion.
    """

    if exact_results:
        return deduplicate_results(
            exact_results
        )

    if fused_results:
        return [
            fused_results[0]
        ]

    return []


def diversify_evidence_types(
    results: list[RetrievedEvidence],
    *,
    limit: int,
) -> list[RetrievedEvidence]:
    """
    Prefer evidence-type coverage while preserving ranking order.

    Used mainly for natural-language queries where there are no protected
    exact-ID results.
    """

    if limit < 1:
        return []

    selected: list[RetrievedEvidence] = []
    selected_ids: set[str] = set()
    selected_types: set[str] = set()

    for item in results:
        if item.document_type in selected_types:
            continue

        selected.append(item)
        selected_ids.add(item.evidence_id)
        selected_types.add(item.document_type)

        if len(selected) >= limit:
            return selected

    for item in results:
        if item.evidence_id in selected_ids:
            continue

        selected.append(item)
        selected_ids.add(item.evidence_id)

        if len(selected) >= limit:
            break

    return selected


def protect_exact_results(
    exact_results: list[RetrievedEvidence],
    other_results: list[RetrievedEvidence],
    *,
    limit: int,
    diversify_types: bool = True,
) -> list[RetrievedEvidence]:
    """
    Preserve exact-ID evidence before filling remaining context positions.

    Example:

        Query: REQ-050

    Exact retrieval can legitimately return:

        FRD REQ-050 Rev01
        FRD REQ-050 Rev02
        OBS-061

    All three exact records must survive final-context selection.
    """

    if limit < 1:
        return []

    exact = deduplicate_results(
        exact_results
    )

    if len(exact) >= limit:
        return exact[:limit]

    remaining_limit = (
        limit - len(exact)
    )

    exact_ids = {
        item.evidence_id
        for item in exact
    }

    deduplicated_other = (
        deduplicate_results(
            other_results
        )
    )

    remaining: list[RetrievedEvidence] = []

    for item in deduplicated_other:
        if item.evidence_id in exact_ids:
            continue

        remaining.append(item)

    if not diversify_types:
        final_results = list(exact)

        final_results.extend(
            remaining[:remaining_limit]
        )

        return final_results

    used_types = {
        item.document_type
        for item in exact
    }

    selected_additional: list[
        RetrievedEvidence
    ] = []

    selected_additional_ids: set[str] = set()

    # First pass:
    # Prefer an evidence type not already represented
    # by the protected exact-ID evidence.

    for item in remaining:
        if item.document_type in used_types:
            continue

        selected_additional.append(item)
        selected_additional_ids.add(
            item.evidence_id
        )
        used_types.add(
            item.document_type
        )

        if (
            len(selected_additional)
            >= remaining_limit
        ):
            break

    # Second pass:
    # If positions remain, use the original ranking.

    if (
        len(selected_additional)
        < remaining_limit
    ):
        for item in remaining:
            if (
                item.evidence_id
                in selected_additional_ids
            ):
                continue

            selected_additional.append(
                item
            )

            selected_additional_ids.add(
                item.evidence_id
            )

            if (
                len(selected_additional)
                >= remaining_limit
            ):
                break

    final_results = list(exact)

    final_results.extend(
        selected_additional[
            :remaining_limit
        ]
    )

    return final_results


def assign_final_ranks(
    results: list[RetrievedEvidence],
) -> tuple[RetrievedEvidence, ...]:
    """Assign one-based ranks to final retrieval results."""

    ranked_results: list[
        RetrievedEvidence
    ] = []

    for rank, item in enumerate(
        results,
        start=1,
    ):
        ranked_results.append(
            RetrievedEvidence(
                evidence_id=item.evidence_id,
                document_type=(
                    item.document_type
                ),
                content=item.content,
                metadata=item.metadata,
                retrieval_method=(
                    item.retrieval_method
                ),
                score=item.score,
                rank=rank,
            )
        )

    return tuple(ranked_results)


class HybridEvidenceRetriever:
    """
    Phase 11 hybrid evidence retrieval pipeline.

    Retrieval flow:

        Query
          |
          v
        Exact-ID retrieval
          |
          +------------------+
          |                  |
          v                  v
        BM25              Semantic
          |                  |
          +--------+---------+
                   |
                   v
             RRF fusion
                   |
                   v
        Select traceability seed
                   |
                   v
        Traceability expansion
                   |
                   v
        Protect exact evidence
                   |
                   v
        Evidence-type diversity
                   |
                   v
            Final context
    """

    def __init__(
        self,
        *,
        corpus: EvidenceCorpus,
        dense_retriever: EvidenceRetriever,
        bm25_index: BM25EvidenceIndex,
        bm25_top_k: int = 5,
        fused_top_k: int = 6,
        final_context_limit: int = 4,
        rrf_rank_constant: int = 60,
        traceability_limit: int = 20,
        diversify_types: bool = True,
    ) -> None:

        if bm25_top_k < 1:
            raise ValueError(
                "bm25_top_k must be positive."
            )

        if fused_top_k < 1:
            raise ValueError(
                "fused_top_k must be positive."
            )

        if final_context_limit < 1:
            raise ValueError(
                "final_context_limit must be positive."
            )

        if rrf_rank_constant < 1:
            raise ValueError(
                "rrf_rank_constant must be positive."
            )

        if traceability_limit < 1:
            raise ValueError(
                "traceability_limit must be positive."
            )

        self.corpus = corpus
        self.dense_retriever = dense_retriever
        self.bm25_index = bm25_index
        self.bm25_top_k = bm25_top_k
        self.fused_top_k = fused_top_k
        self.final_context_limit = (
            final_context_limit
        )
        self.rrf_rank_constant = (
            rrf_rank_constant
        )
        self.traceability_limit = (
            traceability_limit
        )
        self.diversify_types = (
            diversify_types
        )

    def retrieve(
        self,
        query: str,
    ) -> HybridRetrievalResult:
        """
        Run complete Phase 11 hybrid retrieval.

        Retrieval only gathers and ranks evidence.
        No engineering assessment is calculated.
        """

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Query must not be empty."
            )

        # ---------------------------------------------------------
        # 1. Analyze explicit engineering IDs
        # ---------------------------------------------------------

        identifiers = analyze_query_ids(
            clean_query
        )

        # ---------------------------------------------------------
        # 2. Exact-ID retrieval
        # ---------------------------------------------------------

        exact_results = (
            self.dense_retriever
            .exact_id_retrieval(
                identifiers
            )
        )

        exact_results = deduplicate_results(
            exact_results
        )

        # ---------------------------------------------------------
        # 3. Dense semantic retrieval
        # ---------------------------------------------------------

        semantic_results = (
            self.dense_retriever
            .semantic_retrieval(
                clean_query
            )
        )

        semantic_results = (
            deduplicate_results(
                semantic_results
            )
        )

        # ---------------------------------------------------------
        # 4. BM25 lexical retrieval
        # ---------------------------------------------------------

        bm25_results = (
            self.bm25_index.retrieve(
                clean_query,
                top_k=self.bm25_top_k,
            )
        )

        bm25_results = (
            deduplicate_results(
                bm25_results
            )
        )

        # ---------------------------------------------------------
        # 5. Reciprocal Rank Fusion
        # ---------------------------------------------------------

        ranked_lists = {
            "bm25": bm25_results,
            "semantic": semantic_results,
        }

        fused = reciprocal_rank_fusion(
            ranked_lists,
            rank_constant=(
                self.rrf_rank_constant
            ),
        )

        fused_results = (
            fused_to_retrieved(
                fused
            )
        )

        fused_results = (
            fused_results[
                :self.fused_top_k
            ]
        )

        # ---------------------------------------------------------
        # 6. Select traceability seeds
        # ---------------------------------------------------------

        traceability_seeds = (
            select_traceability_seeds(
                exact_results=exact_results,
                fused_results=fused_results,
            )
        )

        # ---------------------------------------------------------
        # 7. Explicit traceability expansion
        # ---------------------------------------------------------

        traceability_results = (
            expand_traceability(
                self.corpus,
                traceability_seeds,
                limit=(
                    self.traceability_limit
                ),
            )
        )

        traceability_results = (
            deduplicate_results(
                traceability_results
            )
        )

        # ---------------------------------------------------------
        # 8. Construct ordered candidate list
        # ---------------------------------------------------------

        ordered_candidates: list[
            RetrievedEvidence
        ] = []

        ordered_candidates.extend(
            exact_results
        )

        ordered_candidates.extend(
            traceability_results
        )

        ordered_candidates.extend(
            fused_results
        )

        ordered_candidates = (
            deduplicate_results(
                ordered_candidates
            )
        )

        # ---------------------------------------------------------
        # 9. Final context selection
        # ---------------------------------------------------------

        if exact_results:
            other_results: list[
                RetrievedEvidence
            ] = []

            other_results.extend(
                traceability_results
            )

            other_results.extend(
                fused_results
            )

            final_results = (
                protect_exact_results(
                    exact_results,
                    other_results,
                    limit=(
                        self.final_context_limit
                    ),
                    diversify_types=(
                        self.diversify_types
                    ),
                )
            )

        elif self.diversify_types:
            final_results = (
                diversify_evidence_types(
                    ordered_candidates,
                    limit=(
                        self.final_context_limit
                    ),
                )
            )

        else:
            final_results = (
                ordered_candidates[
                    :self.final_context_limit
                ]
            )

        # ---------------------------------------------------------
        # 10. Assign final ranks
        # ---------------------------------------------------------

        ranked_final_results = (
            assign_final_ranks(
                final_results
            )
        )

        # ---------------------------------------------------------
        # 11. Return retrieval result
        # ---------------------------------------------------------

        return HybridRetrievalResult(
            query=clean_query,
            identifiers=identifiers,
            exact_results=tuple(
                exact_results
            ),
            bm25_results=tuple(
                bm25_results
            ),
            semantic_results=tuple(
                semantic_results
            ),
            traceability_results=tuple(
                traceability_results
            ),
            evidence=ranked_final_results,
        )
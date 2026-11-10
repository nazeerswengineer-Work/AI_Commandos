"""
Retrieval evaluation for REQTRACE AI.

Metrics:
- Precision@K
- Recall@K
- Reciprocal rank
- evidence-type coverage
- mean aggregate metrics
"""

from __future__ import annotations

from dataclasses import dataclass

from src.hybrid_retrieval import (
    HybridEvidenceRetriever,
)


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """One labelled retrieval query."""

    case_id: str
    query: str
    expected_evidence_ids: tuple[
        str,
        ...
    ] = ()
    expected_document_types: tuple[
        str,
        ...
    ] = ()
    expected_engineering_ids: tuple[
        str,
        ...
    ] = ()


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Metrics for one evaluation case."""

    case_id: str
    query: str

    retrieved_ids: tuple[str, ...]
    retrieved_types: tuple[str, ...]

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    type_coverage: float


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate retrieval-evaluation report."""

    cases: tuple[
        RetrievalCaseResult,
        ...
    ]

    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_type_coverage: float


def precision_at_k(
    retrieved: tuple[str, ...],
    relevant: tuple[str, ...],
) -> float:
    """Calculate precision at the supplied result depth."""

    if not retrieved:
        return 0.0

    relevant_set = set(relevant)

    matches = sum(
        1
        for evidence_id in retrieved
        if evidence_id in relevant_set
    )

    return matches / len(retrieved)


def recall_at_k(
    retrieved: tuple[str, ...],
    relevant: tuple[str, ...],
) -> float:
    """Calculate recall at the supplied result depth."""

    if not relevant:
        return 1.0

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    return (
        len(
            retrieved_set & relevant_set
        )
        / len(relevant_set)
    )


def reciprocal_rank(
    retrieved: tuple[str, ...],
    relevant: tuple[str, ...],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""

    relevant_set = set(relevant)

    for rank, evidence_id in enumerate(
        retrieved,
        start=1,
    ):
        if evidence_id in relevant_set:
            return 1.0 / rank

    return 0.0


def evidence_type_coverage(
    retrieved_types: tuple[str, ...],
    expected_types: tuple[str, ...],
) -> float:
    """Calculate required evidence-type coverage."""

    if not expected_types:
        return 1.0

    return (
        len(
            set(retrieved_types)
            & set(expected_types)
        )
        / len(set(expected_types))
    )


def evaluate_case(
    retriever: HybridEvidenceRetriever,
    case: RetrievalEvaluationCase,
) -> RetrievalCaseResult:
    """Evaluate one labelled query."""

    result = retriever.retrieve(
        case.query
    )

    retrieved_ids = result.evidence_ids
    retrieved_types = tuple(
        item.document_type
        for item in result.evidence
    )

    return RetrievalCaseResult(
        case_id=case.case_id,
        query=case.query,
        retrieved_ids=retrieved_ids,
        retrieved_types=retrieved_types,
        precision_at_k=precision_at_k(
            retrieved_ids,
            case.expected_evidence_ids,
        ),
        recall_at_k=recall_at_k(
            retrieved_ids,
            case.expected_evidence_ids,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_ids,
            case.expected_evidence_ids,
        ),
        type_coverage=(
            evidence_type_coverage(
                retrieved_types,
                case.expected_document_types,
            )
        ),
    )


def evaluate_retriever(
    retriever: HybridEvidenceRetriever,
    cases: tuple[
        RetrievalEvaluationCase,
        ...
    ],
) -> RetrievalEvaluationReport:
    """Evaluate all labelled retrieval cases."""

    if not cases:
        raise ValueError(
            "At least one evaluation case is required."
        )

    results = tuple(
        evaluate_case(
            retriever,
            case,
        )
        for case in cases
    )

    count = len(results)

    return RetrievalEvaluationReport(
        cases=results,
        mean_precision_at_k=(
            sum(
                item.precision_at_k
                for item in results
            )
            / count
        ),
        mean_recall_at_k=(
            sum(
                item.recall_at_k
                for item in results
            )
            / count
        ),
        mean_reciprocal_rank=(
            sum(
                item.reciprocal_rank
                for item in results
            )
            / count
        ),
        mean_type_coverage=(
            sum(
                item.type_coverage
                for item in results
            )
            / count
        ),
    )
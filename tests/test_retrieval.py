from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.retrieval import (
    EvidenceRetriever,
    RetrievedEvidence,
    analyze_query_ids,
    distance_to_score,
)


def test_query_id_extraction() -> None:
    result = analyze_query_ids(
        "Can OBS_023 be closed against "
        "REQ 017 using TC-102?"
    )

    assert result.observation_ids == (
        "OBS-023",
    )

    assert result.requirement_ids == (
        "REQ-017",
    )

    assert result.test_case_ids == (
        "TC-102",
    )


def test_query_without_ids() -> None:
    result = analyze_query_ids(
        "What evidence discusses "
        "null input handling?"
    )

    assert result.has_explicit_ids is False


def test_distance_conversion() -> None:
    assert distance_to_score(
        0.0
    ) == pytest.approx(1.0)

    assert distance_to_score(
        0.25
    ) == pytest.approx(0.75)

    assert distance_to_score(
        2.0
    ) == 0.0


def make_item(
    evidence_id: str,
    method: str,
    score: float,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type="frd",
        content="content",
        metadata={},
        retrieval_method=method,
        score=score,
    )


def test_exact_result_overrides_semantic_duplicate() -> None:
    semantic = [
        make_item(
            "EVIDENCE-1",
            "semantic",
            0.95,
        )
    ]

    exact = [
        make_item(
            "EVIDENCE-1",
            "exact_id",
            1.0,
        )
    ]

    merged = (
        EvidenceRetriever
        .merge_results(
            exact,
            semantic,
        )
    )

    assert len(merged) == 1
    assert (
        merged[0].retrieval_method
        == "exact_id"
    )
    assert merged[0].score == 1.0


def test_exact_results_rank_before_semantic() -> None:
    exact = [
        make_item(
            "EXACT",
            "exact_id",
            1.0,
        )
    ]

    semantic = [
        make_item(
            "SEMANTIC",
            "semantic",
            0.99,
        )
    ]

    merged = (
        EvidenceRetriever
        .merge_results(
            exact,
            semantic,
        )
    )

    assert merged[0].evidence_id == (
        "EXACT"
    )


class FakeCollection:
    def __init__(
        self,
        records: list[
            dict[str, Any]
        ],
    ) -> None:
        self.records = records

    def get(
        self,
        *,
        where: dict[str, str],
        include: list[str],
    ) -> dict[str, Any]:
        field_name, value = next(
            iter(where.items())
        )

        matches = [
            record
            for record in self.records
            if (
                record["metadata"].get(
                    field_name
                )
                == value
            )
        ]

        return {
            "ids": [
                item["id"]
                for item in matches
            ],
            "documents": [
                item["document"]
                for item in matches
            ],
            "metadatas": [
                item["metadata"]
                for item in matches
            ],
        }

    def count(self) -> int:
        return len(self.records)

    def query(
        self,
        *,
        query_embeddings: list[
            list[float]
        ],
        n_results: int,
        include: list[str],
    ) -> dict[str, Any]:
        matches = self.records[
            :n_results
        ]

        return {
            "ids": [[
                item["id"]
                for item in matches
            ]],
            "documents": [[
                item["document"]
                for item in matches
            ]],
            "metadatas": [[
                item["metadata"]
                for item in matches
            ]],
            "distances": [[
                item.get(
                    "distance",
                    0.2,
                )
                for item in matches
            ]],
        }


class FakeClient:
    def __init__(
        self,
        collection: FakeCollection,
    ) -> None:
        self.collection = collection

    def get_collection(
        self,
        *,
        name: str,
    ) -> FakeCollection:
        return self.collection


class FakeEmbeddingClient:
    def embed_text(
        self,
        text: str,
    ) -> tuple[float, ...]:
        return (
            0.1,
            0.2,
            0.3,
        )


def test_exact_id_retrieval() -> None:
    collection = FakeCollection(
        [
            {
                "id": "OBS::OBS-023",
                "document": (
                    "Observation content"
                ),
                "metadata": {
                    "document_type": (
                        "observation"
                    ),
                    "observation_id": (
                        "OBS-023"
                    ),
                },
            },
            {
                "id": "OBS::OBS-031",
                "document": "Other",
                "metadata": {
                    "document_type": (
                        "observation"
                    ),
                    "observation_id": (
                        "OBS-031"
                    ),
                },
            },
        ]
    )

    retriever = EvidenceRetriever(
        client=FakeClient(
            collection
        ),
        collection_names={
            "observation": "obs"
        },
        embedding_client=(
            FakeEmbeddingClient()
        ),
    )

    result = retriever.retrieve(
        "Review OBS-023",
        include_semantic=False,
    )

    assert result.exact_match_count == 1

    assert (
        result.evidence[0]
        .metadata["observation_id"]
        == "OBS-023"
    )


def test_semantic_score_threshold() -> None:
    collection = FakeCollection(
        [
            {
                "id": "A",
                "document": "Relevant",
                "metadata": {
                    "document_type": "frd"
                },
                "distance": 0.10,
            },
            {
                "id": "B",
                "document": "Weak",
                "metadata": {
                    "document_type": "frd"
                },
                "distance": 0.90,
            },
        ]
    )

    retriever = EvidenceRetriever(
        client=FakeClient(
            collection
        ),
        collection_names={
            "frd": "frd"
        },
        embedding_client=(
            FakeEmbeddingClient()
        ),
        vector_top_k=5,
        final_context_limit=5,
        min_relevance_score=0.30,
    )

    result = retriever.retrieve(
        "null input handling"
    )

    ids = result.evidence_ids

    assert "A" in ids
    assert "B" not in ids


def test_final_context_limit() -> None:
    collection = FakeCollection(
        [
            {
                "id": f"E-{index}",
                "document": "Evidence",
                "metadata": {
                    "document_type": "frd"
                },
                "distance": 0.1,
            }
            for index in range(10)
        ]
    )

    retriever = EvidenceRetriever(
        client=FakeClient(
            collection
        ),
        collection_names={
            "frd": "frd"
        },
        embedding_client=(
            FakeEmbeddingClient()
        ),
        vector_top_k=10,
        final_context_limit=4,
    )

    result = retriever.retrieve(
        "engineering query"
    )

    assert len(result.evidence) == 4

    assert tuple(
        item.rank
        for item in result.evidence
    ) == (
        1,
        2,
        3,
        4,
    )
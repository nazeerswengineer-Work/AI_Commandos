"""
REQTRACE AI Phase 10 retrieval.

Implements:

- explicit engineering-ID detection
- exact-ID-first retrieval
- Chroma semantic retrieval
- metadata filtering
- cosine-distance conversion
- evidence deduplication
- deterministic ranking

No LLM generation or engineering decision occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embeddings import OllamaEmbeddingClient
from src.normalization import (
    extract_observation_ids,
    extract_requirement_ids,
    extract_test_case_ids,
)


@dataclass(frozen=True)
class QueryIdentifiers:
    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    @property
    def has_explicit_ids(self) -> bool:
        return bool(
            self.requirement_ids
            or self.observation_ids
            or self.test_case_ids
        )


@dataclass(frozen=True)
class RetrievedEvidence:
    evidence_id: str
    document_type: str
    content: str
    metadata: dict[str, Any]

    retrieval_method: str
    score: float
    rank: int = 0

    @property
    def content_hash(self) -> str | None:
        value = self.metadata.get(
            "content_hash"
        )

        return (
            str(value)
            if value is not None
            else None
        )


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    identifiers: QueryIdentifiers
    evidence: tuple[RetrievedEvidence, ...]

    exact_match_count: int
    semantic_match_count: int

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            item.evidence_id
            for item in self.evidence
        )


def analyze_query_ids(
    query: str,
) -> QueryIdentifiers:
    """Extract normalized explicit IDs from a query."""

    if not query.strip():
        raise ValueError(
            "Query must not be empty."
        )

    return QueryIdentifiers(
        requirement_ids=(
            extract_requirement_ids(query)
        ),
        observation_ids=(
            extract_observation_ids(query)
        ),
        test_case_ids=(
            extract_test_case_ids(query)
        ),
    )


def distance_to_score(
    distance: float,
) -> float:
    """
    Convert cosine distance to a bounded similarity score.

    Chroma cosine distance is smaller for closer vectors.
    """

    score = 1.0 - float(distance)

    return max(
        0.0,
        min(1.0, score),
    )


def make_retrieved_evidence(
    *,
    evidence_id: str,
    content: str,
    metadata: dict[str, Any] | None,
    retrieval_method: str,
    score: float,
) -> RetrievedEvidence:
    """Create one validated retrieval result."""

    clean_metadata = (
        dict(metadata)
        if isinstance(metadata, dict)
        else {}
    )

    document_type = str(
        clean_metadata.get(
            "document_type",
            "unknown",
        )
    )

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type=document_type,
        content=content,
        metadata=clean_metadata,
        retrieval_method=retrieval_method,
        score=max(
            0.0,
            min(1.0, float(score)),
        ),
    )


class EvidenceRetriever:
    """Exact-ID-first Chroma retrieval."""

    def __init__(
        self,
        *,
        client: Any,
        collection_names: dict[str, str],
        embedding_client: OllamaEmbeddingClient,
        vector_top_k: int = 5,
        final_context_limit: int = 4,
        min_relevance_score: float = 0.30,
    ) -> None:
        if vector_top_k < 1:
            raise ValueError(
                "vector_top_k must be positive."
            )

        if final_context_limit < 1:
            raise ValueError(
                "final_context_limit must be positive."
            )

        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError(
                "min_relevance_score must be between 0 and 1."
            )

        if not collection_names:
            raise ValueError(
                "At least one collection must be configured."
            )

        self.client = client
        self.collection_names = dict(
            collection_names
        )
        self.embedding_client = (
            embedding_client
        )
        self.vector_top_k = vector_top_k
        self.final_context_limit = (
            final_context_limit
        )
        self.min_relevance_score = (
            min_relevance_score
        )

    def _collections(
        self,
    ) -> dict[str, Any]:
        """Open configured collections."""

        return {
            evidence_type:
            self.client.get_collection(
                name=collection_name
            )
            for evidence_type, collection_name
            in self.collection_names.items()
        }

    def exact_id_retrieval(
        self,
        identifiers: QueryIdentifiers,
    ) -> list:
        """
        Retrieve evidence using explicit metadata IDs.

        Exact matches receive score 1.0.
        """

        results: list[RetrievedEvidence] = []

        searches: list[
            tuple[str, str]
        ] = []

        searches.extend(
            ("requirement_id", value)
            for value
            in identifiers.requirement_ids
        )

        searches.extend(
            ("observation_id", value)
            for value
            in identifiers.observation_ids
        )

        searches.extend(
            ("test_case_id", value)
            for value
            in identifiers.test_case_ids
        )

        if not searches:
            return results

        for _, collection in (
            self._collections().items()
        ):
            for field_name, identifier in searches:
                response = collection.get(
                    where={
                        field_name: identifier
                    },
                    include=[
                        "documents",
                        "metadatas",
                    ],
                )

                ids = (
                    response.get("ids")
                    or []
                )
                documents = (
                    response.get("documents")
                    or []
                )
                metadatas = (
                    response.get("metadatas")
                    or []
                )

                for (
                    evidence_id,
                    document,
                    metadata,
                ) in zip(
                    ids,
                    documents,
                    metadatas,
                    strict=False,
                ):
                    results.append(
                        make_retrieved_evidence(
                            evidence_id=evidence_id,
                            content=document or "",
                            metadata=metadata,
                            retrieval_method=(
                                "exact_id"
                            ),
                            score=1.0,
                        )
                    )

        return results

    def semantic_retrieval(
        self,
        query: str,
    ) -> list:
        """Search each evidence collection using one query embedding."""

        query_embedding = list(
            self.embedding_client
            .embed_text(query)
        )

        results: list[RetrievedEvidence] = []

        for evidence_type, collection in (
            self._collections().items()
        ):
            count = collection.count()

            if count == 0:
                continue

            n_results = min(
                self.vector_top_k,
                count,
            )

            response = collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=n_results,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            ids_groups = (
                response.get("ids")
                or [[]]
            )
            document_groups = (
                response.get("documents")
                or [[]]
            )
            metadata_groups = (
                response.get("metadatas")
                or [[]]
            )
            distance_groups = (
                response.get("distances")
                or [[]]
            )

            ids = (
                ids_groups[0]
                if ids_groups
                else []
            )
            documents = (
                document_groups[0]
                if document_groups
                else []
            )
            metadatas = (
                metadata_groups[0]
                if metadata_groups
                else []
            )
            distances = (
                distance_groups[0]
                if distance_groups
                else []
            )

            for (
                evidence_id,
                document,
                metadata,
                distance,
            ) in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=False,
            ):
                score = distance_to_score(
                    distance
                )

                if (
                    score
                    < self.min_relevance_score
                ):
                    continue

                item = make_retrieved_evidence(
                    evidence_id=evidence_id,
                    content=document or "",
                    metadata=metadata,
                    retrieval_method=(
                        "semantic"
                    ),
                    score=score,
                )

                results.append(item)

        return results

    @staticmethod
    def merge_results(
        exact_results: list[RetrievedEvidence],
        semantic_results: list[RetrievedEvidence],
    ) -> list:
        """
        Merge by evidence ID.

        Exact-ID retrieval always takes priority over semantic retrieval.
        """

        merged: dict[
            str,
            RetrievedEvidence,
        ] = {}

        for item in semantic_results:
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

        for item in exact_results:
            merged[
                item.evidence_id
                ] = item

        return sorted(
            merged.values(),
            key=lambda item: (
                0
                if item.retrieval_method
                == "exact_id"
                else 1,
                -item.score,
                item.evidence_id,
            ),
        )

    def retrieve(
        self,
        query: str,
        *,
        include_semantic: bool = True,
    ) -> RetrievalResult:
        """Run the Phase 10 retrieval pipeline."""

        query = query.strip()

        if not query:
            raise ValueError(
                "Query must not be empty."
            )

        identifiers = analyze_query_ids(
            query
        )

        exact_results = (
            self.exact_id_retrieval(
                identifiers
            )
        )

        semantic_results: list[
            RetrievedEvidence
        ] = []

        if include_semantic:
            semantic_results = (
                self.semantic_retrieval(
                    query
                )
            )

        combined = self.merge_results(
            exact_results,
            semantic_results,
        )

        limited = combined[
            :self.final_context_limit
        ]

        ranked = tuple(
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
                rank=index,
            )
            for index, item in enumerate(
                limited,
                start=1,
            )
        )

        exact_ids = {
            item.evidence_id
            for item in exact_results
        }

        semantic_ids = {
            item.evidence_id
            for item in semantic_results
        }

        return RetrievalResult(
            query=query,
            identifiers=identifiers,
            evidence=ranked,
            exact_match_count=len(
                exact_ids
            ),
            semantic_match_count=len(
                semantic_ids
            ),
        )

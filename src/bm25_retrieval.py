"""
BM25 lexical retrieval for REQTRACE AI.

Phase 11A builds an in-memory BM25 index from validated
EvidenceRecord objects created by the Phase 8 corpus.

Responsibilities:
- engineering-aware tokenization
- BM25 indexing
- lexical Top-K retrieval
- stable evidence provenance

This module does not:
- generate embeddings
- access ChromaDB
- perform semantic retrieval
- perform traceability expansion
- perform rank fusion
- invoke an LLM
- calculate engineering assessments
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.corpus import EvidenceCorpus
from src.evidence import EvidenceRecord
from src.normalization import normalize_whitespace
from src.retrieval import RetrievedEvidence


TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*|\d+(?:\.\d+)*"
)


def tokenize_engineering_text(
    text: str,
) -> list[str]:
    """
    Tokenize engineering text.

    Important technical tokens are preserved.

    Examples:

        REQ-017
        OBS-023
        TC-102
        process_data
        COMMAND_INVALID_ID

    Compound identifiers are also expanded.

    Example:

        REQ-017

    produces:

        req-017
        req
        017
    """

    normalized = normalize_whitespace(
        text
    ).lower()

    tokens: list[str] = []

    for match in TOKEN_PATTERN.finditer(
        normalized
    ):
        token = match.group(0)

        if not token:
            continue

        tokens.append(token)

        if "-" in token or "_" in token:
            components = re.split(
                r"[-_]+",
                token,
            )

            for component in components:
                if component:
                    tokens.append(
                        component
                    )

    return tokens


def evidence_search_text(
    record: EvidenceRecord,
) -> str:
    """
    Create searchable lexical text from evidence and metadata.

    Only source-derived metadata is used.
    """

    metadata = record.metadata

    values = (
        record.content,
        metadata.evidence_id,
        metadata.document_name,
        metadata.document_revision,
        metadata.requirement_id,
        metadata.observation_id,
        metadata.test_case_id,
        metadata.result_status,
        metadata.file_name,
        metadata.function_name,
        metadata.language,
    )

    searchable_parts: list[str] = []

    for value in values:
        if value is None:
            continue

        searchable_parts.append(
            str(value)
        )

    return "\n".join(
        searchable_parts
    )


@dataclass(frozen=True)
class BM25SearchResult:
    """One BM25-ranked result."""

    evidence: EvidenceRecord
    raw_score: float
    normalized_score: float
    rank: int


class BM25EvidenceIndex:
    """In-memory BM25 index over validated evidence."""

    def __init__(
        self,
        records: tuple[
            EvidenceRecord,
            ...
        ]
        | list[EvidenceRecord],
    ) -> None:

        if not records:
            raise ValueError(
                "BM25 requires at least one evidence record."
            )

        self.records = tuple(
            records
        )

        tokenized_documents: list[
            list[str]
        ] = []

        for record in self.records:
            searchable_text = evidence_search_text(
                record
            )

            tokens = tokenize_engineering_text(
                searchable_text
            )

            if not tokens:
                raise ValueError(
                    "Evidence contains no searchable "
                    f"tokens: {record.metadata.evidence_id}"
                )

            tokenized_documents.append(
                tokens
            )

        self.tokenized_documents = tuple(
            tokenized_documents
        )

        self.index = BM25Okapi(
            tokenized_documents
        )

    @classmethod
    def from_corpus(
        cls,
        corpus: EvidenceCorpus,
    ) -> "BM25EvidenceIndex":
        """Build an index from the validated corpus."""

        return cls(
            corpus.records
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[
        BM25SearchResult,
        ...
    ]:
        """Search validated evidence using BM25."""

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Query must not be empty."
            )

        query_tokens = tokenize_engineering_text(
            clean_query
        )

        if not query_tokens:
            return ()

        raw_scores = self.index.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(
                len(self.records)
            ),
            key=lambda index: (
                -float(
                    raw_scores[index]
                ),
                self.records[
                    index
                ].metadata.evidence_id,
            ),
        )

        positive_indexes: list[int] = []

        for index in ranked_indexes:
            score = float(
                raw_scores[index]
            )

            if score <= 0.0:
                continue

            positive_indexes.append(
                index
            )

            if (
                len(positive_indexes)
                >= top_k
            ):
                break

        if not positive_indexes:
            return ()

        maximum_score = max(
            float(
                raw_scores[index]
            )
            for index in positive_indexes
        )

        results: list[
            BM25SearchResult
        ] = []

        for rank, index in enumerate(
            positive_indexes,
            start=1,
        ):
            raw_score = float(
                raw_scores[index]
            )

            normalized_score = (
                raw_score / maximum_score
                if maximum_score > 0.0
                else 0.0
            )

            results.append(
                BM25SearchResult(
                    evidence=(
                        self.records[index]
                    ),
                    raw_score=raw_score,
                    normalized_score=(
                        normalized_score
                    ),
                    rank=rank,
                )
            )

        return tuple(
            results
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        """
        Return BM25 results using the common Phase 10 retrieval model.
        """

        search_results = self.search(
            query,
            top_k=top_k,
        )

        retrieved: list[
            RetrievedEvidence
        ] = []

        for result in search_results:
            evidence = (
                result.evidence
            )

            retrieved.append(
                RetrievedEvidence(
                    evidence_id=(
                        evidence.metadata
                        .evidence_id
                    ),
                    document_type=(
                        evidence.metadata
                        .document_type
                        .value
                    ),
                    content=(
                        evidence.content
                    ),
                    metadata=(
                        evidence.metadata
                        .to_chroma_metadata()
                    ),
                    retrieval_method=(
                        "bm25"
                    ),
                    score=(
                        result
                        .normalized_score
                    ),
                    rank=result.rank,
                )
            )

        return retrieved
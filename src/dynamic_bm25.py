"""
Dynamic BM25 retrieval for REQTRACE AI Phase 16H.

The original REQTRACE AI BM25 index operates on the validated baseline
EvidenceCorpus.

Phase 16H dynamically ingested evidence may use:

- Markdown
- TXT
- DOCX
- C / C++
- Python
- dynamically added PDF documents

Dynamic records are stored in ingestion_manifest.json and persistent
ChromaDB.

This module provides a supplemental lexical index for those records.

Important small-corpus behavior
-------------------------------

BM25 IDF can be zero or negative when a term occurs in a very small
dynamic corpus, especially when the corpus contains only one document.

Example:

    dynamic corpus = [REQ-900]

    query = REQ-900

The only document contains the query term, but rank_bm25 may return a
non-positive BM25 score because document frequency equals corpus size.

REQTRACE AI therefore retains BM25 ranking where it produces useful
positive scores and uses deterministic token-overlap lexical scoring as
a fallback.

This fallback is:

- lexical only;
- deterministic;
- based on the exact same engineering tokenizer as baseline BM25;
- not semantic retrieval;
- not LLM generated;
- not an engineering assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.bm25_retrieval import (
    tokenize_engineering_text,
)

from src.ingestion_manifest import (
    load_ingestion_manifest,
)

from src.retrieval import (
    RetrievedEvidence,
)


class DynamicBM25Error(
    RuntimeError
):
    """Raised when dynamic lexical evidence cannot be indexed."""


@dataclass(frozen=True)
class DynamicLexicalRecord:
    """One manifest-backed dynamic lexical evidence record."""

    evidence_id: str
    document_type: str

    content: str
    metadata: dict[str, Any]

    def __post_init__(
        self,
    ) -> None:
        """Validate the lexical evidence record."""

        if not self.evidence_id.strip():

            raise DynamicBM25Error(
                "Dynamic lexical evidence_id "
                "must not be empty."
            )

        if not self.document_type.strip():

            raise DynamicBM25Error(
                "Dynamic lexical document_type "
                "must not be empty."
            )

        if not self.content.strip():

            raise DynamicBM25Error(
                "Dynamic lexical content "
                "must not be empty."
            )


def manifest_record_is_dynamic(
    record: dict[str, Any],
) -> bool:
    """Return whether a manifest record was dynamically ingested."""

    value = record.get(
        "dynamic_ingestion",
        False,
    )

    if isinstance(
        value,
        bool,
    ):
        return value

    return (
        str(
            value
        )
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


def dynamic_record_metadata(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Build retrieval metadata from one manifest record.

    content is excluded because RetrievedEvidence carries content
    separately.
    """

    return {
        str(key): value

        for key, value
        in record.items()

        if (
            key != "content"
            and value is not None
        )
    }


def dynamic_search_text(
    record: DynamicLexicalRecord,
) -> str:
    """
    Build engineering-aware searchable lexical text.

    Include source-derived engineering metadata so identifiers,
    filenames, functions, sections, and traceability references are
    searchable.
    """

    metadata = record.metadata

    field_names = (
        "evidence_id",

        "document_name",
        "relative_path",

        "document_type",
        "document_category",

        "section_name",

        "requirement_id",
        "requirement_ids",

        "observation_id",
        "observation_ids",

        "test_case_id",
        "test_case_ids",

        "result_status",

        "file_name",
        "function_name",
        "class_name",

        "language",
        "parsing_method",
    )

    values: list[str] = [
        record.content
    ]

    for field_name in field_names:

        value = metadata.get(
            field_name
        )

        if value is None:
            continue

        text = str(
            value
        ).strip()

        if text:

            values.append(
                text
            )

    return "\n".join(
        values
    )


def dynamic_record_from_manifest(
    record: dict[str, Any],
) -> DynamicLexicalRecord | None:
    """Convert one dynamic manifest record into lexical evidence."""

    if not manifest_record_is_dynamic(
        record
    ):

        return None

    evidence_id = (
        str(
            record.get(
                "evidence_id",
                "",
            )
        )
        .strip()
    )

    document_type = (
        str(
            record.get(
                "document_type",
                "",
            )
        )
        .strip()
    )

    content = (
        str(
            record.get(
                "content",
                "",
            )
        )
        .strip()
    )

    if (
        not evidence_id
        or not document_type
        or not content
    ):

        raise DynamicBM25Error(
            "Dynamic manifest record is missing "
            "evidence_id, document_type, or content."
        )

    return DynamicLexicalRecord(
        evidence_id=evidence_id,
        document_type=document_type,
        content=content,
        metadata=(
            dynamic_record_metadata(
                record
            )
        ),
    )


def load_dynamic_lexical_records(
    manifest_path: str | Path,
) -> tuple[
    DynamicLexicalRecord,
    ...
]:
    """Load current dynamically ingested records from the manifest."""

    manifest = (
        load_ingestion_manifest(
            manifest_path
        )
    )

    output: list[
        DynamicLexicalRecord
    ] = []

    seen_ids: set[str] = set()

    for record in manifest.records:

        lexical_record = (
            dynamic_record_from_manifest(
                record
            )
        )

        if lexical_record is None:
            continue

        if (
            lexical_record.evidence_id
            in seen_ids
        ):

            raise DynamicBM25Error(
                "Duplicate dynamic evidence ID: "
                f"{lexical_record.evidence_id}"
            )

        seen_ids.add(
            lexical_record.evidence_id
        )

        output.append(
            lexical_record
        )

    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.document_type,
                str(
                    item.metadata.get(
                        "document_name",
                        "",
                    )
                ).lower(),
                item.evidence_id,
            ),
        )
    )


# =====================================================================
# Lexical scoring
# =====================================================================

def lexical_overlap_score(
    *,
    query_tokens: list[str],
    document_tokens: list[str],
) -> float:
    """
    Calculate deterministic query-token coverage.

    Example:

        query:
            REQ-900

        tokenizer:
            req-900
            req
            900

        document contains all three:
            overlap = 3 / 3 = 1.0

    Repeated query tokens do not increase the score.
    """

    query_set = set(
        query_tokens
    )

    document_set = set(
        document_tokens
    )

    if not query_set:
        return 0.0

    matched = (
        query_set
        & document_set
    )

    if not matched:
        return 0.0

    return (
        len(
            matched
        )
        / len(
            query_set
        )
    )


def normalized_positive_bm25_scores(
    raw_scores,
) -> tuple[float, ...]:
    """
    Normalize only positive BM25 scores.

    Zero and negative values are treated as unavailable because they
    are not useful for REQTRACE AI's normalized lexical relevance
    representation.
    """

    positive_scores = [
        max(
            0.0,
            float(
                score
            ),
        )
        for score in raw_scores
    ]

    maximum = max(
        positive_scores,
        default=0.0,
    )

    if maximum <= 0.0:

        return tuple(
            0.0
            for _ in positive_scores
        )

    return tuple(
        score
        / maximum

        for score in positive_scores
    )


# =====================================================================
# Dynamic index
# =====================================================================

class DynamicBM25Index:
    """
    Supplemental lexical index over current dynamic-manifest evidence.

    Ranking strategy:

    1. Generate ordinary BM25 scores.
    2. Normalize positive BM25 values.
    3. Calculate deterministic query-token overlap.
    4. Use the stronger lexical signal.

    Token overlap primarily protects correct lexical behavior when the
    dynamic corpus is too small for useful BM25 IDF statistics.
    """

    def __init__(
        self,
        records: tuple[
            DynamicLexicalRecord,
            ...
        ],
    ) -> None:
        """Build dynamic lexical index."""

        self.records = tuple(
            records
        )

        self.index: (
            BM25Okapi
            | None
        ) = None

        self.tokenized_documents: tuple[
            list[str],
            ...
        ] = ()

        if not self.records:

            return

        tokenized_documents: list[
            list[str]
        ] = []

        for record in self.records:

            searchable_text = (
                dynamic_search_text(
                    record
                )
            )

            tokens = (
                tokenize_engineering_text(
                    searchable_text
                )
            )

            if not tokens:

                raise DynamicBM25Error(
                    "Dynamic evidence contains no "
                    "searchable tokens: "
                    f"{record.evidence_id}"
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
    def from_manifest(
        cls,
        manifest_path: str | Path,
    ) -> "DynamicBM25Index":
        """Build dynamic lexical index directly from manifest."""

        return cls(
            load_dynamic_lexical_records(
                manifest_path
            )
        )

    @property
    def record_count(
        self,
    ) -> int:
        """Return number of of dynamically indexed evidence records."""

        return len(
            self.records
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[
        RetrievedEvidence
    ]:
        """Retrieve lexical evidence from the current dynamic corpus."""

        if top_k < 1:

            raise ValueError(
                "top_k must be at least 1."
            )

        clean_query = (
            query.strip()
        )

        if not clean_query:

            raise ValueError(
                "Query must not be empty."
            )

        if (
            self.index is None
            or not self.records
        ):

            return []

        query_tokens = (
            tokenize_engineering_text(
                clean_query
            )
        )

        if not query_tokens:

            return []

        raw_bm25_scores = (
            self.index.get_scores(
                query_tokens
            )
        )

        normalized_bm25 = (
            normalized_positive_bm25_scores(
                raw_bm25_scores
            )
        )

        combined_scores: list[
            float
        ] = []

        for index, document_tokens in enumerate(
            self.tokenized_documents
        ):

            overlap = (
                lexical_overlap_score(
                    query_tokens=(
                        query_tokens
                    ),

                  document_tokens=(
                        document_tokens
                    ),
                )
            )

            bm25_score = (
                normalized_bm25[
                    index
                ]
            )

            # For large dynamic corpora BM25 normally dominates.
            # For tiny corpora lexical coverage prevents mathematically
            # non-positive IDF from hiding an obvious exact lexical
            # match.
            combined_scores.append(
                max(
                    bm25_score,
                    overlap,
                )
            )

        ranked_indexes = sorted(
            range(
                len(
                    self.records
                )
            ),

            key=lambda index: (
                -combined_scores[
                    index
                ],

                self.records[
                    index
                ].evidence_id,
            ),
        )

        positive_indexes = [
            index
            for index in ranked_indexes

            if (
                combined_scores[
                    index
                ]
                > 0.0
            )
        ][
            :top_k
        ]

        if not positive_indexes:

            return []

        output: list[
            RetrievedEvidence
        ] = []

        for rank, index in enumerate(
            positive_indexes,
            start=1,
        ):

            record = (
                self.records[
                    index
                ]
            )

            output.append(
                RetrievedEvidence(
                    evidence_id=(
                        record.evidence_id
                    ),

                    document_type=(
                        record.document_type
                    ),

                    content=(
                        record.content
                    ),

                    metadata=dict(
                        record.metadata
                    ),

                    retrieval_method=(
                        "bm25_dynamic"
                    ),

                    score=(
                        combined_scores[
                            index
                        ]
                    ),

                    rank=rank,
                )
            )

        return output
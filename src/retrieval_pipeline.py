"""
Reusable Phase 11 retrieval pipeline.

Combines:

- BM25 lexical retrieval
- Dense semantic retrieval
- Exact-ID retrieval
- Reciprocal Rank Fusion
- Entity-coherent candidate filtering
- Controlled traceability expansion
- Final bounded context assembly

This module performs no LLM generation.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Iterable

import chromadb

from config import Settings

from src.bm25_retrieval import (
    BM25EvidenceIndex,
)

from src.context_builder import (
    FinalEvidenceContext,
    build_final_context,
)

from src.corpus import (
    build_project_corpus,
)

from src.embeddings import (
    EmbeddingClientConfig,
    OllamaEmbeddingClient,
)

from src.fusion import (
    normalize_fused_results,
    reciprocal_rank_fusion,
)

from src.retrieval import (
    EvidenceRetriever,
    RetrievedEvidence,
    analyze_query_ids,
)

from src.traceability import (
    expand_engineering_family,
)

from src.bm25_combined import (
    merge_bm25_results,
)

from src.dynamic_bm25 import (
    DynamicBM25Index,
)


# =====================================================================
# Explicit engineering identifiers
# =====================================================================

ENGINEERING_ID_PATTERN = re.compile(
    r"\b(?P<prefix>REQ|OBS|TC)"
    r"[\s_-]?"
    r"(?P<number>\d+)\b",
    re.IGNORECASE,
)


def normalize_engineering_id(
    prefix: str,
    number: str,
) -> str:
    """Normalize an explicit engineering identifier."""

    return (
        f"{prefix.upper()}-{number}"
    )


def explicit_query_ids(
    query: str,
) -> tuple[str, ...]:
    """
    Extract explicit REQ, OBS and TC identifiers from the query.

    Examples:

        REQ-900
        OBS_023
        TC 501
    """

    output: list[str] = []
    seen: set[str] = set()

    for match in (
        ENGINEERING_ID_PATTERN
        .finditer(
            query
        )
    ):
        value = normalize_engineering_id(
            match.group(
                "prefix"
            ),
            match.group(
                "number"
            ),
        )

        if value in seen:
            continue

        seen.add(
            value
        )

        output.append(
            value
        )

    return tuple(
        output
    )


# =====================================================================
# Evidence identifier extraction
# =====================================================================

def normalize_metadata_id(
    value: object,
) -> str | None:
    """Normalize one metadata engineering identifier."""

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    match = (
        ENGINEERING_ID_PATTERN
        .fullmatch(
            text
        )
    )

    if match is None:
        return text.upper()

    return normalize_engineering_id(
        match.group(
            "prefix"
        ),
        match.group(
            "number"
        ),
    )


def metadata_id_values(
    value: object,
) -> tuple[str, ...]:
    """
    Extract identifiers from scalar or pipe-delimited metadata.

    Dynamic ingestion stores plural identifiers such as:

        requirement_ids = REQ-900|REQ-901
        test_case_ids   = TC-900
    """

    if value is None:
        return ()

    if isinstance(
        value,
        (
            tuple,
            list,
            set,
        ),
    ):
        raw_values = [
            str(item)
            for item in value
        ]

    else:
        raw_values = re.split(
            r"[|,;]",
            str(
                value
            ),
        )

    output: list[str] = []

    seen: set[str] = set()

    for raw_value in raw_values:
        normalized = (
            normalize_metadata_id(
                raw_value
            )
        )

        if (
            normalized is None
            or normalized in seen
        ):
            continue

        seen.add(
            normalized
        )

        output.append(
            normalized
        )

    return tuple(
        output
    )


def evidence_engineering_ids(
    evidence: RetrievedEvidence,
) -> tuple[str, ...]:
    """
    Return all structured engineering IDs carried by evidence.

    Structured metadata has authority over semantic similarity.

    Free-text content is deliberately not used to establish entity
    coherence.
    """

    metadata = dict(
        evidence.metadata
    )

    field_names = (
        "requirement_id",
        "requirement_ids",
        "observation_id",
        "observation_ids",
        "test_case_id",
        "test_case_ids",
    )

    output: list[str] = []

    seen: set[str] = set()

    for field_name in field_names:
        for identifier in metadata_id_values(
            metadata.get(
                field_name
            )
        ):
            if identifier in seen:
                continue

            seen.add(
                identifier
            )

            output.append(
                identifier
            )

    return tuple(
        output
    )


def collect_anchor_family_ids(
    exact_results: Iterable[
        RetrievedEvidence
    ],
    requested_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Build the authoritative engineering family from exact evidence.

    Example:

        Query:
            REQ-900

        Exact FRD metadata:
            requirement_id  = REQ-900
            test_case_ids   = TC-900

        Family:
            REQ-900
            TC-900

    A future Test Report for TC-900 can therefore join the context,
    while OBS-052 / REQ-041 cannot.
    """

    output: list[str] = list(
        requested_ids
    )

    seen: set[str] = set(
        requested_ids
    )

    for evidence in exact_results:
        for identifier in (
            evidence_engineering_ids(
                evidence
            )
        ):
            if identifier in seen:
                continue

            seen.add(
                identifier
            )

            output.append(
                identifier
            )

    return tuple(
        output
    )


# =====================================================================
# Entity-coherence filtering
# =====================================================================

def evidence_is_entity_coherent(
    evidence: RetrievedEvidence,
    *,
    authoritative_ids: tuple[str, ...],
) -> bool:
    """
    Return whether candidate evidence belongs to the exact entity family.

    A candidate is coherent when its structured metadata shares at
    least one authoritative engineering identifier.

    Examples:

        REQ-900 candidate
            IDs: REQ-900
            -> accepted

        TC-900 Test candidate
            IDs: REQ-900, TC-900
            -> accepted

        OBS-052 candidate
            IDs: REQ-041, OBS-052
            -> rejected
    """

    if not authoritative_ids:
        return True

    candidate_ids = set(
        evidence_engineering_ids(
            evidence
        )
    )

    if not candidate_ids:
        return False

    return bool(
        candidate_ids
        & set(
            authoritative_ids
        )
    )


def filter_entity_coherent_results(
    results: Iterable[
        RetrievedEvidence
    ],
    *,
    authoritative_ids: tuple[str, ...],
    protected_ids: set[str],
) -> list[
    RetrievedEvidence
]:
    """
    Retain exact protected records and coherent related evidence.

    protected_ids are always retained because they were returned by
    exact-ID retrieval.

    Semantic, lexical and expanded records must share a structured
    authoritative ID.
    """

    output: list[
        RetrievedEvidence
    ] = []

    seen: set[str] = set()

    for evidence in results:
        if evidence.evidence_id in seen:
            continue

        if (
            evidence.evidence_id
            in protected_ids
            or evidence_is_entity_coherent(
                evidence,
                authoritative_ids=(
                    authoritative_ids
                ),
            )
        ):
            seen.add(
                evidence.evidence_id
            )

            output.append(
                evidence
            )

    return output


# =====================================================================
# Retrieval pipeline
# =====================================================================

def retrieve_final_context(
    query: str,
    settings: Settings,
    base_directory: str | Path,
) -> FinalEvidenceContext:
    """Run the validated Phase 11 retrieval pipeline."""

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "Query must not be empty."
        )

    base_path = (
        Path(
            base_directory
        )
        .expanduser()
        .resolve()
    )

    # -------------------------------------------------------------
    # Build the current project corpus and BM25 index.
    #
    # Dynamic Chroma evidence is immediately available to dense and
    # exact retrieval. The existing project corpus remains the BM25
    # source until runtime corpus refresh is added.
    # -------------------------------------------------------------

    corpus = build_project_corpus(
        base_path
    )

    # -------------------------------------------------------------
    # Baseline lexical evidence.
    # -------------------------------------------------------------

    bm25_index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    baseline_bm25_results = (
        bm25_index.retrieve(
            clean_query,
            top_k=(
                settings.bm25_top_k
            ),
        )
    )

    # -------------------------------------------------------------
    # Phase 16H dynamic lexical evidence.
    #
    # The manifest contains the current dynamic evidence after every
    # successful synchronization. Building this lightweight index for
    # each production request guarantees that newly indexed, changed,
    # and deleted documents appear immediately without an application
    # restart.
    # -------------------------------------------------------------

    dynamic_manifest_path = (
        base_path
        / "data"
        / "ingestion_manifest.json"
    )

    dynamic_bm25_index = (
        DynamicBM25Index
        .from_manifest(
            dynamic_manifest_path
        )
    )

    dynamic_bm25_results = (
        dynamic_bm25_index.retrieve(
            clean_query,
            top_k=(
                settings.bm25_top_k
            ),
        )
    )

    # -------------------------------------------------------------
    # Merge baseline and dynamic lexical channels.
    #
    # Both are represented as the same RetrievedEvidence model before RRF.
    # -------------------------------------------------------------

    bm25_results = (
        merge_bm25_results(
            baseline_bm25_results,
            dynamic_bm25_results,
            top_k=(
                settings.bm25_top_k
            ),
        )
    )

    # -------------------------------------------------------------
    # Open the existing persistent Chroma database.
    # -------------------------------------------------------------

    client = (
        chromadb
        .PersistentClient(
            path=str(
                settings
                .chroma_db_path
            )
        )
    )

    collection_names = {
        "frd": (
            settings
            .frd_collection_name
        ),

        "observation": (
            settings
            .observation_collection_name
        ),

        "code": (
            settings
            .code_collection_name
        ),

        "test": (
            settings
            .test_collection_name
        ),

        "review": (
            settings
            .reviews_collection_name
        ),
    }

    identifiers = analyze_query_ids(
        clean_query
    )

    requested_ids = explicit_query_ids(
        clean_query
    )

    embedding_config = (
        EmbeddingClientConfig(
            base_url=(
                settings
                .ollama_base_url
            ),

            model=(
                settings
                .ollama_embed_model
            ),

            connect_timeout_seconds=(
                settings
                .ollama_connect_timeout_seconds
            ),

            read_timeout_seconds=(
                settings
                .llm_timeout_seconds
            ),
        )
    )

    with OllamaEmbeddingClient(
        embedding_config
    ) as embedding_client:

        dense = EvidenceRetriever(
            client=client,

            collection_names=(
                collection_names
            ),

            embedding_client=(
                embedding_client
            ),

            vector_top_k=(
                settings
                .vector_top_k
            ),

            final_context_limit=(
                settings
                .fused_top_k
            ),

            min_relevance_score=(
                settings
                .min_relevance_score
            ),
        )

        exact_results = (
            dense.exact_id_retrieval(
                identifiers
            )
        )

        semantic_results = (
            dense.semantic_retrieval(
                clean_query
            )
        )

    # -------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -------------------------------------------------------------

    fused = reciprocal_rank_fusion(
        {
            "bm25": (
                bm25_results
            ),

            "semantic": (
                semantic_results
            ),
        },
        rank_constant=60,
    )

    rrf_results = (
        normalize_fused_results(
            fused,
            top_k=(
                settings
                .fused_top_k
            ),
        )
    )

    # -------------------------------------------------------------
    # Entity-Coherence Gate
    #
    # Apply only when:
    #
    # - the query contains explicit engineering IDs; and
    # - exact evidence exists for those IDs.
    #
    # Natural-language queries without explicit IDs retain the original
    # broad hybrid-retrieval behavior.
    # -------------------------------------------------------------

    protected_exact_ids = {
        evidence.evidence_id
        for evidence in exact_results
    }

    if (
        requested_ids
        and exact_results
    ):
        authoritative_ids = (
            collect_anchor_family_ids(
                exact_results,
                requested_ids,
            )
        )

        coherent_rrf_results = (
            filter_entity_coherent_results(
                rrf_results,
                authoritative_ids=(
                    authoritative_ids
                ),
                protected_ids=(
                    protected_exact_ids
                ),
            )
        )

    else:
        authoritative_ids = ()

        coherent_rrf_results = list(
            rrf_results
        )

    # -------------------------------------------------------------
    # Traceability anchors
    # -------------------------------------------------------------

    if exact_results:
        anchors = list(
            exact_results
        )

    elif coherent_rrf_results:
        anchors = [
            coherent_rrf_results[0]
        ]

    else:
        anchors = []

    expansion = expand_engineering_family(
        corpus,
        anchors,
        maximum_iterations=3,
        maximum_results=20,
    )

    # -------------------------------------------------------------
    # Filter expanded evidence after expansion.
    #
    # This prevents an unrelated evidence family from becoming
    # decision-authoritative merely because it is semantically similar.
    # -------------------------------------------------------------

    if authoritative_ids:
        coherent_expansion = (
            filter_entity_coherent_results(
                expansion.expanded,
                authoritative_ids=(
                    authoritative_ids
                ),
                protected_ids=(
                    protected_exact_ids
                ),
            )
        )

    else:
        coherent_expansion = list(
            expansion.expanded
        )

    # -------------------------------------------------------------
    # Final bounded engineering context
    # -------------------------------------------------------------

    return build_final_context(
        exact_results=list(
            exact_results
        ),

        traceability_results=(
            coherent_expansion
        ),

        rrf_results=(
            coherent_rrf_results
        ),

        context_limit=(
            settings
            .final_context_limit
        ),
    )
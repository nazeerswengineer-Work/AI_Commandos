"""
Tests for clean Phase 11A BM25 lexical retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bm25_retrieval import (
    BM25EvidenceIndex,
    evidence_search_text,
    tokenize_engineering_text,
)
from src.corpus import (
    build_project_corpus,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def test_engineering_tokenization() -> None:
    tokens = tokenize_engineering_text(
        "REQ-017 OBS_023 TC-102 "
        "process_data COMMAND_INVALID_ID"
    )

    assert "req-017" in tokens
    assert "req" in tokens
    assert "017" in tokens

    assert "obs_023" in tokens
    assert "obs" in tokens
    assert "023" in tokens

    assert "tc-102" in tokens

    assert "process_data" in tokens

    assert (
        "command_invalid_id"
        in tokens
    )


def test_empty_query_is_rejected() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    with pytest.raises(
        ValueError
    ):
        index.search(
            "   "
        )


def test_invalid_top_k_is_rejected() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    with pytest.raises(
        ValueError
    ):
        index.search(
            "REQ-017",
            top_k=0,
        )


def test_search_text_contains_metadata() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    record = next(
        item
        for item in corpus.records
        if (
            item.metadata.requirement_id
            == "REQ-017"
        )
    )

    searchable = evidence_search_text(
        record
    )

    assert "REQ-017" in searchable
    assert (
        record.metadata.document_name
        in searchable
    )


def test_bm25_req_017() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    results = index.search(
        "REQ-017",
        top_k=5,
    )

    assert results

    assert any(
        item.evidence
        .metadata
        .requirement_id
        == "REQ-017"
        for item in results
    )


def test_bm25_obs_023() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    results = index.search(
        "OBS-023",
        top_k=5,
    )

    assert results

    assert any(
        item.evidence
        .metadata
        .observation_id
        == "OBS-023"
        for item in results
    )


def test_bm25_command_identifier_scenario() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    results = index.search(
        "command identifier 17 rejected",
        top_k=5,
    )

    assert results

    evidence_ids = tuple(
        item.evidence
        .metadata
        .evidence_id
        for item in results
    )

    assert any(
        "REQ-021" in evidence_id
        for evidence_id
        in evidence_ids
    )

    assert any(
        item.evidence
        .metadata
        .function_name
        == "validate_command"
        for item in results
    )


def test_bm25_revision_conflict_scenario() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    results = index.search(
        "REQ-050 alert reset",
        top_k=8,
    )

    req_050_records = [
        item.evidence
        for item in results
        if (
            item.evidence
            .metadata
            .requirement_id
            == "REQ-050"
            and item.evidence
            .metadata
            .document_type
            .value
            == "frd"
        )
    ]

    revisions = {
        record.metadata
        .document_revision
        for record
        in req_050_records
    }

    assert "Rev01" in revisions
    assert "Rev02" in revisions


def test_bm25_retrieve_common_model() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = (
        BM25EvidenceIndex
        .from_corpus(
            corpus
        )
    )

    results = index.retrieve(
        "null input pointer",
        top_k=5,
    )

    assert results

    assert all(
        item.retrieval_method
        == "bm25"
        for item in results
    )

    assert all(
        0.0
        <= item.score
        <= 1.0
        for item in results
    )

    assert tuple(
        item.rank
        for item in results
    ) == tuple(
        range(
            1,
            len(results) + 1,
        )
    )
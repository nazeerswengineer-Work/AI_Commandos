from __future__ import annotations

from pathlib import Path

import pytest

from src.corpus import (
    CorpusValidationError,
    build_project_corpus,
    detect_duplicate_content,
    validate_unique_evidence_ids,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def test_project_corpus_builds() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    assert corpus.record_count > 0
    assert corpus.records


def test_corpus_contains_all_document_types() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    document_types = {
        record.metadata.document_type.value
        for record in corpus.records
    }

    assert document_types == {
        "frd",
        "observation",
        "code",
        "test",
        "review",
    }


def test_evidence_ids_are_unique() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    assert len(corpus.evidence_ids) == len(
        set(corpus.evidence_ids)
    )


def test_req_017_has_cross_document_evidence() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    matching = [
        record
        for record in corpus.records
        if record.metadata.requirement_id
        == "REQ-017"
    ]

    types = {
        record.metadata.document_type.value
        for record in matching
    }

    assert "frd" in types
    assert "observation" in types
    assert "code" in types
    assert "test" in types


def test_obs_023_has_cross_document_evidence() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    matching = [
        record
        for record in corpus.records
        if record.metadata.observation_id
        == "OBS-023"
    ]

    types = {
        record.metadata.document_type.value
        for record in matching
    }

    assert "observation" in types
    assert "code" in types
    assert "review" in types


def test_req_050_revision_variants_are_preserved() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    records = [
        record
        for record in corpus.records
        if (
            record.metadata.document_type.value
            == "frd"
            and record.metadata.requirement_id
            == "REQ-050"
        )
    ]

    revisions = {
        record.metadata.document_revision
        for record in records
    }

    assert revisions == {
        "Rev01",
        "Rev02",
    }


def test_obs_052_reference_is_not_rewritten() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    observation = next(
        record
        for record in corpus.records
        if (
            record.metadata.document_type.value
            == "observation"
            and record.metadata.observation_id
            == "OBS-052"
        )
    )

    assert (
        observation.metadata.requirement_id
        == "REQ-041"
    )


def test_duplicate_detection_runs() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    duplicates = detect_duplicate_content(
        list(corpus.records)
    )

    assert isinstance(
        duplicates,
        tuple,
    )


def test_corpus_does_not_create_vector_store() -> None:
    chroma_directory = (
        BASE_DIR / "chroma_db"
    )

    before = (
        set(chroma_directory.iterdir())
        if chroma_directory.exists()
        else set()
    )

    build_project_corpus(
        BASE_DIR
    )

    after = (
        set(chroma_directory.iterdir())
        if chroma_directory.exists()
        else set()
    )

    assert after == before
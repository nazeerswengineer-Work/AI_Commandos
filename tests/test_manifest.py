from __future__ import annotations

import json
from pathlib import Path

from src.corpus import build_project_corpus
from src.manifest import (
    build_ingestion_manifest,
    build_traceability_index,
    write_corpus_artifacts,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def test_manifest_matches_corpus_count() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    manifest = build_ingestion_manifest(
        corpus
    )

    assert (
        manifest["record_count"]
        == corpus.record_count
    )

    assert len(
        manifest["records"]
    ) == corpus.record_count


def test_manifest_contains_all_types() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    manifest = build_ingestion_manifest(
        corpus
    )

    assert set(
        manifest["record_counts_by_type"]
    ) == {
        "frd",
        "observation",
        "code",
        "test",
        "review",
    }


def test_traceability_contains_req_017() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = build_traceability_index(
        corpus
    )

    assert "REQ-017" in (
        index["requirements"]
    )

    assert len(
        index["requirements"]["REQ-017"]
    ) >= 4


def test_traceability_contains_obs_023() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = build_traceability_index(
        corpus
    )

    assert "OBS-023" in (
        index["observations"]
    )

    evidence_ids = (
        index["observations"]["OBS-023"]
    )

    assert any(
        evidence_id.startswith("OBS::")
        for evidence_id in evidence_ids
    )

    assert any(
        evidence_id.startswith("CODE::")
        for evidence_id in evidence_ids
    )

    assert any(
        evidence_id.startswith("REVIEW::")
        for evidence_id in evidence_ids
    )


def test_traceability_preserves_obs_052_req_041() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    index = build_traceability_index(
        corpus
    )

    obs_052_records = [
        record
        for record in corpus.records
        if (
            record.metadata.observation_id
            == "OBS-052"
            and record.metadata.document_type.value
            == "observation"
        )
    ]

    assert len(obs_052_records) == 1

    assert (
        obs_052_records[0]
        .metadata
        .requirement_id
        == "REQ-041"
    )


def test_corpus_artifacts_write_valid_json(
    tmp_path: Path,
) -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    manifest_path, traceability_path = (
        write_corpus_artifacts(
            corpus,
            tmp_path,
        )
    )

    assert manifest_path.exists()
    assert traceability_path.exists()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    traceability = json.loads(
        traceability_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["record_count"]
        == corpus.record_count
    )

    assert "requirements" in traceability
    assert "observations" in traceability
"""
Tests for Phase 11C controlled traceability expansion.
"""

from __future__ import annotations

from pathlib import Path

from src.corpus import (
    build_project_corpus,
)
from src.retrieval import (
    RetrievedEvidence,
)
from src.traceability import (
    TraceabilityKeys,
    collect_traceability_keys,
    expand_engineering_family,
    record_matches_keys,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def make_anchor(
    *,
    evidence_id: str,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
) -> RetrievedEvidence:
    """Create one controlled test anchor."""

    metadata: dict[str, str] = {}

    if requirement_id:
        metadata[
            "requirement_id"
        ] = requirement_id

    if observation_id:
        metadata[
            "observation_id"
        ] = observation_id

    if test_case_id:
        metadata[
            "test_case_id"
        ] = test_case_id

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type="frd",
        content="Anchor content",
        metadata=metadata,
        retrieval_method="rrf",
        score=1.0,
        rank=1,
    )


def test_collect_traceability_keys() -> None:
    anchors = [
        make_anchor(
            evidence_id="A",
            requirement_id="REQ-021",
            observation_id="OBS-031",
            test_case_id="TC-201",
        )
    ]

    keys = collect_traceability_keys(
        anchors
    )

    assert keys.requirement_ids == (
        "REQ-021",
    )

    assert keys.observation_ids == (
        "OBS-031",
    )

    assert keys.test_case_ids == (
        "TC-201",
    )


def test_req_021_expands_to_complete_family() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    anchors = [
        make_anchor(
            evidence_id="ANCHOR-REQ-021",
            requirement_id="REQ-021",
        )
    ]

    result = expand_engineering_family(
        corpus,
        anchors,
        maximum_iterations=3,
        maximum_results=20,
    )

    evidence_types = {
        item.document_type
        for item in result.all_evidence
    }

    assert "frd" in evidence_types
    assert "observation" in evidence_types
    assert "code" in evidence_types
    assert "test" in evidence_types
    assert "review" in evidence_types

    assert (
        "REQ-021"
        in result.keys.requirement_ids
    )

    assert (
        "OBS-031"
        in result.keys.observation_ids
    )

    assert (
        "TC-201"
        in result.keys.test_case_ids
    )


def test_req_021_expansion_excludes_unrelated_families() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    anchors = [
        make_anchor(
            evidence_id="ANCHOR-REQ-021",
            requirement_id="REQ-021",
        )
    ]

    result = expand_engineering_family(
        corpus,
        anchors,
    )

    assert (
        "REQ-050"
        not in result.keys.requirement_ids
    )

    assert (
        "REQ-017"
        not in result.keys.requirement_ids
    )

    assert (
        "OBS-023"
        not in result.keys.observation_ids
    )

    assert (
        "TC-102"
        not in result.keys.test_case_ids
    )


def test_req_050_preserves_both_revisions() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    anchors = [
        make_anchor(
            evidence_id="ANCHOR-REQ-050",
            requirement_id="REQ-050",
        )
    ]

    result = expand_engineering_family(
        corpus,
        anchors,
    )

    req_050_frd = [
    item
    for item in result.expanded
        if (
            item.document_type == "frd"
            and item.metadata.get(
                "requirement_id"
            )
            == "REQ-050"
        )
    ]

    revisions = {
        item.metadata.get(
            "document_revision"
        )
        for item in req_050_frd
    }

    assert revisions == {
        "Rev01",
        "Rev02",
    }


def test_req_050_does_not_pull_obs_052() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    anchors = [
        make_anchor(
            evidence_id="ANCHOR-REQ-050",
            requirement_id="REQ-050",
        )
    ]

    result = expand_engineering_family(
        corpus,
        anchors,
    )

    assert (
        "OBS-052"
        not in result.keys.observation_ids
    )

    assert not any(
        item.metadata.get(
            "function_name"
        )
        == "process_alert"
        for item in result.all_evidence
    )


def test_obs_052_remains_linked_to_req_041() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    observation = next(
        record
        for record in corpus.records
        if (
            record.metadata
            .document_type
            .value
            == "observation"
            and record.metadata
            .observation_id
            == "OBS-052"
        )
    )

    anchor = RetrievedEvidence(
        evidence_id=(
            observation.metadata
            .evidence_id
        ),
        document_type="observation",
        content=observation.content,
        metadata=(
            observation.metadata
            .to_chroma_metadata()
        ),
        retrieval_method="exact_id",
        score=1.0,
        rank=1,
    )

    result = expand_engineering_family(
        corpus,
        [anchor],
    )

    assert (
        "REQ-041"
        in result.keys.requirement_ids
    )

    assert observation.metadata.requirement_id == (
        "REQ-041"
    )


def test_empty_anchor_returns_no_expansion() -> None:
    corpus = build_project_corpus(
        BASE_DIR
    )

    result = expand_engineering_family(
        corpus,
        [],
    )

    assert result.expanded == ()
    assert result.iterations == 0
    assert result.keys.is_empty is True
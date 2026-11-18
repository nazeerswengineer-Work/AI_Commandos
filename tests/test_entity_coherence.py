"""
Tests for explicit-ID entity-coherent context filtering.
"""

from __future__ import annotations

from src.retrieval import (
    RetrievedEvidence,
)

from src.retrieval_pipeline import (
    collect_anchor_family_ids,
    evidence_engineering_ids,
    evidence_is_entity_coherent,
    explicit_query_ids,
    filter_entity_coherent_results,
)


def evidence(
    evidence_id: str,
    document_type: str,
    metadata: dict,
) -> RetrievedEvidence:

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type=document_type,
        content="Synthetic evidence.",
        metadata=metadata,
        retrieval_method="test",
        score=1.0,
        rank=1,
    )


def test_explicit_query_id_extraction() -> None:

    assert (
        explicit_query_ids(
            "Compare REQ_900 with TC 900."
        )
        == (
            "REQ-900",
            "TC-900",
        )
    )


def test_anchor_family_includes_referenced_test() -> None:

    anchor = evidence(
        "DYN::REQ-900",
        "frd",
        {
            "requirement_id": "REQ-900",
            "test_case_ids": "TC-900",
        },
    )

    result = collect_anchor_family_ids(
        [
            anchor
        ],
        (
            "REQ-900",
        ),
    )

    assert result == (
        "REQ-900",
        "TC-900",
    )


def test_unrelated_observation_is_rejected() -> None:

    candidate = evidence(
        "OBS-052",
        "observation",
        {
            "requirement_id": "REQ-041",
            "observation_id": "OBS-052",
        },
    )

    assert (
        evidence_is_entity_coherent(
            candidate,
            authoritative_ids=(
                "REQ-900",
                "TC-900",
            ),
        )
        is False
    )


def test_related_future_test_is_accepted() -> None:

    candidate = evidence(
        "TEST-900",
        "test",
        {
            "requirement_id": "REQ-900",
            "test_case_id": "TC-900",
            "result_status": "PASS",
        },
    )

    assert (
        evidence_is_entity_coherent(
            candidate,
            authoritative_ids=(
                "REQ-900",
                "TC-900",
            ),
        )
        is True
    )


def test_exact_evidence_is_always_protected() -> None:

    exact = evidence(
        "EXACT-REQ-900",
        "frd",
        {
            "requirement_id": "REQ-900",
        },
    )

    output = filter_entity_coherent_results(
        [
            exact
        ],
        authoritative_ids=(
            "REQ-900",
        ),
        protected_ids={
            "EXACT-REQ-900"
        },
    )

    assert output == [
        exact
    ]


def test_plural_pipe_metadata_is_supported() -> None:

    item = evidence(
        "DYNAMIC",
        "frd",
        {
            "requirement_ids": (
                "REQ-900|REQ-901"
            ),
            "test_case_ids": (
                "TC-900|TC-901"
            ),
        },
    )

    assert (
        evidence_engineering_ids(
            item
        )
        == (
            "REQ-900",
            "REQ-901",
            "TC-900",
            "TC-901",
        )
    )
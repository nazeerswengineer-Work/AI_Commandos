"""
Tests for Phase 11D / 11E final evidence-context assembly.

These tests verify:

- final context limit
- exact-ID protection
- distinct FRD revision preservation
- traceability priority
- evidence deduplication
- core engineering evidence-type coverage
- Review as secondary evidence
- redundant exact records do not block missing core evidence
- one-based final context ranking
"""

from __future__ import annotations

from src.context_builder import (
    build_final_context,
    choose_diverse_additions,
)
from src.retrieval import RetrievedEvidence


def make_evidence(
    evidence_id: str,
    document_type: str,
    method: str,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
    document_revision: str | None = None,
) -> RetrievedEvidence:
    """Create one synthetic retrieval record for context-builder tests."""

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

    if document_revision:
        metadata[
            "document_revision"
        ] = document_revision

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type=document_type,
        content="Synthetic evidence",
        metadata=metadata,
        retrieval_method=method,
        score=1.0,
        rank=1,
    )


def test_context_limit_is_enforced() -> None:
    """Final context must never exceed the configured limit."""

    rrf = [
        make_evidence(
            evidence_id=f"E-{index}",
            document_type="frd",
            method="rrf",
        )
        for index in range(10)
    ]

    result = build_final_context(
        exact_results=[],
        traceability_results=[],
        rrf_results=rrf,
        context_limit=4,
    )

    assert len(
        result.evidence
    ) == 4


def test_exact_results_preserve_distinct_frd_revisions() -> None:
    """
    Distinct exact FRD revisions must survive final-context assembly.

    This protects the REQ-050 revision-conflict scenario.
    """

    exact = [
        make_evidence(
            evidence_id=(
                "FRD-REQ050-REV01"
            ),
            document_type="frd",
            method="exact_id",
            requirement_id="REQ-050",
            document_revision="Rev01",
        ),
        make_evidence(
            evidence_id=(
                "FRD-REQ050-REV02"
            ),
            document_type="frd",
            method="exact_id",
            requirement_id="REQ-050",
            document_revision="Rev02",
        ),
        make_evidence(
            evidence_id="OBS-061",
            document_type="observation",
            method="exact_id",
            requirement_id="REQ-050",
            observation_id="OBS-061",
        ),
    ]

    traceability = [
        make_evidence(
            evidence_id=(
                "REVIEW-OBS061"
            ),
            document_type="review",
            method="traceability",
            observation_id="OBS-061",
        )
    ]

    result = build_final_context(
        exact_results=exact,
        traceability_results=(
            traceability
        ),
        rrf_results=[],
        context_limit=4,
    )

    assert set(
        result.evidence_ids
    ) == {
        "FRD-REQ050-REV01",
        "FRD-REQ050-REV02",
        "OBS-061",
        "REVIEW-OBS061",
    }

    revisions = {
        item.metadata.get(
            "document_revision"
        )
        for item in result.evidence
        if item.document_type == "frd"
    }

    assert revisions == {
        "Rev01",
        "Rev02",
    }


def test_traceability_has_priority_over_unrelated_rrf() -> None:
    """
    Controlled traceability evidence should be selected before an
    unrelated lower-priority RRF candidate.
    """

    exact = [
        make_evidence(
            evidence_id="REQ-021",
            document_type="frd",
            method="exact_id",
            requirement_id="REQ-021",
        )
    ]

    traceability = [
        make_evidence(
            evidence_id="OBS-031",
            document_type=(
                "observation"
            ),
            method="traceability",
            requirement_id="REQ-021",
            observation_id="OBS-031",
        )
    ]

    rrf = [
        make_evidence(
            evidence_id="UNRELATED",
            document_type="observation",
            method="rrf",
        )
    ]

    result = build_final_context(
        exact_results=exact,
        traceability_results=(
            traceability
        ),
        rrf_results=rrf,
        context_limit=2,
    )

    assert result.evidence_ids == (
        "REQ-021",
        "OBS-031",
    )


def test_duplicate_evidence_is_not_repeated() -> None:
    """The same stable evidence ID must appear only once."""

    item = make_evidence(
        evidence_id="REQ-021",
        document_type="frd",
        method="exact_id",
        requirement_id="REQ-021",
    )

    result = build_final_context(
        exact_results=[
            item,
            item,
        ],
        traceability_results=[
            item,
        ],
        rrf_results=[
            item,
        ],
        context_limit=4,
    )

    assert result.evidence_ids == (
        "REQ-021",
    )


def test_missing_core_evidence_types_are_preferred() -> None:
    """
    Code and Test should be selected before another FRD when FRD and
    Observation are already represented.
    """

    existing = [
        make_evidence(
            evidence_id="FRD-1",
            document_type="frd",
            method="exact_id",
        ),
        make_evidence(
            evidence_id="OBS-1",
            document_type=(
                "observation"
            ),
            method="exact_id",
        ),
    ]

    candidates = [
        make_evidence(
            evidence_id="FRD-2",
            document_type="frd",
            method="traceability",
        ),
        make_evidence(
            evidence_id="CODE-1",
            document_type="code",
            method="traceability",
        ),
        make_evidence(
            evidence_id="TEST-1",
            document_type="test",
            method="traceability",
        ),
    ]

    selected = (
        choose_diverse_additions(
            candidates,
            existing,
            limit=2,
        )
    )

    assert tuple(
        item.document_type
        for item in selected
    ) == (
        "code",
        "test",
    )


def test_final_ranks_are_one_based() -> None:
    """Final context ranks must begin at one."""

    result = build_final_context(
        exact_results=[],
        traceability_results=[],
        rrf_results=[
            make_evidence(
                evidence_id="A",
                document_type="frd",
                method="rrf",
            ),
            make_evidence(
                evidence_id="B",
                document_type="code",
                method="rrf",
            ),
        ],
        context_limit=4,
    )

    assert tuple(
        item.rank
        for item in result.evidence
    ) == (
        1,
        2,
    )


def test_redundant_exact_results_do_not_block_core_types() -> None:
    """
    Redundant exact records of one evidence type must not consume all
    context positions.

    This protects scenarios such as OBS-052 where multiple exact code
    records should not prevent required FRD evidence from entering.
    """

    exact = [
        make_evidence(
            evidence_id=(
                f"EXACT-{index}"
            ),
            document_type="frd",
            method="exact_id",
        )
        for index in range(6)
    ]

    traceability = [
        make_evidence(
            evidence_id="TRACE-CODE",
            document_type="code",
            method="traceability",
        )
    ]

    rrf = [
        make_evidence(
            evidence_id="RRF-TEST",
            document_type="test",
            method="rrf",
        )
    ]

    result = build_final_context(
        exact_results=exact,
        traceability_results=(
            traceability
        ),
        rrf_results=rrf,
        context_limit=4,
    )

    assert "EXACT-0" in (
        result.evidence_ids
    )

    assert "TRACE-CODE" in (
        result.evidence_ids
    )

    assert "RRF-TEST" in (
        result.evidence_ids
    )

    document_types = {
        item.document_type
        for item in result.evidence
    }

    assert "frd" in document_types
    assert "code" in document_types
    assert "test" in document_types

    assert len(
        result.evidence
    ) <= 4


def test_core_engineering_types_precede_review() -> None:
    """
    Review evidence must not displace missing Code or Test evidence.
    """

    existing = [
        make_evidence(
            evidence_id="FRD-1",
            document_type="frd",
            method="exact_id",
        ),
        make_evidence(
            evidence_id="OBS-1",
            document_type=(
                "observation"
            ),
            method="exact_id",
        ),
    ]

    candidates = [
        make_evidence(
            evidence_id="REVIEW-1",
            document_type="review",
            method="traceability",
        ),
        make_evidence(
            evidence_id="CODE-1",
            document_type="code",
            method="rrf",
        ),
        make_evidence(
            evidence_id="TEST-1",
            document_type="test",
            method="rrf",
        ),
    ]

    selected = (
        choose_diverse_additions(
            candidates,
            existing,
            limit=2,
        )
    )

    assert tuple(
        item.document_type
        for item in selected
    ) == (
        "code",
        "test",
    )


def test_review_is_secondary_to_missing_code() -> None:
    """
    When FRD, Observation and Test already exist, missing Code should
    be selected before Review.
    """

    existing = [
        make_evidence(
            evidence_id="FRD-1",
            document_type="frd",
            method="exact_id",
        ),
        make_evidence(
            evidence_id="OBS-1",
            document_type=(
                "observation"
            ),
            method="exact_id",
        ),
        make_evidence(
            evidence_id="TEST-1",
            document_type="test",
            method="traceability",
        ),
    ]

    candidates = [
        make_evidence(
            evidence_id="REVIEW-1",
            document_type="review",
            method="traceability",
        ),
        make_evidence(
            evidence_id="CODE-1",
            document_type="code",
            method="rrf",
        ),
    ]

    selected = (
        choose_diverse_additions(
            candidates,
            existing,
            limit=1,
        )
    )

    assert len(selected) == 1

    assert (
        selected[0].document_type
        == "code"
    )


def test_context_origin_counts_are_consistent() -> None:
    """Origin statistics must describe the selected final evidence."""

    exact = [
        make_evidence(
            evidence_id="FRD-REQ017",
            document_type="frd",
            method="exact_id",
            requirement_id="REQ-017",
        )
    ]

    traceability = [
        make_evidence(
            evidence_id="OBS-023",
            document_type=(
                "observation"
            ),
            method="traceability",
            requirement_id="REQ-017",
            observation_id="OBS-023",
        )
    ]

    rrf = [
        make_evidence(
            evidence_id="CODE-DATA",
            document_type="code",
            method="rrf",
            requirement_id="REQ-017",
        )
    ]

    result = build_final_context(
        exact_results=exact,
        traceability_results=(
            traceability
        ),
        rrf_results=rrf,
        context_limit=4,
    )

    assert result.exact_count == 1
    assert result.traceability_count == 1
    assert result.rrf_count == 1

    assert (
        result.record_count
        == 3
    )
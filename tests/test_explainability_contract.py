"""
Phase 15C explainability-contract tests.

The tests confirm that ProductionResponse contains everything
required by the Evidence Matrix, Decision Trace, Provenance,
and Safety UI tabs.
"""

from __future__ import annotations

from pathlib import Path

from config import get_settings

from src.production_pipeline import (
    ProductionPipelineConfig,
    ReqTraceProductionPipeline,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def make_pipeline():
    """Use deterministic mode for UI explainability tests."""

    return ReqTraceProductionPipeline(
        settings=get_settings(),
        base_directory=BASE_DIR,
        config=(
            ProductionPipelineConfig(
                narration_enabled=False,
                allow_repair_attempt=False,
            )
        ),
    )


def test_evidence_matrix_has_provenance() -> None:

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert result.citations

    assert all(
        citation.citation_id
        for citation
        in result.citations
    )

    assert all(
        citation.document_type
        for citation
        in result.citations
    )

    assert all(
        citation.provenance_location
        for citation
        in result.citations
    )


def test_decision_trace_has_rule_and_reason() -> None:

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert result.assessment_rule
    assert result.assessment_reason
    assert result.confidence_reason
    assert result.closure_reason


def test_code_provenance_has_physical_lines() -> None:

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    code_citations = [
        citation
        for citation
        in result.citations
        if (
            citation.document_type
            == "code"
        )
    ]

    assert code_citations

    assert any(
        "lines"
        in citation
        .provenance_location
        .lower()

        for citation
        in code_citations
    )


def test_test_provenance_preserves_result() -> None:

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    test_citations = [
        citation
        for citation
        in result.citations
        if (
            citation.document_type
            == "test"
        )
    ]

    assert test_citations

    assert any(
        citation.test_case_id
        == "TC-102"

        and citation.result_status
        == "FAIL"

        for citation
        in test_citations
    )


def test_traceability_explainability_preserves_req_041() -> None:

    result = (
        make_pipeline()
        .run(
            "OBS-052"
        )
    )

    assert "REQ-041" in (
        result.requirement_ids
    )

    assert "REQ-042" not in (
        result.requirement_ids
    )


def test_revision_conflict_remains_explainable() -> None:

    result = (
        make_pipeline()
        .run(
            "REQ-050"
        )
    )

    frd_records = [
        citation
        for citation
        in result.citations
        if (
            citation.document_type
            == "frd"
        )
    ]

    assert len(
        frd_records
    ) >= 2

    assert (
        result.assessment
        == "INCONCLUSIVE"
    )


def test_closure_trace_is_separate() -> None:

    result = (
        make_pipeline()
        .run(
            (
                "What is the closure status "
                "of OBS-070?"
            )
        )
    )

    assert (
        result.assessment
        == "SUPPORTED"
    )

    assert (
        result.closure_status
        == "CLOSURE NOT ESTABLISHED"
    )


def test_safety_tab_has_delivery_data() -> None:

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert result.grounding_passed is True
    assert result.safe_delivery is True

    assert (
        result.generation_attempts
        == 0
    )

    assert result.used_fallback is True
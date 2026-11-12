"""
Phase 15B UI-facing production-contract tests.
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
    """Use deterministic mode for UI contract tests."""

    return ReqTraceProductionPipeline(
        settings=get_settings(),
        base_directory=BASE_DIR,
        config=ProductionPipelineConfig(
            narration_enabled=False,
            allow_repair_attempt=False,
        ),
    )


def test_ui_receives_decision_fields() -> None:

    result = make_pipeline().run(
        "OBS-023"
    )

    assert result.assessment
    assert result.confidence
    assert result.closure_status

    assert result.assessment_reason
    assert result.confidence_reason


def test_ui_receives_traceability_ids() -> None:

    result = make_pipeline().run(
        "OBS-023"
    )

    assert "REQ-017" in (
        result.requirement_ids
    )

    assert "OBS-023" in (
        result.observation_ids
    )

    assert "TC-102" in (
        result.test_case_ids
    )


def test_ui_receives_citations() -> None:

    result = make_pipeline().run(
        "OBS-023"
    )

    assert result.citations

    assert all(
        citation.citation_id
        for citation
        in result.citations
    )

    assert all(
        citation.provenance_location
        for citation
        in result.citations
    )


def test_ui_receives_safe_final_answer() -> None:

    result = make_pipeline().run(
        "OBS-023"
    )

    assert result.final_answer

    assert result.safe_delivery is True

    assert result.grounding_passed is True


def test_ui_traceability_concern_preserves_req_041() -> None:

    result = make_pipeline().run(
        "OBS-052"
    )

    assert "REQ-041" in (
        result.requirement_ids
    )

    assert "REQ-042" not in (
        result.requirement_ids
    )


def test_ui_revision_conflict_remains_inconclusive() -> None:

    result = make_pipeline().run(
        "REQ-050"
    )

    assert (
        result.assessment
        == "INCONCLUSIVE"
    )


def test_ui_closure_is_separate() -> None:

    result = make_pipeline().run(
        "OBS-070",
        evaluate_closure=True,
    )

    assert (
        result.assessment
        == "SUPPORTED"
    )

    assert (
        result.closure_status
        == "CLOSURE NOT ESTABLISHED"
    )
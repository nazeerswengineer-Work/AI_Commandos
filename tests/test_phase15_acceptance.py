"""
Final Phase 15 production acceptance tests.
"""

from __future__ import annotations

from pathlib import Path

from config import get_settings

from src.narration_models import (
    NarrationStatus,
)

from src.production_pipeline import (
    ProductionPipelineConfig,
    ReqTraceProductionPipeline,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def pipeline():
    """Create deterministic production acceptance pipeline."""

    return ReqTraceProductionPipeline(
        settings=get_settings(),
        base_directory=BASE_DIR,
        config=ProductionPipelineConfig(
            narration_enabled=False,
            allow_repair_attempt=False,
        ),
    )


def test_obs_023_acceptance() -> None:

    result = pipeline().run(
        "OBS-023"
    )

    assert result.assessment == "SUPPORTED"
    assert result.confidence == "HIGH"

    assert (
        result.closure_status
        == "NOT ASSESSED"
    )


def test_obs_031_acceptance() -> None:

    result = pipeline().run(
        "OBS-031"
    )

    assert (
        result.assessment
        == "NOT SUPPORTED"
    )

    assert result.confidence == "HIGH"


def test_obs_044_acceptance() -> None:

    result = pipeline().run(
        "OBS-044"
    )

    assert (
        result.assessment
        == "INCONCLUSIVE"
    )

    assert "TC-301" in (
        result.test_case_ids
    )


def test_obs_052_traceability_acceptance() -> None:

    result = pipeline().run(
        "OBS-052"
    )

    assert (
        result.assessment
        == "INCONCLUSIVE"
    )

    assert "REQ-041" in (
        result.requirement_ids
    )

    assert "REQ-042" not in (
        result.requirement_ids
    )


def test_req_050_revision_acceptance() -> None:

    result = pipeline().run(
        "REQ-050"
    )

    assert (
        result.assessment
        == "INCONCLUSIVE"
    )

    frd_citations = [
        citation
        for citation
        in result.citations
        if (
            citation.document_type
            == "frd"
        )
    ]

    assert len(
        frd_citations
    ) >= 2


def test_obs_070_automatic_closure_acceptance() -> None:

    result = pipeline().run(
        "What is the closure status of OBS-070?"
    )

    assert (
        result.assessment
        == "SUPPORTED"
    )

    assert (
        result.closure_status
        == "CLOSURE NOT ESTABLISHED"
    )


def test_safe_delivery_acceptance() -> None:

    result = pipeline().run(
        "OBS-023"
    )

    assert result.safe_delivery is True

    assert result.grounding_passed is True

    assert (
        result.delivery_status
        == NarrationStatus
        .DETERMINISTIC_FALLBACK
    )


def test_phase15_provenance_acceptance() -> None:

    result = pipeline().run(
        "OBS-023"
    )

    assert result.citations

    assert all(
        citation.evidence_id
        for citation
        in result.citations
    )

    assert all(
        citation.provenance_location
        for citation
        in result.citations
    )


def test_phase15_timing_acceptance() -> None:

    result = pipeline().run(
        "OBS-023"
    )

    assert (
        result.timing.total_seconds
        >= 0.0
    )

    assert (
        result.timing
        .deterministic_seconds
        >= 0.0
    )


def test_no_llm_dependency_in_deterministic_mode() -> None:

    result = pipeline().run(
        "OBS-023"
    )

    assert (
        result.generated_by_llm
        is False
    )

    assert (
        result.generation_attempts
        == 0
    )
    
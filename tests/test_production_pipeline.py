"""
Tests for Phase 15A unified production pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import get_settings

from src.narration_models import (
    NarrationStatus,
)
from src.production_pipeline import (
    ProductionPipelineConfig,
    ReqTraceProductionPipeline,
)
from src.production_pipeline import (
    ProductionPipelineConfig,
    ReqTraceProductionPipeline,
    detect_closure_intent,
)

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

def test_closure_intent_detection() -> None:

    assert (
        detect_closure_intent(
            "What is the closure status of OBS-070?"
        )
        is True
    )

    assert (
        detect_closure_intent(
            "Is OBS-070 closed?"
        )
        is True
    )

    assert (
        detect_closure_intent(
            "Is OBS-023 still open?"
        )
        is True
    )

    assert (
        detect_closure_intent(
            "Can OBS-070 be closed?"
        )
        is True
    )

    assert (
        detect_closure_intent(
            "Is OBS-070 ready for closure?"
        )
        is True
    )

    assert (
        detect_closure_intent(
            "Analyze OBS-023"
        )
        is False
    )

def test_natural_language_closure_query_auto_evaluates() -> None:

    result = (
        make_pipeline()
        .run(
            "What is the closure status of OBS-070?"
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

def test_plain_observation_query_does_not_auto_evaluate_closure() -> None:

    result = (
        make_pipeline()
        .run(
            "OBS-070"
        )
    )

    assert (
        result.closure_status
        == "NOT ASSESSED"
    )

def make_pipeline() -> (
    ReqTraceProductionPipeline
):
    """
    Create a deterministic-only production
    pipeline for unit tests.
    """

    return ReqTraceProductionPipeline(
        settings=get_settings(),

        base_directory=(
            BASE_DIR
        ),

        config=(
            ProductionPipelineConfig(
                narration_enabled=False,
                allow_repair_attempt=False,
            )
        ),
    )


def test_empty_query_is_rejected() -> None:
    """Production queries must not be empty."""

    pipeline = make_pipeline()

    with pytest.raises(
        ValueError
    ):

        pipeline.run(
            "   "
        )


def test_obs_023_production_response() -> None:
    """OBS-023 should preserve deterministic supported result."""

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert (
        result.assessment
        == "SUPPORTED"
    )

    assert (
        result.confidence
        == "HIGH"
    )

    assert (
        result.closure_status
        == "NOT ASSESSED"
    )

    assert result.grounding_passed is True

    assert (
        result.delivery_status
        == NarrationStatus
        .DETERMINISTIC_FALLBACK
    )

    assert result.used_fallback is True

    assert (
        result.generation_attempts
        == 0
    )


def test_obs_031_production_response() -> None:
    """OBS-031 should remain NOT SUPPORTED."""

    result = (
        make_pipeline()
        .run(
            "OBS-031"
        )
    )

    assert (
        result.assessment
        == "NOT SUPPORTED"
    )

    assert (
        result.confidence
        == "HIGH"
    )


def test_obs_052_preserves_req_041() -> None:
    """Explicit traceability must remain REQ-041."""

    result = (
        make_pipeline()
        .run(
            "OBS-052"
        )
    )

    assert (
        result.assessment
        == "INCONCLUSIVE"
    )

    assert (
        "REQ-041"
        in result.requirement_ids
    )

    assert (
        "REQ-042"
        not in result.requirement_ids
    )


def test_req_050_remains_inconclusive() -> None:
    """Revision conflict must remain unresolved."""

    result = (
        make_pipeline()
        .run(
            "REQ-050"
        )
    )

    assert (
        result.assessment
        == "INCONCLUSIVE"
    )

    assert (
        "REQ-050"
        in result.requirement_ids
    )


def test_obs_070_closure_is_separate() -> None:
    """Supported defect can still have unestablished closure."""

    result = (
        make_pipeline()
        .run(
            "OBS-070",
            evaluate_closure=True,
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


def test_production_citations_are_present() -> None:
    """Application result must preserve validated provenance."""

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert result.citations

    citation_ids = {
        citation.citation_id
        for citation
        in result.citations
    }

    assert "CIT-001" in (
        citation_ids
    )


def test_production_test_citation_preserves_result() -> None:
    """Actual TC-102 FAIL evidence must survive."""

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


def test_deterministic_only_is_safe_delivery() -> None:
    """Disabling the LLM must remain a safe production mode."""

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert result.safe_delivery is True

    assert (
        result.generated_by_llm
        is False
    )


def test_response_is_json_safe() -> None:
    """Production response should serialize cleanly."""

    result = (
        make_pipeline()
        .run(
            "OBS-023"
        )
    )

    payload = (
        result.to_dict()
    )

    serialized = (
        json.dumps(
            payload
        )
    )

    assert serialized

    assert (
        payload[
            "assessment"
        ]
        == "SUPPORTED"
    )

    assert (
        payload[
            "safe_delivery"
        ]
        is True
    )
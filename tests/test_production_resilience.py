"""
Phase 15D production-resilience tests.
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
from src.identifier_integrity import (
    ExplicitIdentifierNotFound,
)

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def make_deterministic_pipeline():
    """Create production pipeline without LLM dependency."""

    return ReqTraceProductionPipeline(
        settings=get_settings(),
        base_directory=BASE_DIR,
        config=ProductionPipelineConfig(
            narration_enabled=False,
            allow_repair_attempt=False,
        ),
    )


def test_empty_query_is_rejected() -> None:

    with pytest.raises(
        ValueError
    ):
        make_deterministic_pipeline().run(
            "   "
        )


def test_deterministic_mode_needs_no_ollama() -> None:

    result = (
        make_deterministic_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert (
        result.delivery_status
        == NarrationStatus
        .DETERMINISTIC_FALLBACK
    )

    assert result.safe_delivery is True
    assert result.generation_attempts == 0


def test_fallback_is_grounded() -> None:

    result = (
        make_deterministic_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert result.grounding_passed is True
    assert result.final_answer
    assert result.citations


def test_timing_is_recorded() -> None:

    result = (
        make_deterministic_pipeline()
        .run(
            "OBS-023"
        )
    )

    assert (
        result.timing
        .deterministic_seconds
        >= 0.0
    )

    assert (
        result.timing
        .narration_seconds
        >= 0.0
    )

    assert (
        result.timing
        .total_seconds
        >= result.timing
        .deterministic_seconds
    )


def test_response_metrics_are_json_safe() -> None:

    result = (
        make_deterministic_pipeline()
        .run(
            "OBS-023"
        )
    )

    payload = result.to_dict()

    assert "timing" in payload

    assert json.dumps(
        payload
    )


def test_closure_intent_remains_operational() -> None:

    result = (
        make_deterministic_pipeline()
        .run(
            "Is OBS-070 open or closed?"
        )
    )

    assert (
        result.closure_status
        == "CLOSURE NOT ESTABLISHED"
    )


def test_traceability_remains_protected() -> None:

    result = (
        make_deterministic_pipeline()
        .run(
            "Is OBS-052 open or closed?"
        )
    )

    assert "REQ-041" in (
        result.requirement_ids
    )

    assert "REQ-042" not in (
        result.requirement_ids
    )

def test_unknown_observation_is_not_substituted() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ) as error:

        make_deterministic_pipeline().run(
            "OBS-999"
        )

    assert error.value.missing_ids == (
        "OBS-999",
    )


def test_unknown_requirement_is_not_substituted() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ):

        make_deterministic_pipeline().run(
            "REQ-999"
        )


def test_unknown_test_case_is_not_substituted() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ):

        make_deterministic_pipeline().run(
            "TC-999"
        )


def test_unknown_closure_query_is_not_substituted() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ):

        make_deterministic_pipeline().run(
            (
                "What is the closure status "
                "of OBS-999?"
            )
        )
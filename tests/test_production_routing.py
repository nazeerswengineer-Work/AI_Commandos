"""
Tests for Phase 16I production routing.
"""

from __future__ import annotations

from src.generalized_qa_models import (
    GeneralizedQAResponse,
)

from src.production_pipeline import (
    ProductionResponse,
)

from src.query_intent import (
    QueryIntent,
    classify_query_intent,
)


def test_routed_response_types_are_distinct() -> None:
    """
    Inventory QA and engineering assessment use separate
    application-facing response models.
    """

    assert (
        GeneralizedQAResponse
        is not ProductionResponse
    )
def test_greeting_routes_to_general() -> None:
    """A greeting must route to GENERAL rather than engineering analysis."""

    result = classify_query_intent(
        "hi"
    )

    assert (
        result.intent
        == QueryIntent.GENERAL
    )

    assert (
        result.matched_rule
        == "QI-009_GENERAL"
    )
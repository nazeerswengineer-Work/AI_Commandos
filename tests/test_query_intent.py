"""
Tests for deterministic REQTRACE AI query-intent routing.
"""

from __future__ import annotations

import pytest

from src.query_intent import (
    QueryIntent,
    classify_query_intent,
)


@pytest.mark.parametrize(
    (
        "query",
        "expected",
    ),
    [
        (
            "How many observations are open?",
            QueryIntent.COUNT,
        ),

        (
            "Count the closed observations",
            QueryIntent.COUNT,
        ),

        (
            "What is the total number of requirements?",
            QueryIntent.COUNT,
        ),

        (
            "List all open observations",
            QueryIntent.LIST,
        ),

        (
            "Show all requirements",
            QueryIntent.LIST,
        ),

        (
            "Which observations reference REQ-017?",
            QueryIntent.LIST,
        ),

        (
            "What does REQ-900 specify?",
            QueryIntent.LOOKUP,
        ),

        (
            "Describe OBS-023",
            QueryIntent.LOOKUP,
        ),

        (
            "Compare expected behavior against OBS-023",
            QueryIntent.COMPARE,
        ),

        (
            "Compare REQ-017 with the available observation",
            QueryIntent.COMPARE,
        ),

        (
            "Is OBS-023 supported?",
            QueryIntent.ASSESSMENT,
        ),

        (
            "Does the evidence contradict OBS-031?",
            QueryIntent.ASSESSMENT,
        ),

        (
            "Which requirements have missing test evidence?",
            QueryIntent.EVIDENCE_AVAILABILITY,
        ),

        (
            "Find verification evidence for REQ-017",
            QueryIntent.EVIDENCE_AVAILABILITY,
        ),

        (
            "Can OBS-070 be closed?",
            QueryIntent.CLOSURE,
        ),

        (
            "What is the closure status of OBS-070?",
            QueryIntent.CLOSURE,
        ),
    ],
)
def test_intent_classification(
    query: str,
    expected: QueryIntent,
) -> None:

    result = (
        classify_query_intent(
            query
        )
    )

    assert (
        result.intent
        == expected
    )


def test_count_has_priority_over_closed_word() -> None:
    """
    "How many observations are closed?" is an inventory count,
    not a closure-readiness assessment.
    """

    result = classify_query_intent(
        "How many observations are closed?"
    )

    assert (
        result.intent
        == QueryIntent.COUNT
    )


def test_explicit_id_defaults_to_lookup() -> None:

    result = classify_query_intent(
        "REQ-900"
    )

    assert (
        result.intent
        == QueryIntent.LOOKUP
    )

    assert (
        result.confidence
        == "MEDIUM"
    )


def test_empty_query_is_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        classify_query_intent(
            "   "
        )

def test_list_closed_observations_is_list_not_closure() -> None:
    """
    Entity inventory status is different from
    engineering closure-readiness assessment.
    """

    result = classify_query_intent(
        "List all closed observations"
    )

    assert (
        result.intent
        == QueryIntent.LIST
    )

def test_missing_test_evidence_has_priority_over_list() -> None:
    """Evidence availability must outrank generic list wording."""

    result = classify_query_intent(
        (
            "Which requirements have "
            "missing test evidence?"
        )
    )

    assert (
        result.intent
        == (
            QueryIntent
            .EVIDENCE_AVAILABILITY
        )
    )
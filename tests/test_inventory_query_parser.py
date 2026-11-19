"""
Tests for deterministic inventory-query parsing.
"""

from __future__ import annotations

import pytest

from src.evidence_inventory import (
    InventoryEntityType,
    InventoryOperation,
)

from src.inventory_query_parser import (
    InventoryQueryParseError,
    parse_inventory_query,
)


def test_parse_open_observation_count() -> None:

    result = parse_inventory_query(
        "How many observations are open?"
    )

    query = result.inventory_query

    assert (
        query.operation
        == InventoryOperation.COUNT
    )

    assert (
        query.entity_type
        == InventoryEntityType
        .OBSERVATION
    )

    assert (
        query.status_filter
        == "OPEN"
    )


def test_parse_closed_observation_list() -> None:

    result = parse_inventory_query(
        "List all closed observations"
    )

    query = result.inventory_query

    assert (
        query.operation
        == InventoryOperation.LIST
    )

    assert (
        query.status_filter
        == "CLOSED"
    )


def test_parse_observations_for_requirement() -> None:

    result = parse_inventory_query(
        "Which observations reference REQ-017?"
    )

    query = result.inventory_query

    assert (
        query.entity_type
        == InventoryEntityType
        .OBSERVATION
    )

    assert (
        query.requirement_id
        == "REQ-017"
    )


def test_parse_failed_test_count() -> None:

    result = parse_inventory_query(
        "How many Test Cases failed?"
    )

    query = result.inventory_query

    assert (
        query.entity_type
        == InventoryEntityType.TEST
    )

    assert (
        query.result_filter
        == "FAIL"
    )


def test_parse_not_executed_test_list() -> None:

    result = parse_inventory_query(
        "List tests that were not executed"
    )

    query = result.inventory_query

    assert (
        query.operation
        == InventoryOperation.LIST
    )

    assert (
        query.result_filter
        == "NOT EXECUTED"
    )


def test_parse_requirement_count() -> None:

    result = parse_inventory_query(
        "How many requirements are indexed?"
    )

    query = result.inventory_query

    assert (
        query.entity_type
        == InventoryEntityType
        .REQUIREMENT
    )

    assert (
        query.operation
        == InventoryOperation.COUNT
    )


def test_parse_tests_for_requirement() -> None:

    result = parse_inventory_query(
        "List Test Cases related to REQ-017"
    )

    query = result.inventory_query

    assert (
        query.entity_type
        == InventoryEntityType.TEST
    )

    assert (
        query.requirement_id
        == "REQ-017"
    )


def test_count_closed_is_not_closure_assessment() -> None:

    result = parse_inventory_query(
        "How many observations are closed?"
    )

    assert (
        result.inventory_query
        .operation
        == InventoryOperation.COUNT
    )

    assert (
        result.inventory_query
        .status_filter
        == "CLOSED"
    )


def test_assessment_query_is_rejected() -> None:

    with pytest.raises(
        InventoryQueryParseError
    ):

        parse_inventory_query(
            "Is OBS-023 supported?"
        )


def test_closure_query_is_rejected() -> None:

    with pytest.raises(
        InventoryQueryParseError
    ):

        parse_inventory_query(
            "Can OBS-070 be closed?"
        )


def test_missing_entity_is_rejected() -> None:

    with pytest.raises(
        InventoryQueryParseError
    ):

        parse_inventory_query(
            "How many are open?"
        )


def test_conflicting_observation_status_is_rejected() -> None:

    with pytest.raises(
        InventoryQueryParseError
    ):

        parse_inventory_query(
            (
                "How many observations are "
                "open and closed?"
            )
        )
def test_parse_missing_test_evidence_query() -> None:

    result = parse_inventory_query(
        (
            "Which requirements have "
            "missing test evidence?"
        )
    )

    assert (
        result.inventory_query
        .entity_type
        == InventoryEntityType
        .REQUIREMENT
    )

    assert (
        result.inventory_query
        .operation
        == InventoryOperation
        .EVIDENCE_AVAILABILITY
    )

    assert (
        result.evidence_type
        == "test"
    )
def test_parse_missing_code_evidence_query() -> None:

    result = parse_inventory_query(
        (
            "Which requirements have "
            "missing code evidence?"
        )
    )

    assert (
        result.inventory_query
        .entity_type
        == InventoryEntityType
        .REQUIREMENT
    )

    assert (
        result.evidence_type
        == "code"
    )
def test_parse_missing_observation_evidence_query() -> None:

    result = parse_inventory_query(
        (
            "Which requirements have "
            "missing observation evidence?"
        )
    )

    assert (
        result.inventory_query
        .entity_type
        == InventoryEntityType
        .REQUIREMENT
    )

    assert (
        result.evidence_type
        == "observation"
    )
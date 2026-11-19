"""
Tests for Phase 16I deterministic evidence inventory.
"""

from __future__ import annotations

from src.evidence_inventory import (
    InventoryEntityType,
    InventoryOperation,
    InventoryQuery,
    answer_inventory_query,
    answer_requirement_evidence_availability,
    build_inventory_summary,
    requirement_evidence_availability,
    requirements_missing_actual_test_evidence,
    requirements_missing_code_evidence,
    requirements_missing_observation_evidence,
)

from src.ingestion_manifest import (
    IngestionManifest,
)


def make_manifest() -> IngestionManifest:
    """Create inventory test manifest."""

    manifest = (
        IngestionManifest.empty()
    )

    manifest.records = [
        {
            "evidence_id": (
                "OBS::OBS-001"
            ),

            "document_type": (
                "observation"
            ),

            "document_name": (
                "OBS-001.md"
            ),

            "observation_id": (
                "OBS-001"
            ),

            "requirement_id": (
                "REQ-001"
            ),

            "status": "OPEN",

            "content": (
                "Observation ID: OBS-001\n"
                "Status: OPEN"
            ),
        },

        {
            "evidence_id": (
                "OBS::OBS-002"
            ),

            "document_type": (
                "observation"
            ),

            "document_name": (
                "OBS-002.md"
            ),

            "observation_id": (
                "OBS-002"
            ),

            "requirement_id": (
                "REQ-002"
            ),

            "observation_status": (
                "CLOSED"
            ),

            "content": (
                "Observation ID: OBS-002"
            ),
        },

        {
            "evidence_id": (
                "OBS::OBS-003-A"
            ),

            "document_type": (
                "observation"
            ),

            "document_name": (
                "OBS-003.pdf"
            ),

            "observation_id": (
                "OBS-003"
            ),

            "requirement_id": (
                "REQ-001"
            ),

            "content": (
                "Observation Status: OPEN"
            ),
        },

        {
            "evidence_id": (
                "OBS::OBS-003-B"
            ),

            "document_type": (
                "observation"
            ),

            "document_name": (
                "OBS-003.pdf"
            ),

            "observation_id": (
                "OBS-003"
            ),

            "requirement_id": (
                "REQ-001"
            ),

            "content": (
                "Additional evidence section."
            ),
        },

        {
            "evidence_id": (
                "FRD::REQ-001"
            ),

            "document_type": "frd",

            "document_name": (
                "FRD_REQ-001.pdf"
            ),

            "requirement_id": (
                "REQ-001"
            ),
        },

        {
            "evidence_id": (
                "FRD::REQ-002"
            ),

            "document_type": "frd",

            "document_name": (
                "FRD_REQ-002.pdf"
            ),

            "requirement_id": (
                "REQ-002"
            ),
        },

        {
            "evidence_id": (
                "TEST::TC-001"
            ),

            "document_type": "test",

            "document_name": (
                "Test_TC-001.pdf"
            ),

            "test_case_id": (
                "TC-001"
            ),

            "requirement_id": (
                "REQ-001"
            ),

            "result_status": "FAIL",
        },

        {
            "evidence_id": (
                "TEST::TC-002"
            ),

            "document_type": "test",

            "document_name": (
                "Test_TC-002.pdf"
            ),

            "test_case_id": (
                "TC-002"
            ),

            "requirement_id": (
                "REQ-002"
            ),

            "content": (
                "Result Status: PASS"
            ),
        },
    ]

    return manifest


def test_inventory_deduplicates_observation_ids() -> None:
    """Multiple evidence chunks must not inflate entity count."""

    summary = build_inventory_summary(
        make_manifest()
    )

    assert (
        tuple(
            record.entity_id
            for record
            in summary.observations
        )
        == (
            "OBS-001",
            "OBS-002",
            "OBS-003",
        )
    )


def test_open_observation_count() -> None:
    """Open observations should be counted deterministically."""

    summary = build_inventory_summary(
        make_manifest()
    )

    answer = answer_inventory_query(
        summary,

        InventoryQuery(
            operation=(
                InventoryOperation.COUNT
            ),

            entity_type=(
                InventoryEntityType
                .OBSERVATION
            ),

            status_filter="OPEN",
        ),
    )

    assert answer.count == 2

    assert (
        tuple(
            record.entity_id
            for record
            in answer.records
        )
        == (
            "OBS-001",
            "OBS-003",
        )
    )


def test_closed_observation_list() -> None:

    summary = build_inventory_summary(
        make_manifest()
    )

    answer = answer_inventory_query(
        summary,

        InventoryQuery(
            operation=(
                InventoryOperation.LIST
            ),

            entity_type=(
                InventoryEntityType
                .OBSERVATION
            ),

            status_filter="CLOSED",
        ),
    )

    assert answer.count == 1

    assert (
        "- OBS-002"
        in answer.answer_text
    )


def test_observations_related_to_requirement() -> None:

    summary = build_inventory_summary(
        make_manifest()
    )

    answer = answer_inventory_query(
        summary,

        InventoryQuery(
            operation=(
                InventoryOperation.LIST
            ),

            entity_type=(
                InventoryEntityType
                .OBSERVATION
            ),

            requirement_id=(
                "REQ-001"
            ),
        ),
    )

    assert answer.count == 2

    assert (
        tuple(
            record.entity_id
            for record
            in answer.records
        )
        == (
            "OBS-001",
            "OBS-003",
        )
    )


def test_failed_test_count() -> None:

    summary = build_inventory_summary(
        make_manifest()
    )

    answer = answer_inventory_query(
        summary,

        InventoryQuery(
            operation=(
                InventoryOperation.COUNT
            ),

            entity_type=(
                InventoryEntityType.TEST
            ),

            result_filter="FAIL",
        ),
    )

    assert answer.count == 1

    assert (
        answer.records[0]
        .entity_id
        == "TC-001"
    )


def test_total_requirement_count() -> None:

    summary = build_inventory_summary(
        make_manifest()
    )

    answer = answer_inventory_query(
        summary,

        InventoryQuery(
            operation=(
                InventoryOperation.COUNT
            ),

            entity_type=(
                InventoryEntityType
                .REQUIREMENT
            ),
        ),
    )

    assert answer.count == 2


def test_indexed_status_is_not_observation_status() -> None:
    """Phase 16H ledger status must not become engineering status."""

    manifest = (
        IngestionManifest.empty()
    )

    manifest.records = [
        {
            "evidence_id": (
                "OBS::OBS-010"
            ),

            "document_type": (
                "observation"
            ),

            "observation_id": (
                "OBS-010"
            ),

            "status": "INDEXED",

            "content": (
                "Observation without "
                "source-reported status."
            ),
        }
    ]

    summary = build_inventory_summary(
        manifest
    )

    assert (
        summary.observations[0]
        .status
        is None
    )

def test_observation_result_status_is_engineering_status() -> None:
    """
    Baseline observation records store OPEN/CLOSED state
    in result_status.
    """

    manifest = (
        IngestionManifest.empty()
    )

    manifest.records = [
        {
            "evidence_id": (
                "OBS::OBS-023"
            ),

            "document_type": (
                "observation"
            ),

            "document_name": (
                "Observation_Report_OBS-023.pdf"
            ),

            "observation_id": (
                "OBS-023"
            ),

            "requirement_id": (
                "REQ-017"
            ),

            "result_status": (
                "OPEN"
            ),

            "status": (
                "INDEXED"
            ),
        }
    ]

    summary = (
        build_inventory_summary(
            manifest
        )
    )

    assert (
        summary.observations[0]
        .status
        == "OPEN"
    )

def test_requirement_missing_actual_test_evidence() -> None:

    summary = build_inventory_summary(
        make_manifest()
    )

    missing = (
        requirements_missing_actual_test_evidence(
            summary
        )
    )

    assert isinstance(
        missing,
        tuple,
    )

def test_test_reference_does_not_count_as_actual_test_evidence() -> None:

    manifest = (
        IngestionManifest.empty()
    )

    manifest.records = [
        {
            "evidence_id": "FRD::REQ-900",
            "document_type": "frd",
            "requirement_id": "REQ-900",
            "test_case_ids": "TC-900",
        }
    ]

    summary = (
        build_inventory_summary(
            manifest
        )
    )

    assert (
        requirements_missing_actual_test_evidence(
            summary
        )
        == (
            "REQ-900",
        )
    )

def test_actual_test_document_satisfies_test_evidence() -> None:

    manifest = (
        IngestionManifest.empty()
    )

    manifest.records = [
        {
            "evidence_id": "FRD::REQ-900",
            "document_type": "frd",
            "requirement_id": "REQ-900",
            "test_case_ids": "TC-900",
        },

        {
            "evidence_id": "TEST::TC-900",
            "document_type": "test",
            "requirement_id": "REQ-900",
            "test_case_id": "TC-900",
            "result_status": "PASS",
        },
    ]

    summary = (
        build_inventory_summary(
            manifest
        )
    )

    assert (
        requirements_missing_actual_test_evidence(
            summary
        )
        == ()
    )
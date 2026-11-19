"""
Deterministic evidence inventory and aggregation for REQTRACE AI.

This module answers inventory-oriented questions such as:

- How many observations are open?
- List all closed observations.
- How many requirements are indexed?
- Which observations reference REQ-017?
- How many tests failed?
- Which requirements lack actual Test evidence?

The engine reads the current synchronized ingestion manifest.

The LLM does not calculate:

- counts;
- statuses;
- entity membership;
- evidence availability;
- engineering identifiers.

No missing metadata is fabricated.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from src.ingestion_manifest import (
    IngestionManifest,
    load_ingestion_manifest,
)


# =====================================================================
# Errors and enums
# =====================================================================


class EvidenceInventoryError(
    RuntimeError
):
    """Raised when evidence inventory processing fails."""


class InventoryEntityType(
    str,
    Enum,
):
    """Supported inventory entity types."""

    OBSERVATION = "observation"
    REQUIREMENT = "requirement"
    TEST = "test"
    CODE = "code"
    REVIEW = "review"


InventoryEntityType.__test__ = False


class InventoryOperation(
    str,
    Enum,
):
    """Supported deterministic inventory operations."""

    COUNT = "COUNT"
    LIST = "LIST"
    LOOKUP = "LOOKUP"
    EVIDENCE_AVAILABILITY = (
        "EVIDENCE_AVAILABILITY"
    )


InventoryOperation.__test__ = False


# =====================================================================
# Engineering-ID patterns
# =====================================================================


ENGINEERING_ID_PATTERN = re.compile(
    r"\b(?P<prefix>REQ|OBS|TC)"
    r"[\s_-]?"
    r"(?P<number>\d+)\b",
    re.IGNORECASE,
)


OBSERVATION_STATUS_PATTERN = re.compile(
    r"\b(?:observation\s+)?status"
    r"\s*[:=-]\s*"
    r"(?P<status>"
    r"OPEN|"
    r"CLOSED|"
    r"RESOLVED|"
    r"REJECTED|"
    r"DEFERRED|"
    r"IN\s+PROGRESS|"
    r"UNDER\s+REVIEW"
    r")\b",
    re.IGNORECASE,
)


TEST_RESULT_PATTERN = re.compile(
    r"\b(?:result(?:\s+status)?|test\s+result)"
    r"\s*[:=-]\s*"
    r"(?P<status>"
    r"PASS(?:ED)?|"
    r"FAIL(?:ED)?|"
    r"NOT\s+EXECUTED|"
    r"NOT\s+RUN|"
    r"BLOCKED"
    r")\b",
    re.IGNORECASE,
)


# =====================================================================
# Normalization
# =====================================================================


def normalize_engineering_id(
    value: object,
) -> str | None:
    """Normalize one explicit REQ, OBS or TC identifier."""

    if value is None:

        return None

    text = str(
        value
    ).strip()

    if not text:

        return None

    match = ENGINEERING_ID_PATTERN.fullmatch(
        text
    )

    if match is None:

        return None

    return (
        f"{match.group('prefix').upper()}-"
        f"{match.group('number')}"
    )


def split_metadata_values(
    value: object,
) -> tuple[str, ...]:
    """Split scalar, iterable or delimiter-separated metadata."""

    if value is None:

        return ()

    if isinstance(
        value,
        (
            tuple,
            list,
            set,
        ),
    ):

        raw_values = [
            str(item)
            for item in value
        ]

    else:

        raw_values = re.split(
            r"[|,;]",
            str(value),
        )

    output: list[str] = []

    seen: set[str] = set()

    for raw_value in raw_values:

        normalized = (
            normalize_engineering_id(
                raw_value
            )
        )

        if (
            normalized is None
            or normalized in seen
        ):

            continue

        seen.add(
            normalized
        )

        output.append(
            normalized
        )

    return tuple(
        output
    )


def normalize_observation_status(
    value: object,
) -> str | None:
    """Normalize a source-recorded observation status."""

    if value is None:

        return None

    normalized = " ".join(
        str(value)
        .strip()
        .upper()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
        .split()
    )

    aliases = {
        "OPEN": "OPEN",
        "CLOSED": "CLOSED",
        "RESOLVED": "RESOLVED",
        "REJECTED": "REJECTED",
        "DEFERRED": "DEFERRED",
        "IN PROGRESS": "IN PROGRESS",
        "UNDER REVIEW": "UNDER REVIEW",
    }

    return aliases.get(
        normalized
    )


def normalize_test_result(
    value: object,
) -> str | None:
    """Normalize a source-reported Test result."""

    if value is None:

        return None

    normalized = " ".join(
        str(value)
        .strip()
        .upper()
        .replace(
            "_",
            " ",
        )
        .split()
    )

    aliases = {
        "PASS": "PASS",
        "PASSED": "PASS",
        "FAIL": "FAIL",
        "FAILED": "FAIL",
        "NOT EXECUTED": (
            "NOT EXECUTED"
        ),
        "NOT RUN": (
            "NOT EXECUTED"
        ),
        "BLOCKED": "BLOCKED",
    }

    return aliases.get(
        normalized
    )


def ordered_unique(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Return deterministic duplicate-free values."""

    output: list[str] = []

    seen: set[str] = set()

    for value in values:

        if value in seen:
            continue

        seen.add(
            value
        )

        output.append(
            value
        )

    return tuple(
        output
    )


# =====================================================================
# Inventory records
# =====================================================================


@dataclass(frozen=True)
class InventoryRecord:
    """One deduplicated engineering inventory entity."""

    entity_id: str
    entity_type: InventoryEntityType

    status: str | None

    document_names: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    requirement_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]

    result_status: str | None = None

    dynamic_ingestion: bool = False

    def __post_init__(
        self,
    ) -> None:
        """Validate inventory record."""

        if not self.entity_id.strip():

            raise EvidenceInventoryError(
                "Inventory entity_id must not be empty."
            )

        if not self.evidence_ids:

            raise EvidenceInventoryError(
                "Inventory record must retain "
                "at least one evidence ID."
            )


@dataclass(frozen=True)
class InventorySummary:
    """Current deterministic evidence inventory."""

    observations: tuple[
        InventoryRecord,
        ...
    ]

    requirements: tuple[
        InventoryRecord,
        ...
    ]

    tests: tuple[
        InventoryRecord,
        ...
    ]

    code_records: tuple[
        InventoryRecord,
        ...
    ]

    reviews: tuple[
        InventoryRecord,
        ...
    ]

    manifest_version: str

    @property
    def total_entities(
        self,
    ) -> int:
        """Return complete unique entity count."""

        return (
            len(
                self.observations
            )
            + len(
                self.requirements
            )
            + len(
                self.tests
            )
            + len(
                self.code_records
            )
            + len(
                self.reviews
            )
        )

    def records_for_type(
        self,
        entity_type: InventoryEntityType,
    ) -> tuple[InventoryRecord, ...]:
        """Return records for one entity type."""

        mapping = {
            InventoryEntityType.OBSERVATION: self.observations,
            InventoryEntityType.REQUIREMENT: self.requirements,
            InventoryEntityType.TEST: self.tests,
            InventoryEntityType.CODE: self.code_records,
            InventoryEntityType.REVIEW: self.reviews,
        }
        return mapping[entity_type]

    def all_records(
        self,
    ) -> tuple[InventoryRecord, ...]:
        """Return every inventory entity in deterministic order."""

        return (
            self.requirements
            + self.observations
            + self.tests
            + self.code_records
            + self.reviews
        )

# =====================================================================
# Query and answer models
# =====================================================================


@dataclass(frozen=True)
class InventoryQuery:
    """Structured deterministic inventory request."""

    operation: InventoryOperation
    entity_type: InventoryEntityType

    status_filter: str | None = None
    result_filter: str | None = None

    requirement_id: str | None = None
    observation_id: str | None = None
    test_case_id: str | None = None


@dataclass(frozen=True)
class InventoryAnswer:
    """Deterministic inventory answer."""

    query: InventoryQuery

    count: int

    records: tuple[
        InventoryRecord,
        ...
    ]

    answer_text: str

    source_statement: str

    def __post_init__(
        self,
    ) -> None:
        """Validate answer consistency."""

        if self.count != len(
            self.records
        ):

            raise EvidenceInventoryError(
                "Inventory count does not match "
                "returned records."
            )

        if not self.answer_text.strip():

            raise EvidenceInventoryError(
                "Inventory answer_text must not be empty."
            )


# =====================================================================
# Evidence availability model
# =====================================================================


@dataclass(frozen=True)
class EvidenceAvailabilityRecord:
    """Actual downstream evidence availability for one requirement."""

    requirement_id: str
    has_requirement_evidence: bool
    has_observation_evidence: bool
    has_code_evidence: bool
    has_test_evidence: bool
    observation_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]


# =====================================================================
# Manifest metadata extraction
# =====================================================================


def record_document_type(
    record: dict[str, Any],
) -> str:
    """Return normalized document type."""

    return (
        str(
            record.get(
                "document_type",
                "other",
            )
        )
        .strip()
        .lower()
    )


def record_content(
    record: dict[str, Any],
) -> str:
    """Return available record text."""

    return str(
        record.get(
            "content",
            "",
        )
    )


def record_ids(
    record: dict[str, Any],
    singular_field: str,
    plural_field: str,
) -> tuple[str, ...]:
    """Extract engineering IDs from singular and plural metadata."""

    values: list[str] = []

    singular = normalize_engineering_id(
        record.get(
            singular_field
        )
    )

    if singular:

        values.append(
            singular
        )

    values.extend(
        split_metadata_values(
            record.get(
                plural_field
            )
        )
    )

    return ordered_unique(
        values
    )


def record_requirement_ids(
    record: dict[str, Any],
) -> tuple[str, ...]:
    """Return requirement IDs from one record."""

    return record_ids(
        record,
        "requirement_id",
        "requirement_ids",
    )


def record_observation_ids(
    record: dict[str, Any],
) -> tuple[str, ...]:
    """Return observation IDs from one record."""

    return record_ids(
        record,
        "observation_id",
        "observation_ids",
    )


def record_test_case_ids(
    record: dict[str, Any],
) -> tuple[str, ...]:
    """Return Test Case IDs from one record."""

    return record_ids(
        record,
        "test_case_id",
        "test_case_ids",
    )


def extract_observation_status(
    record: dict[str, Any],
) -> str | None:
    """
    Extract the source-recorded engineering status of an Observation.

    Important distinction:

        records[].result_status
            -> engineering Observation status
               such as OPEN / CLOSED

        documents{}.status
            -> ingestion-ledger status
               such as INDEXED

    These two concepts must never be mixed.
    """

    if (
        record_document_type(
            record
        )
        != "observation"
    ):
        return None

    # ---------------------------------------------------------
    # Current baseline observation records store the source
    # engineering status in result_status.
    #
    # Example:
    #
    #     document_type = observation
    #     observation_id = OBS-023
    #     result_status = OPEN
    # ---------------------------------------------------------

    result_status = (
        normalize_observation_status(
            record.get(
                "result_status"
            )
        )
    )

    if result_status:
        return result_status

    # ---------------------------------------------------------
    # Phase 16H / future dynamic records may use a more explicit
    # observation-status field.
    # ---------------------------------------------------------

    explicit_fields = (
        "observation_status",
        "source_status",
        "record_status",
    )

    for field_name in explicit_fields:

        status = (
            normalize_observation_status(
                record.get(
                    field_name
                )
            )
        )

        if status:
            return status

    # ---------------------------------------------------------
    # Compatibility:
    #
    # Some evidence-record schemas may store source engineering
    # status simply as "status".
    #
    # INDEXED must not be interpreted as an engineering status.
    # normalize_observation_status() naturally rejects INDEXED.
    # ---------------------------------------------------------

    candidate = (
        normalize_observation_status(
            record.get(
                "status"
            )
        )
    )

    if candidate:
        return candidate

    # ---------------------------------------------------------
    # Final source-text fallback for dynamically parsed evidence.
    #
    # No status is fabricated if the source text contains no
    # explicit status.
    # ---------------------------------------------------------

    content_match = (
        OBSERVATION_STATUS_PATTERN
        .search(
            record_content(
                record
            )
        )
    )

    if content_match is None:
        return None

    return normalize_observation_status(
        content_match.group(
            "status"
        )
    )


def extract_test_result(
    record: dict[str, Any],
) -> str | None:
    """Extract actual Test result only from Test evidence."""

    if (
        record_document_type(
            record
        )
        != "test"
    ):

        return None

    result = normalize_test_result(
        record.get(
            "result_status"
        )
    )

    if result:

        return result

    content_match = (
        TEST_RESULT_PATTERN
        .search(
            record_content(
                record
            )
        )
    )

    if content_match is None:

        return None

    return normalize_test_result(
        content_match.group(
            "status"
        )
    )


def is_dynamic_record(
    record: dict[str, Any],
) -> bool:
    """Return whether record was dynamically ingested."""

    value = record.get(
        "dynamic_ingestion",
        False,
    )

    if isinstance(
        value,
        bool,
    ):

        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


# =====================================================================
# Inventory construction
# =====================================================================


def select_entity_ids(
    record: dict[str, Any],
) -> tuple[
    InventoryEntityType,
    tuple[str, ...],
]:
    """Return primary entity type and IDs represented by the record."""

    document_type = (
        record_document_type(
            record
        )
    )

    if document_type == "observation":

        return (
            InventoryEntityType
            .OBSERVATION,

            record_observation_ids(
                record
            ),
        )

    if document_type == "frd":

        return (
            InventoryEntityType
            .REQUIREMENT,

            record_requirement_ids(
                record
            ),
        )

    if document_type == "test":

        return (
            InventoryEntityType
            .TEST,

            record_test_case_ids(
                record
            ),
        )

    if document_type == "code":

        evidence_id = str(
            record.get(
                "evidence_id",
                "",
            )
        ).strip()

        return (
            InventoryEntityType.CODE,

            (
                evidence_id,
            )
            if evidence_id
            else (),
        )

    if document_type == "review":

        evidence_id = str(
            record.get(
                "evidence_id",
                "",
            )
        ).strip()

        return (
            InventoryEntityType.REVIEW,

            (
                evidence_id,
            )
            if evidence_id
            else (),
        )

    return (
        InventoryEntityType.REVIEW,
        (),
    )


def build_inventory_summary(
    manifest: IngestionManifest,
) -> InventorySummary:
    """Build unique engineering inventory from manifest records."""

    grouped: dict[
        tuple[
            InventoryEntityType,
            str,
        ],
        list[
            dict[str, Any]
        ],
    ] = {}

    for record in manifest.records:

        (
            entity_type,
            entity_ids,
        ) = select_entity_ids(
            record
        )

        for entity_id in entity_ids:

            grouped.setdefault(
                (
                    entity_type,
                    entity_id,
                ),
                [],
            ).append(
                record
            )

    inventory_by_type: dict[
        InventoryEntityType,
        list[
            InventoryRecord
        ],
    ] = {
        entity_type: []
        for entity_type
        in InventoryEntityType
    }

    for (
        entity_type,
        entity_id,
    ), records in grouped.items():

        evidence_ids = ordered_unique(
            str(
                record.get(
                    "evidence_id",
                    "",
                )
            ).strip()

            for record in records

            if str(
                record.get(
                    "evidence_id",
                    "",
                )
            ).strip()
        )

        document_names = ordered_unique(
            str(
                record.get(
                    "document_name",
                    record.get(
                        "file_name",
                        "",
                    ),
                )
            ).strip()

            for record in records

            if str(
                record.get(
                    "document_name",
                    record.get(
                        "file_name",
                        "",
                    ),
                )
            ).strip()
        )

        requirement_ids = ordered_unique(
            requirement_id

            for record in records

            for requirement_id
            in record_requirement_ids(
                record
            )
        )

        observation_ids = ordered_unique(
            observation_id

            for record in records

            for observation_id
            in record_observation_ids(
                record
            )
        )

        test_case_ids = ordered_unique(
            test_case_id

            for record in records

            for test_case_id
            in record_test_case_ids(
                record
            )
        )

        statuses = ordered_unique(
            status

            for record in records

            for status in [
                extract_observation_status(
                    record
                )
            ]

            if status
        )

        result_statuses = ordered_unique(
            result

            for record in records

            for result in [
                extract_test_result(
                    record
                )
            ]

            if result
        )

        inventory_by_type[
            entity_type
        ].append(
            InventoryRecord(
                entity_id=entity_id,

                entity_type=(
                    entity_type
                ),

                status=(
                    statuses[0]
                    if len(
                        statuses
                    )
                    == 1
                    else None
                ),

                document_names=(
                    document_names
                ),

                evidence_ids=(
                    evidence_ids
                ),

                requirement_ids=(
                    requirement_ids
                ),

                observation_ids=(
                    observation_ids
                ),

                test_case_ids=(
                    test_case_ids
                ),

                result_status=(
                    result_statuses[0]
                    if len(
                        result_statuses
                    )
                    == 1
                    else None
                ),

                dynamic_ingestion=any(
                    is_dynamic_record(
                        record
                    )
                    for record
                    in records
                ),
            )
        )

    def sorted_records(
        entity_type: InventoryEntityType,
    ) -> tuple[
        InventoryRecord,
        ...
    ]:

        return tuple(
            sorted(
                inventory_by_type[
                    entity_type
                ],

                key=lambda item: (
                    item.entity_id
                ),
            )
        )

    return InventorySummary(
        observations=sorted_records(
            InventoryEntityType
            .OBSERVATION
        ),

        requirements=sorted_records(
            InventoryEntityType
            .REQUIREMENT
        ),

        tests=sorted_records(
            InventoryEntityType
            .TEST
        ),

        code_records=sorted_records(
            InventoryEntityType
            .CODE
        ),

        reviews=sorted_records(
            InventoryEntityType
            .REVIEW
        ),

        manifest_version=(
            manifest.manifest_version
        ),
    )


def load_inventory_summary(
    manifest_path: str | Path,
) -> InventorySummary:
    """Load manifest and build current inventory."""

    manifest = load_ingestion_manifest(
        manifest_path
    )

    return build_inventory_summary(
        manifest
    )


# =====================================================================
# Inventory filtering
# =====================================================================


def filter_inventory(
    summary: InventorySummary,
    query: InventoryQuery,
) -> tuple[
    InventoryRecord,
    ...
]:
    """Apply deterministic inventory filters."""

    records = list(
        summary.records_for_type(
            query.entity_type
        )
    )

    if query.status_filter:

        normalized_status = (
            normalize_observation_status(
                query.status_filter
            )
        )

        records = [
            record
            for record in records
            if (
                record.status
                == normalized_status
            )
        ]

    if query.result_filter:

        normalized_result = (
            normalize_test_result(
                query.result_filter
            )
        )

        records = [
            record
            for record in records
            if (
                record.result_status
                == normalized_result
            )
        ]

    if query.requirement_id:

        requirement_id = (
            normalize_engineering_id(
                query.requirement_id
            )
        )

        records = [
            record
            for record in records
            if (
                requirement_id
                in record.requirement_ids
            )
        ]

    if query.observation_id:

        observation_id = (
            normalize_engineering_id(
                query.observation_id
            )
        )

        records = [
            record
            for record in records
            if (
                observation_id
                in record.observation_ids
            )
        ]

    if query.test_case_id:

        test_case_id = (
            normalize_engineering_id(
                query.test_case_id
            )
        )

        records = [
            record
            for record in records
            if (
                test_case_id
                in record.test_case_ids
            )
        ]

    return tuple(
        records
    )


# =====================================================================
# Entity lookup and evidence availability
# =====================================================================


def find_inventory_entity(
    summary: InventorySummary,
    entity_id: str,
) -> InventoryRecord | None:
    """Find one inventory entity by normalized engineering ID."""

    normalized_id = normalize_engineering_id(entity_id)
    if normalized_id is None:
        return None
    for record in summary.all_records():
        if record.entity_id == normalized_id:
            return record
    return None


def lookup_inventory_entity(
    summary: InventorySummary,
    *,
    entity_id: str,
) -> InventoryAnswer:
    """Build a concise deterministic engineering-entity lookup."""

    record = find_inventory_entity(summary, entity_id)
    if record is None:
        return InventoryAnswer(
            query=InventoryQuery(
                operation=InventoryOperation.LOOKUP,
                entity_type=InventoryEntityType.REQUIREMENT,
            ),
            count=0,
            records=(),
            answer_text=(
                f"{entity_id} was not found in the currently indexed evidence set."
            ),
            source_statement=(
                "Result calculated deterministically from the current "
                "synchronized evidence manifest."
            ),
        )

    lines = [record.entity_id]
    if record.status:
        lines.append(f"Status: {record.status}")
    if record.result_status:
        lines.append(f"Result: {record.result_status}")
    if record.requirement_ids:
        lines.append("Requirements: " + ", ".join(record.requirement_ids))
    if record.observation_ids:
        lines.append("Observations: " + ", ".join(record.observation_ids))
    if record.test_case_ids:
        lines.append("Test Cases: " + ", ".join(record.test_case_ids))
    if record.document_names:
        lines.append("Source: " + ", ".join(record.document_names))

    return InventoryAnswer(
        query=InventoryQuery(
            operation=InventoryOperation.LOOKUP,
            entity_type=record.entity_type,
        ),
        count=1,
        records=(record,),
        answer_text="\n".join(lines),
        source_statement=(
            "Result calculated deterministically from the current "
            "synchronized evidence manifest."
        ),
    )


def requirement_evidence_availability(
    summary: InventorySummary,
) -> tuple[EvidenceAvailabilityRecord, ...]:
    """Determine actual downstream evidence availability by requirement.

    TC references in FRD, Observation, or Code metadata remain traceability
    only. Actual Test evidence requires an indexed Test entity.
    """

    output: list[EvidenceAvailabilityRecord] = []
    for requirement in summary.requirements:
        requirement_id = requirement.entity_id
        observations = tuple(
            item for item in summary.observations
            if requirement_id in item.requirement_ids
        )
        code_records = tuple(
            item for item in summary.code_records
            if requirement_id in item.requirement_ids
        )
        tests = tuple(
            item for item in summary.tests
            if requirement_id in item.requirement_ids
        )
        output.append(
            EvidenceAvailabilityRecord(
                requirement_id=requirement_id,
                has_requirement_evidence=True,
                has_observation_evidence=bool(observations),
                has_code_evidence=bool(code_records),
                has_test_evidence=bool(tests),
                observation_ids=ordered_unique(
                    obs_id for item in observations for obs_id in item.observation_ids
                ),
                test_case_ids=ordered_unique(
                    test_id for item in tests for test_id in item.test_case_ids
                ),
            )
        )
    return tuple(output)


def requirements_missing_actual_test_evidence(
    summary: InventorySummary,
) -> tuple[str, ...]:
    """Return requirements without actual Test-document evidence."""

    return tuple(
        item.requirement_id
        for item in requirement_evidence_availability(summary)
        if not item.has_test_evidence
    )


def requirements_missing_code_evidence(
    summary: InventorySummary,
) -> tuple[str, ...]:
    """Return requirements without actual source-code evidence."""

    return tuple(
        item.requirement_id
        for item in requirement_evidence_availability(summary)
        if not item.has_code_evidence
    )


def requirements_missing_observation_evidence(
    summary: InventorySummary,
) -> tuple[str, ...]:
    """Return requirements without actual Observation evidence."""

    return tuple(
        item.requirement_id
        for item in requirement_evidence_availability(summary)
        if not item.has_observation_evidence
    )


def answer_requirement_evidence_availability(
    summary: InventorySummary,
    *,
    evidence_type: str,
) -> InventoryAnswer:
    """Answer which requirements lack one actual evidence type."""

    normalized_type = evidence_type.strip().lower()
    if normalized_type == "test":
        missing_ids = requirements_missing_actual_test_evidence(summary)
        label = "actual Test evidence"
    elif normalized_type == "code":
        missing_ids = requirements_missing_code_evidence(summary)
        label = "source-code evidence"
    elif normalized_type in {"observation", "observations"}:
        missing_ids = requirements_missing_observation_evidence(summary)
        label = "Observation evidence"
    else:
        raise EvidenceInventoryError(
            f"Unsupported evidence-availability type: {evidence_type!r}."
        )

    requirement_index = {item.entity_id: item for item in summary.requirements}
    records = tuple(requirement_index[item_id] for item_id in missing_ids)
    if missing_ids:
        answer_text = (
            f"{len(missing_ids)} requirements lack {label} in the currently indexed "
            "evidence set.\n\n" + ", ".join(missing_ids)
        )
    else:
        answer_text = f"No indexed requirements lack {label}."

    return InventoryAnswer(
        query=InventoryQuery(
            operation=InventoryOperation.EVIDENCE_AVAILABILITY,
            entity_type=InventoryEntityType.REQUIREMENT,
        ),
        count=len(records),
        records=records,
        answer_text=answer_text,
        source_statement=(
            "Evidence availability calculated deterministically from actual "
            "indexed evidence types. Traceability references alone are not "
            "treated as evidence."
        ),
    )


# =====================================================================
# Answer generation
# =====================================================================


def plural_label(
    entity_type: InventoryEntityType,
) -> str:
    """Return user-facing plural entity label."""

    labels = {
        InventoryEntityType
        .OBSERVATION: "observations",

        InventoryEntityType
        .REQUIREMENT: "requirements",

        InventoryEntityType
        .TEST: "Test Cases",

        InventoryEntityType
        .CODE: "source-code evidence records",

        InventoryEntityType
        .REVIEW: "review evidence records",
    }

    return labels[
        entity_type
    ]


def answer_inventory_query(
    summary: InventorySummary,
    query: InventoryQuery,
) -> InventoryAnswer:
    """Execute deterministic count/list inventory query."""

    records = filter_inventory(
        summary,
        query,
    )

    label = plural_label(
        query.entity_type
    )

    qualifiers: list[str] = []

    if query.status_filter:

        normalized_status = (
            normalize_observation_status(
                query.status_filter
            )
            or query.status_filter.upper()
        )

        qualifiers.append(
            normalized_status
        )

    if query.result_filter:

        normalized_result = (
            normalize_test_result(
                query.result_filter
            )
            or query.result_filter.upper()
        )

        qualifiers.append(
            normalized_result
        )

    if query.requirement_id:

        qualifiers.append(
            f"related to {query.requirement_id}"
        )

    qualifier_text = (
        " ".join(
            qualifiers
        )
    )

    entity_ids = tuple(
        record.entity_id
        for record in records
    )

    if query.operation == InventoryOperation.COUNT:

        if qualifier_text:

            answer_text = (
                f"{len(records)} {qualifier_text} "
                f"{label} are present in the "
                "currently indexed evidence set."
            )

        else:

            answer_text = (
                f"{len(records)} {label} are present "
                "in the currently indexed evidence set."
            )

        if entity_ids:

            answer_text += (
                "\n\n"
                + ", ".join(
                    entity_ids
                )
            )

    elif query.operation == InventoryOperation.LIST:

        if entity_ids:

            heading = (
                f"{qualifier_text} {label}"
                if qualifier_text
                else label
            )

            answer_text = (
                f"{heading.capitalize()}:\n\n"
                + "\n".join(
                    f"- {entity_id}"
                    for entity_id
                    in entity_ids
                )
            )

        else:

            answer_text = (
                f"No matching {label} were found "
                "in the currently indexed evidence set."
            )

    else:

        raise EvidenceInventoryError(
            "This Phase 16I.2 handler supports "
            "COUNT and LIST operations only."
        )

    return InventoryAnswer(
        query=query,

        count=len(
            records
        ),

        records=records,

        answer_text=(
            answer_text
        ),

        source_statement=(
            "Result calculated deterministically from "
            "the current synchronized evidence manifest."
        ),
    )
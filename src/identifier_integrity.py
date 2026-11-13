"""
Explicit engineering-identifier integrity for REQTRACE AI.

This module prevents semantic retrieval from silently replacing an
explicitly requested engineering identifier.

Example unsafe behavior prevented:

    Query:
        OBS-999

    Semantic retrieval:
        OBS-052

    Incorrect result:
        Decision about OBS-052 presented as the answer to OBS-999

Required behavior:

    Query:
        OBS-999

    Result:
        ExplicitIdentifierNotFound

The gate validates explicit IDs against structured evidence metadata,
not against free-text content.

Supported ID types:

- REQ-NNN
- OBS-NNN
- TC-NNN
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.context_builder import (
    FinalEvidenceContext,
)


class EngineeringIdentifierType(
    str,
    Enum,
):
    """Supported explicit engineering identifier types."""

    REQUIREMENT = "REQUIREMENT"
    OBSERVATION = "OBSERVATION"
    TEST_CASE = "TEST_CASE"


EngineeringIdentifierType.__test__ = False


class IdentifierIntegrityError(
    RuntimeError
):
    """Base error for explicit-identifier validation."""


class ExplicitIdentifierNotFound(
    IdentifierIntegrityError
):
    """
    Raised when an explicitly requested ID is absent from
    authoritative structured evidence.
    """

    def __init__(
        self,
        *,
        missing_ids: tuple[str, ...],
        requested_ids: tuple[str, ...],
        available_ids: tuple[str, ...],
    ) -> None:

        self.missing_ids = missing_ids
        self.requested_ids = requested_ids
        self.available_ids = available_ids

        missing_text = ", ".join(
            missing_ids
        )

        super().__init__(
            "No authoritative evidence was found for "
            f"the explicitly requested identifier(s): "
            f"{missing_text}. "
            "REQTRACE AI did not substitute another "
            "engineering entity."
        )


class ExplicitIdentifierMismatch(
    IdentifierIntegrityError
):
    """
    Raised when retrieved authoritative IDs conflict with the
    explicitly requested identity.
    """


@dataclass(frozen=True)
class ExplicitIdentifier:
    """One normalized engineering identifier."""

    value: str
    identifier_type: EngineeringIdentifierType

    def __post_init__(
        self,
    ) -> None:

        if not self.value.strip():
            raise ValueError(
                "Explicit identifier must not be empty."
            )


@dataclass(frozen=True)
class IdentifierIntegrityResult:
    """Result of explicit-identifier validation."""

    passed: bool

    requested_ids: tuple[str, ...]
    available_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]

    @property
    def explicit_identifiers_present(
        self,
    ) -> bool:
        """Return whether the query contained explicit IDs."""

        return bool(
            self.requested_ids
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "passed": self.passed,
            "explicit_identifiers_present": (
                self.explicit_identifiers_present
            ),
            "requested_ids": list(
                self.requested_ids
            ),
            "available_ids": list(
                self.available_ids
            ),
            "missing_ids": list(
                self.missing_ids
            ),
        }


ENGINEERING_ID_PATTERN = re.compile(
    r"\b(?P<prefix>REQ|OBS|TC)"
    r"[\s_-]?"
    r"(?P<number>\d+)\b",
    re.IGNORECASE,
)


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered, normalized, duplicate-free values."""

    output: list[str] = []
    seen: set[str] = set()

    for value in values:

        normalized = str(
            value
        ).strip()

        if not normalized:
            continue

        if normalized in seen:
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


def normalize_engineering_id(
    value: str,
) -> str:
    """
    Normalize an engineering identifier.

    Examples:

        obs_023  -> OBS-023
        OBS 023  -> OBS-023
        req-50   -> REQ-50
    """

    match = ENGINEERING_ID_PATTERN.fullmatch(
        value.strip()
    )

    if match is None:
        return (
            value
            .strip()
            .upper()
            .replace("_", "-")
            .replace(" ", "-")
        )

    prefix = (
        match.group(
            "prefix"
        )
        .upper()
    )

    number = match.group(
        "number"
    )

    return (
        f"{prefix}-{number}"
    )


def identifier_type_from_prefix(
    prefix: str,
) -> EngineeringIdentifierType:
    """Map an identifier prefix to its type."""

    normalized = (
        prefix
        .strip()
        .upper()
    )

    if normalized == "REQ":
        return (
            EngineeringIdentifierType
            .REQUIREMENT
        )

    if normalized == "OBS":
        return (
            EngineeringIdentifierType
            .OBSERVATION
        )

    if normalized == "TC":
        return (
            EngineeringIdentifierType
            .TEST_CASE
        )

    raise ValueError(
        "Unsupported engineering identifier "
        f"prefix: {prefix}"
    )


def extract_explicit_identifiers(
    query: str,
) -> tuple[
    ExplicitIdentifier,
    ...
]:
    """
    Extract explicit engineering IDs from a query.

    Natural-language queries without explicit IDs return an empty tuple.
    """

    identifiers: list[
        ExplicitIdentifier
    ] = []

    seen: set[str] = set()

    for match in (
        ENGINEERING_ID_PATTERN.finditer(
            query
        )
    ):

        prefix = (
            match.group(
                "prefix"
            )
            .upper()
        )

        number = match.group(
            "number"
        )

        value = (
            f"{prefix}-{number}"
        )

        if value in seen:
            continue

        seen.add(
            value
        )

        identifiers.append(
            ExplicitIdentifier(
                value=value,
                identifier_type=(
                    identifier_type_from_prefix(
                        prefix
                    )
                ),
            )
        )

    return tuple(
        identifiers
    )


def optional_metadata_id(
    metadata: dict[str, Any],
    field_name: str,
) -> str | None:
    """Return one normalized structured metadata ID."""

    value = metadata.get(
        field_name
    )

    if value is None:
        return None

    normalized = str(
        value
    ).strip()

    if not normalized:
        return None

    return normalize_engineering_id(
        normalized
    )


def authoritative_ids_from_context(
    context: FinalEvidenceContext,
) -> tuple[str, ...]:
    """
    Collect authoritative IDs from structured evidence metadata.

    Free-text content is deliberately excluded.

    This prevents accidental identity validation merely because an
    unrelated document mentions a requested ID in prose.
    """

    values: list[str] = []

    metadata_fields = (
        "requirement_id",
        "observation_id",
        "test_case_id",
    )

    for evidence in context.evidence:

        metadata = dict(
            evidence.metadata
        )

        for field_name in metadata_fields:

            identifier = optional_metadata_id(
                metadata,
                field_name,
            )

            if identifier:
                values.append(
                    identifier
                )

    return ordered_unique(
        values
    )


def validate_explicit_identifier_integrity(
    *,
    query: str,
    context: FinalEvidenceContext,
    strict: bool = True,
) -> IdentifierIntegrityResult:
    """
    Validate all explicitly requested engineering IDs.

    If the query contains no explicit engineering ID, semantic and
    lexical retrieval remain permitted.

    If one or more explicit IDs are present, every requested ID must
    be represented by structured evidence metadata.
    """

    requested = extract_explicit_identifiers(
        query
    )

    requested_ids = tuple(
        identifier.value
        for identifier in requested
    )

    available_ids = (
        authoritative_ids_from_context(
            context
        )
    )

    available_set = set(
        available_ids
    )

    missing_ids = tuple(
        identifier
        for identifier in requested_ids
        if identifier not in available_set
    )

    result = IdentifierIntegrityResult(
        passed=not missing_ids,
        requested_ids=requested_ids,
        available_ids=available_ids,
        missing_ids=missing_ids,
    )

    if (
        strict
        and missing_ids
    ):
        raise ExplicitIdentifierNotFound(
            missing_ids=missing_ids,
            requested_ids=requested_ids,
            available_ids=available_ids,
        )

    return result
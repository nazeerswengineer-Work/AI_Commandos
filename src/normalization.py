"""
Engineering identifier and content normalization for REQTRACE AI.

This module contains deterministic helpers only. It does not call an LLM,
perform semantic matching, or create permanent traceability links.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


REQUIREMENT_ID_PATTERN = re.compile(
    r"(?<![A-Z0-9])REQ[\s_-]*(\d+)(?![A-Z0-9])",
    re.IGNORECASE,
)

OBSERVATION_ID_PATTERN = re.compile(
    r"(?<![A-Z0-9])OBS[\s_-]*(\d+)(?![A-Z0-9])",
    re.IGNORECASE,
)

TEST_CASE_ID_PATTERN = re.compile(
    r"(?<![A-Z0-9])TC[\s_-]*(\d+)(?![A-Z0-9])",
    re.IGNORECASE,
)

GENERIC_ENGINEERING_ID_PATTERN = re.compile(
    r"^(REQ|OBS|TC)[\s_-]*(\d+)$",
    re.IGNORECASE,
)

REVISION_PATTERN = re.compile(
    r"^(?:REV(?:ISION)?[\s_-]*)?(\d+)$",
    re.IGNORECASE,
)

WHITESPACE_PATTERN = re.compile(r"\s+")


class EngineeringIDError(ValueError):
    """Raised when an engineering identifier is invalid or ambiguous."""


def normalize_unicode(value: str) -> str:
    """
    Normalize Unicode text without changing ordinary engineering content.

    NFKC normalization converts compatible characters into a consistent
    representation while preserving readable text.
    """

    if not isinstance(value, str):
        raise TypeError("Expected a string value.")

    return unicodedata.normalize("NFKC", value)


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace into one ordinary space."""

    normalized = normalize_unicode(value)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_multiline_content(value: str) -> str:
    """
    Normalize content for deterministic hashing.

    Rules:
    - normalize Unicode
    - normalize line endings
    - remove trailing whitespace
    - remove leading and trailing blank lines
    - preserve meaningful line boundaries
    """

    normalized = normalize_unicode(value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in normalized.split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def _normalize_prefixed_numeric_id(
    value: str,
    expected_prefix: str,
) -> str:
    """Normalize one prefixed numeric engineering identifier."""

    cleaned = normalize_whitespace(value).upper()

    match = GENERIC_ENGINEERING_ID_PATTERN.fullmatch(cleaned)

    if match is None:
        raise EngineeringIDError(
            f"Invalid {expected_prefix} identifier: {value!r}."
        )

    actual_prefix = match.group(1).upper()
    numeric_part = match.group(2)

    if actual_prefix != expected_prefix:
        raise EngineeringIDError(
            f"Expected a {expected_prefix} identifier but received "
            f"{actual_prefix}."
        )

    normalized_number = str(int(numeric_part))

    if len(normalized_number) < 3:
        normalized_number = normalized_number.zfill(3)

    return f"{expected_prefix}-{normalized_number}"


def normalize_requirement_id(value: str) -> str:
    """Normalize a requirement ID to REQ-NNN notation."""

    return _normalize_prefixed_numeric_id(value, "REQ")


def normalize_observation_id(value: str) -> str:
    """Normalize an observation ID to OBS-NNN notation."""

    return _normalize_prefixed_numeric_id(value, "OBS")


def normalize_test_case_id(value: str) -> str:
    """Normalize a test-case ID to TC-NNN notation."""

    return _normalize_prefixed_numeric_id(value, "TC")


def normalize_engineering_id(value: str) -> str:
    """Normalize any supported engineering identifier."""

    cleaned = normalize_whitespace(value).upper()
    match = GENERIC_ENGINEERING_ID_PATTERN.fullmatch(cleaned)

    if match is None:
        raise EngineeringIDError(
            f"Unsupported engineering identifier: {value!r}."
        )

    prefix = match.group(1).upper()

    if prefix == "REQ":
        return normalize_requirement_id(cleaned)

    if prefix == "OBS":
        return normalize_observation_id(cleaned)

    if prefix == "TC":
        return normalize_test_case_id(cleaned)

    raise EngineeringIDError(
        f"Unsupported engineering ID prefix: {prefix}."
    )


def normalize_optional_requirement_id(
    value: str | None,
) -> str | None:
    """Normalize a requirement ID while preserving a missing value."""

    if value is None:
        return None

    if not value.strip():
        return None

    return normalize_requirement_id(value)


def normalize_optional_observation_id(
    value: str | None,
) -> str | None:
    """Normalize an observation ID while preserving a missing value."""

    if value is None:
        return None

    if not value.strip():
        return None

    return normalize_observation_id(value)


def normalize_optional_test_case_id(
    value: str | None,
) -> str | None:
    """Normalize a test-case ID while preserving a missing value."""

    if value is None:
        return None

    if not value.strip():
        return None

    return normalize_test_case_id(value)


def normalize_revision(value: str | None) -> str | None:
    """
    Normalize a numeric revision to RevNN.

    Examples:
        1 -> Rev01
        Rev1 -> Rev01
        REV_02 -> Rev02

    Non-numeric project-specific revisions are rejected rather than guessed.
    """

    if value is None:
        return None

    cleaned = normalize_whitespace(value)

    if not cleaned:
        return None

    match = REVISION_PATTERN.fullmatch(cleaned)

    if match is None:
        raise ValueError(
            f"Invalid revision value: {value!r}. "
            "Expected a numeric revision such as Rev01."
        )

    revision_number = int(match.group(1))
    return f"Rev{revision_number:02d}"


def _extract_ids(
    text: str,
    pattern: re.Pattern[str],
    prefix: str,
) -> tuple[str, ...]:
    """Extract ordered, duplicate-free engineering IDs."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    identifiers: list[str] = []
    seen: set[str] = set()

    for match in pattern.finditer(text):
        identifier = _normalize_prefixed_numeric_id(
            f"{prefix}-{match.group(1)}",
            prefix,
        )

        if identifier not in seen:
            seen.add(identifier)
            identifiers.append(identifier)

    return tuple(identifiers)


def extract_requirement_ids(text: str) -> tuple[str, ...]:
    """Extract normalized requirement IDs in source order."""

    return _extract_ids(
        text,
        REQUIREMENT_ID_PATTERN,
        "REQ",
    )


def extract_observation_ids(text: str) -> tuple[str, ...]:
    """Extract normalized observation IDs in source order."""

    return _extract_ids(
        text,
        OBSERVATION_ID_PATTERN,
        "OBS",
    )


def extract_test_case_ids(text: str) -> tuple[str, ...]:
    """Extract normalized test-case IDs in source order."""

    return _extract_ids(
        text,
        TEST_CASE_ID_PATTERN,
        "TC",
    )


def ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return ordered, duplicate-free non-empty strings."""

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = normalize_whitespace(value)

        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)

    return tuple(result)
"""
Natural-language parser for deterministic evidence-inventory queries.

This module converts COUNT and LIST questions into the structured
InventoryQuery model used by Phase 16I.2.

Examples:

    How many observations are open?

        InventoryQuery(
            operation=COUNT,
            entity_type=OBSERVATION,
            status_filter="OPEN",
        )

    List failed Test Cases related to REQ-017.

        InventoryQuery(
            operation=LIST,
            entity_type=TEST,
            result_filter="FAIL",
            requirement_id="REQ-017",
        )

The parser never guesses an unsupported status or entity type.
"""

from __future__ import annotations

import re

from dataclasses import dataclass

from src.evidence_inventory import (
    InventoryEntityType,
    InventoryOperation,
    InventoryQuery,
)

from src.query_intent import (
    QueryIntent,
    QueryIntentResult,
    classify_query_intent,
    normalize_query,
)


class InventoryQueryParseError(
    ValueError
):
    """Raised when an inventory query cannot be parsed safely."""


@dataclass(frozen=True)
class ParsedInventoryRequest:
    """Complete deterministic inventory-query interpretation."""

    original_query: str

    intent_result: QueryIntentResult

    inventory_query: InventoryQuery

    matched_entity_rule: str

    matched_filter_rules: tuple[
        str,
        ...
    ]

    evidence_type: str | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate structured parse result."""

        if not self.original_query.strip():

            raise InventoryQueryParseError(
                "original_query must not be empty."
            )

        if not self.matched_entity_rule.strip():

            raise InventoryQueryParseError(
                "matched_entity_rule must not be empty."
            )


# =====================================================================
# Engineering identifiers
# =====================================================================


ENGINEERING_ID_PATTERN = re.compile(
    r"\b(?P<prefix>REQ|OBS|TC)"
    r"[\s_-]?"
    r"(?P<number>\d+)\b",
    re.IGNORECASE,
)


def normalize_identifier(
    prefix: str,
    number: str,
) -> str:
    """Normalize explicit engineering identifier."""

    return (
        f"{prefix.upper()}-{number}"
    )


def extract_engineering_ids(
    query: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Extract REQ, OBS and TC identifiers from query."""

    requirements: list[str] = []

    observations: list[str] = []

    test_cases: list[str] = []

    seen: set[str] = set()

    for match in (
        ENGINEERING_ID_PATTERN
        .finditer(
            query
        )
    ):

        prefix = (
            match.group(
                "prefix"
            )
            .upper()
        )

        identifier = normalize_identifier(
            prefix,
            match.group(
                "number"
            ),
        )

        if identifier in seen:

            continue

        seen.add(
            identifier
        )

        if prefix == "REQ":

            requirements.append(
                identifier
            )

        elif prefix == "OBS":

            observations.append(
                identifier
            )

        elif prefix == "TC":

            test_cases.append(
                identifier
            )

    return (
        tuple(
            requirements
        ),

        tuple(
            observations
        ),

        tuple(
            test_cases
        ),
    )


# =====================================================================
# Entity classification
# =====================================================================


ENTITY_PATTERNS = (
    (
        InventoryEntityType
        .OBSERVATION,

        "IQP-ENTITY-OBSERVATION",

        (
            re.compile(
                r"\bobservations?\b"
            ),

            re.compile(
                r"\bobs[\s_-]?\d+\b"
            ),
        ),
    ),

    (
        InventoryEntityType
        .REQUIREMENT,

        "IQP-ENTITY-REQUIREMENT",

        (
            re.compile(
                r"\brequirements?\b"
            ),

            re.compile(
                r"\bfrds?\b"
            ),

            re.compile(
                r"\breq[\s_-]?\d+\b"
            ),
        ),
    ),

    (
        InventoryEntityType.TEST,

        "IQP-ENTITY-TEST",

        (
            re.compile(
                r"\btest cases?\b"
            ),

            re.compile(
                r"\btests?\b"
            ),

            re.compile(
                r"\btc[\s_-]?\d+\b"
            ),
        ),
    ),

    (
        InventoryEntityType.CODE,

        "IQP-ENTITY-CODE",

        (
            re.compile(
                r"\bsource code\b"
            ),

            re.compile(
                r"\bcode evidence\b"
            ),

            re.compile(
                r"\bimplementation evidence\b"
            ),
        ),
    ),

    (
        InventoryEntityType.REVIEW,

        "IQP-ENTITY-REVIEW",

        (
            re.compile(
                r"\breviews?\b"
            ),

            re.compile(
                r"\breview evidence\b"
            ),
        ),
    ),
)


def detect_entity_type(
    normalized_query: str,
) -> tuple[
    InventoryEntityType,
    str,
]:
    """Detect the primary inventory entity type."""

    matches: list[
        tuple[
            InventoryEntityType,
            str,
        ]
    ] = []

    for (
        entity_type,
        rule,
        patterns,
    ) in ENTITY_PATTERNS:

        if any(
            pattern.search(
                normalized_query
            )
            is not None

            for pattern
            in patterns
        ):

            matches.append(
                (
                    entity_type,
                    rule,
                )
            )

    if not matches:

        raise InventoryQueryParseError(
            "The inventory entity type could not be "
            "determined. Specify observations, "
            "requirements, Test Cases, code evidence, "
            "or review evidence."
        )

    unique_types = {
        entity_type
        for entity_type, _
        in matches
    }

    if len(
        unique_types
    ) > 1:

        # Relationship queries often contain an entity collection and
        # a filter entity, for example:
        #
        # Which observations reference REQ-017?
        #
        # OBSERVATION remains the requested inventory entity because
        # the explicit REQ ID is a relationship filter.
        plural_entity_matches = [
            (
                entity_type,
                rule,
            )
            for (
                entity_type,
                rule,
            )
            in matches
            if (
                entity_type
                == InventoryEntityType
                .OBSERVATION
                and re.search(
                    r"\bobservations?\b",
                    normalized_query,
                )
            )
            or (
                entity_type
                == InventoryEntityType
                .REQUIREMENT
                and re.search(
                    r"\brequirements?\b",
                    normalized_query,
                )
            )
            or (
                entity_type
                == InventoryEntityType.TEST
                and re.search(
                    r"\b(?:test cases?|tests?)\b",
                    normalized_query,
                )
            )
        ]

        if len(
            plural_entity_matches
        ) == 1:

            return (
                plural_entity_matches[0]
            )

        raise InventoryQueryParseError(
            "The query contains multiple inventory "
            "entity types and cannot be aggregated "
            "safely as one result."
        )

    return matches[0]


# =====================================================================
# Operation
# =====================================================================


def operation_from_intent(
    intent: QueryIntent,
) -> InventoryOperation:
    """Map deterministic query intent to inventory operation."""

    if intent == QueryIntent.COUNT:

        return InventoryOperation.COUNT

    if intent == QueryIntent.LIST:

        return InventoryOperation.LIST

    if (
        intent
        == QueryIntent
        .EVIDENCE_AVAILABILITY
    ):

        return (
            InventoryOperation
            .EVIDENCE_AVAILABILITY
        )

    if intent == QueryIntent.LOOKUP:

        return InventoryOperation.LOOKUP

    raise InventoryQueryParseError(
        f"Query intent {intent.value!r} is not "
        "handled by the inventory parser."
    )


# =====================================================================
# Status and result filters
# =====================================================================


OBSERVATION_STATUS_RULES = (
    (
        "OPEN",
        "IQP-FILTER-OBS-OPEN",
        (
            re.compile(
                r"\bopen\b"
            ),
        ),
    ),

    (
        "CLOSED",
        "IQP-FILTER-OBS-CLOSED",
        (
            re.compile(
                r"\bclosed\b"
            ),

            re.compile(
                r"\bclosure completed\b"
            ),
        ),
    ),

    (
        "RESOLVED",
        "IQP-FILTER-OBS-RESOLVED",
        (
            re.compile(
                r"\bresolved\b"
            ),
        ),
    ),

    (
        "REJECTED",
        "IQP-FILTER-OBS-REJECTED",
        (
            re.compile(
                r"\brejected\b"
            ),
        ),
    ),

    (
        "DEFERRED",
        "IQP-FILTER-OBS-DEFERRED",
        (
            re.compile(
                r"\bdeferred\b"
            ),
        ),
    ),

    (
        "IN PROGRESS",
        "IQP-FILTER-OBS-IN-PROGRESS",
        (
            re.compile(
                r"\bin progress\b"
            ),
        ),
    ),

    (
        "UNDER REVIEW",
        "IQP-FILTER-OBS-UNDER-REVIEW",
        (
            re.compile(
                r"\bunder review\b"
            ),
        ),
    ),
)


TEST_RESULT_RULES = (
    (
        "FAIL",
        "IQP-FILTER-TEST-FAIL",
        (
            re.compile(
                r"\bfailed?\b"
            ),

            re.compile(
                r"\bfailing\b"
            ),
        ),
    ),

    (
        "PASS",
        "IQP-FILTER-TEST-PASS",
        (
            re.compile(
                r"\bpassed?\b"
            ),

            re.compile(
                r"\bpassing\b"
            ),
        ),
    ),

    (
        "NOT EXECUTED",
        "IQP-FILTER-TEST-NOT-EXECUTED",
        (
            re.compile(
                r"\bnot executed\b"
            ),

            re.compile(
                r"\bnot run\b"
            ),

            re.compile(
                r"\bunexecuted\b"
            ),
        ),
    ),

    (
        "BLOCKED",
        "IQP-FILTER-TEST-BLOCKED",
        (
            re.compile(
                r"\bblocked\b"
            ),
        ),
    ),
)


def detect_filter(
    normalized_query: str,
    rules,
) -> tuple[
    str | None,
    str | None,
]:
    """Detect at most one deterministic filter value."""

    matches: list[
        tuple[
            str,
            str,
        ]
    ] = []

    for (
        value,
        rule,
        patterns,
    ) in rules:

        if any(
            pattern.search(
                normalized_query
            )
            is not None

            for pattern
            in patterns
        ):

            matches.append(
                (
                    value,
                    rule,
                )
            )

    if not matches:

        return (
            None,
            None,
        )

    unique_values = {
        value
        for value, _
        in matches
    }

    if len(
        unique_values
    ) > 1:

        raise InventoryQueryParseError(
            "The query contains conflicting status "
            "or result filters."
        )

    return matches[0]


# =====================================================================
# Complete parser
# =====================================================================


def parse_inventory_query(
    query: str,
    *,
    intent_result: (
        QueryIntentResult
        | None
    ) = None,
) -> ParsedInventoryRequest:
    """Parse natural-language inventory question deterministically."""

    normalized = normalize_query(
        query
    )

    active_intent = (
        intent_result
        if intent_result is not None
        else classify_query_intent(
            query
        )
    )

    supported_intents = {
        QueryIntent.COUNT,
        QueryIntent.LIST,
        QueryIntent.LOOKUP,
        QueryIntent
        .EVIDENCE_AVAILABILITY,
    }

    if (
        active_intent.intent
        not in supported_intents
    ):

        raise InventoryQueryParseError(
            "This query is not an inventory-oriented "
            "request and must be handled by another "
            "REQTRACE AI workflow."
        )

    operation = operation_from_intent(
        active_intent.intent
    )

    # ---------------------------------------------------------
    # Evidence-availability queries contain two conceptual nouns.
    #
    # Example:
    #
    #     Which requirements have missing test evidence?
    #
    # "requirements" is the requested inventory entity.
    # "test evidence" is the evidence-availability filter.
    #
    # Remove the evidence-type phrase before primary entity
    # detection so Test/Code/Observation evidence is not
    # misinterpreted as a second inventory entity.
    # ---------------------------------------------------------

    evidence_type: str | None = None
    entity_detection_query = normalized

    if active_intent.intent == QueryIntent.EVIDENCE_AVAILABILITY:
        if re.search(r"\btest evidence\b", normalized):
            evidence_type = "test"
            entity_detection_query = re.sub(
                r"\btest evidence\b",
                "evidence",
                entity_detection_query,
            )
        elif re.search(
            r"\b(?:code|implementation) evidence\b",
            normalized,
        ):
            evidence_type = "code"
            entity_detection_query = re.sub(
                r"\b(?:code|implementation) evidence\b",
                "evidence",
                entity_detection_query,
            )
        elif re.search(r"\bobservation evidence\b", normalized):
            evidence_type = "observation"
            entity_detection_query = re.sub(
                r"\bobservation evidence\b",
                "evidence",
                entity_detection_query,
            )
        else:
            raise InventoryQueryParseError(
                "Evidence-availability query must identify Test, code, "
                "or Observation evidence."
            )

    (
        entity_type,
        entity_rule,
    ) = detect_entity_type(
        entity_detection_query
    )

    matched_filters: list[str] = []

    status_filter: str | None = None

    result_filter: str | None = None

    if (
        entity_type
        == InventoryEntityType
        .OBSERVATION
    ):

        (
            status_filter,
            status_rule,
        ) = detect_filter(
            normalized,
            OBSERVATION_STATUS_RULES,
        )

        if status_rule:

            matched_filters.append(
                status_rule
            )

    if (
        entity_type
        == InventoryEntityType.TEST
    ):

        (
            result_filter,
            result_rule,
        ) = detect_filter(
            normalized,
            TEST_RESULT_RULES,
        )

        if result_rule:

            matched_filters.append(
                result_rule
            )

    (
        requirement_ids,
        observation_ids,
        test_case_ids,
    ) = extract_engineering_ids(
        normalized
    )

    # An ID matching the primary entity is treated as lookup identity.
    # IDs of other entity types become relationship filters.
    requirement_filter = (
        requirement_ids[0]
        if (
            requirement_ids
            and entity_type
            != InventoryEntityType
            .REQUIREMENT
        )
        else None
    )

    observation_filter = (
        observation_ids[0]
        if (
            observation_ids
            and entity_type
            != InventoryEntityType
            .OBSERVATION
        )
        else None
    )

    test_case_filter = (
        test_case_ids[0]
        if (
            test_case_ids
            and entity_type
            != InventoryEntityType.TEST
        )
        else None
    )

    if requirement_filter:

        matched_filters.append(
            "IQP-FILTER-REQUIREMENT-ID"
        )

    if observation_filter:

        matched_filters.append(
            "IQP-FILTER-OBSERVATION-ID"
        )

    if test_case_filter:

        matched_filters.append(
            "IQP-FILTER-TEST-ID"
        )

    return ParsedInventoryRequest(
        original_query=query,

        intent_result=(
            active_intent
        ),

        inventory_query=(
            InventoryQuery(
                operation=operation,

                entity_type=(
                    entity_type
                ),

                status_filter=(
                    status_filter
                ),

                result_filter=(
                    result_filter
                ),

                requirement_id=(
                    requirement_filter
                ),

                observation_id=(
                    observation_filter
                ),

                test_case_id=(
                    test_case_filter
                ),
            )
        ),

        matched_entity_rule=(
            entity_rule
        ),

        matched_filter_rules=tuple(
            matched_filters
        ),

        evidence_type=evidence_type,
    )
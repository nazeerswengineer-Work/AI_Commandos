"""
Deterministic query-intent classification for REQTRACE AI.

The router determines which engineering workflow should answer
the user's query.

The LLM does not classify query intent.

Supported intents:

- COUNT
- LIST
- LOOKUP
- EVIDENCE_AVAILABILITY
- COMPARE
- ASSESSMENT
- CLOSURE
- GENERAL

Routing precedence is important.

Examples:

    How many observations are closed?
        -> COUNT

    List all closed observations
        -> LIST

    Which requirements have missing Test evidence?
        -> EVIDENCE_AVAILABILITY

    Can OBS-070 be closed?
        -> CLOSURE

    Is OBS-023 supported?
        -> ASSESSMENT
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from enum import Enum


# =====================================================================
# Query intent model
# =====================================================================


class QueryIntent(
    str,
    Enum,
):
    """Supported deterministic query intents."""

    COUNT = "COUNT"

    LIST = "LIST"

    LOOKUP = "LOOKUP"

    EVIDENCE_AVAILABILITY = (
        "EVIDENCE_AVAILABILITY"
    )

    COMPARE = "COMPARE"

    ASSESSMENT = "ASSESSMENT"

    CLOSURE = "CLOSURE"

    GENERAL = "GENERAL"


QueryIntent.__test__ = False


@dataclass(
    frozen=True
)
class QueryIntentResult:
    """Deterministic query-intent classification result."""

    intent: QueryIntent

    confidence: str

    matched_rule: str

    normalized_query: str

    def __post_init__(
        self,
    ) -> None:
        """Validate intent result."""

        if self.confidence not in {
            "HIGH",
            "MEDIUM",
        }:

            raise ValueError(
                "Intent confidence must be "
                "HIGH or MEDIUM."
            )

        if not self.matched_rule.strip():

            raise ValueError(
                "matched_rule must not be empty."
            )

        if not self.normalized_query.strip():

            raise ValueError(
                "normalized_query must not be empty."
            )


# =====================================================================
# Query normalization
# =====================================================================


def normalize_query(
    query: str,
) -> str:
    """Normalize the user query for deterministic matching."""

    normalized = " ".join(
        query
        .strip()
        .lower()
        .split()
    )

    if not normalized:

        raise ValueError(
            "Query must not be empty."
        )

    return normalized


# =====================================================================
# Intent patterns
# =====================================================================


COUNT_PATTERNS = (
    re.compile(
        r"\bhow many\b"
    ),

    re.compile(
        r"\bcount\b"
    ),

    re.compile(
        r"\bnumber of\b"
    ),

    re.compile(
        r"\btotal (?:number of )?"
    ),
)


LIST_PATTERNS = (
    re.compile(
        r"^list\b"
    ),

    re.compile(
        r"^show (?:me )?(?:all )?"
    ),

    re.compile(
        r"^find all\b"
    ),

    re.compile(
        r"\bwhich observations\b"
    ),

    re.compile(
        r"\bwhich requirements\b"
    ),

    re.compile(
        r"\bwhich test cases\b"
    ),

    re.compile(
        r"\bwhich tests\b"
    ),
)


COMPARE_PATTERNS = (
    re.compile(
        r"\bcompare\b"
    ),

    re.compile(
        r"\bdifference between\b"
    ),

    re.compile(
        r"\bversus\b"
    ),

    re.compile(
        r"\bvs\.?\b"
    ),

    re.compile(
        r"\bexpected behavior.*observ"
    ),

    re.compile(
        r"\bobserv.*expected behavior"
    ),
)


CLOSURE_PATTERNS = (
    re.compile(
        r"\bclosure\b"
    ),

    re.compile(
        r"\bclosed\b"
    ),

    re.compile(
        r"\bclose\b"
    ),

    re.compile(
        r"\bready for closure\b"
    ),

    re.compile(
        r"\bcan .* be closed\b"
    ),
)


ASSESSMENT_PATTERNS = (
    re.compile(
        r"\bsupported\b"
    ),

    re.compile(
        r"\bnot supported\b"
    ),

    re.compile(
        r"\bcompliant\b"
    ),

    re.compile(
        r"\bcompliance\b"
    ),

    re.compile(
        r"\bcontradict"
    ),

    re.compile(
        r"\bdiscrepanc"
    ),

    re.compile(
        r"\bassessment\b"
    ),

    re.compile(
        r"\bvalid observation\b"
    ),
)


EVIDENCE_PATTERNS = (
    re.compile(
        r"\bmissing evidence\b"
    ),

    re.compile(
        r"\bverification evidence\b"
    ),

    re.compile(
        r"\btest evidence\b"
    ),

    re.compile(
        r"\bcode evidence\b"
    ),

    re.compile(
        r"\bimplementation evidence\b"
    ),

    re.compile(
        r"\bobservation evidence\b"
    ),

    re.compile(
        r"\bevidence available\b"
    ),

    re.compile(
        r"\bevidence availability\b"
    ),

    # Generalized missing-evidence pattern.
    #
    # Examples:
    #   missing test evidence
    #   missing code evidence
    #   missing implementation evidence
    #   missing observation evidence
    re.compile(
        r"\bmissing .* evidence\b"
    ),

    re.compile(
        r"\bwithout .* evidence\b"
    ),

    re.compile(
        r"\bdon't have .* evidence\b"
    ),

    re.compile(
        r"\bdo not have .* evidence\b"
    ),

    re.compile(
        r"\black(?:s|ing)? .* evidence\b"
    ),
)



LOOKUP_PATTERNS = (
    re.compile(
        r"\bwhat does\b"
    ),

    re.compile(
        r"\bwhat is\b"
    ),

    re.compile(
        r"\bdescribe\b"
    ),

    re.compile(
        r"\bexplain\b"
    ),

    re.compile(
        r"\bdetails of\b"
    ),

    re.compile(
        r"\btell me about\b"
    ),

    re.compile(
        r"\bfind\b"
    ),

    re.compile(
        r"\bretrieve\b"
    ),
)


EXPLICIT_ENGINEERING_ID_PATTERN = re.compile(
    r"\b(?:REQ|OBS|TC)"
    r"[\s_-]?"
    r"\d+\b",
    re.IGNORECASE,
)


# =====================================================================
# Pattern matching
# =====================================================================


def matches_any(
    query: str,
    patterns: tuple[
        re.Pattern[str],
        ...
    ],
) -> bool:
    """Return whether any supplied pattern matches the query."""

    return any(
        pattern.search(
            query
        )
        is not None

        for pattern
        in patterns
    )


# =====================================================================
# Deterministic intent router
# =====================================================================


def classify_query_intent(
    query: str,
) -> QueryIntentResult:
    """
    Classify the user's query deterministically.

    Routing priority:

        1. COUNT
        2. EVIDENCE_AVAILABILITY
        3. LIST
        4. CLOSURE
        5. COMPARE
        6. ASSESSMENT
        7. LOOKUP
        8. Explicit-ID lookup
        9. GENERAL

    Why precedence matters
    ----------------------

    "How many observations are closed?"

        COUNT

    not:

        CLOSURE

    because the user is asking for an inventory aggregate.

    Likewise:

    "List all closed observations"

        LIST

    while:

    "Can OBS-070 be closed?"

        CLOSURE

    Evidence availability also precedes LIST:

    "Which requirements have missing test evidence?"

        EVIDENCE_AVAILABILITY

    rather than a generic list query.
    """

    normalized = normalize_query(
        query
    )

    # ---------------------------------------------------------
    # QI-001 - COUNT
    #
    # Highest priority because count questions may contain
    # closure-status terms such as "closed".
    # ---------------------------------------------------------

    if matches_any(
        normalized,
        COUNT_PATTERNS,
    ):

        return QueryIntentResult(
            intent=QueryIntent.COUNT,

            confidence="HIGH",

            matched_rule=(
                "QI-001_COUNT"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-005 - EVIDENCE AVAILABILITY
    #
    # Must precede LIST because queries such as:
    #
    # "Which requirements have missing test evidence?"
    #
    # are evidence-availability questions even though they
    # contain list-style wording.
    # ---------------------------------------------------------

    if matches_any(
        normalized,
        EVIDENCE_PATTERNS,
    ):

        return QueryIntentResult(
            intent=(
                QueryIntent
                .EVIDENCE_AVAILABILITY
            ),

            confidence="HIGH",

            matched_rule=(
                "QI-005_EVIDENCE_AVAILABILITY"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-006 - LIST
    #
    # Must precede CLOSURE so:
    #
    # "List all closed observations"
    #
    # remains an inventory-list request.
    # ---------------------------------------------------------

    if matches_any(
        normalized,
        LIST_PATTERNS,
    ):

        return QueryIntentResult(
            intent=QueryIntent.LIST,

            confidence="HIGH",

            matched_rule=(
                "QI-006_LIST"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-002 - CLOSURE
    #
    # Applies after aggregate/list inventory queries have been
    # removed.
    # ---------------------------------------------------------

    if matches_any(
        normalized,
        CLOSURE_PATTERNS,
    ):

        return QueryIntentResult(
            intent=QueryIntent.CLOSURE,

            confidence="HIGH",

            matched_rule=(
                "QI-002_CLOSURE"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-003 - COMPARE
    # ---------------------------------------------------------

    if matches_any(
        normalized,
        COMPARE_PATTERNS,
    ):

        return QueryIntentResult(
            intent=QueryIntent.COMPARE,

            confidence="HIGH",

            matched_rule=(
                "QI-003_COMPARE"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-004 - ASSESSMENT
    # ---------------------------------------------------------

    if matches_any(
        normalized,
        ASSESSMENT_PATTERNS,
    ):

        return QueryIntentResult(
            intent=QueryIntent.ASSESSMENT,

            confidence="HIGH",

            matched_rule=(
                "QI-004_ASSESSMENT"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-007 - LOOKUP
    # ---------------------------------------------------------

    if matches_any(
        normalized,
        LOOKUP_PATTERNS,
    ):

        return QueryIntentResult(
            intent=QueryIntent.LOOKUP,

            confidence="HIGH",

            matched_rule=(
                "QI-007_LOOKUP"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-008 - EXPLICIT ENGINEERING-ID LOOKUP
    #
    # A bare identifier such as:
    #
    # REQ-900
    #
    # defaults to lookup rather than assessment.
    #
    # The production router may later promote the query when
    # stronger intent wording is present.
    # ---------------------------------------------------------

    if EXPLICIT_ENGINEERING_ID_PATTERN.search(
        normalized
    ):

        return QueryIntentResult(
            intent=QueryIntent.LOOKUP,

            confidence="MEDIUM",

            matched_rule=(
                "QI-008_EXPLICIT_ID_LOOKUP"
            ),

            normalized_query=(
                normalized
            ),
        )

    # ---------------------------------------------------------
    # QI-009 - GENERAL
    # ---------------------------------------------------------

    return QueryIntentResult(
        intent=QueryIntent.GENERAL,

        confidence="MEDIUM",

        matched_rule=(
            "QI-009_GENERAL"
        ),

        normalized_query=(
            normalized
        ),
    )
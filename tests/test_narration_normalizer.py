"""
Tests for deterministic narration output normalization.
"""

from __future__ import annotations

from src.narration_normalizer import (
    normalize_narration_output,
)


def test_clean_narration_is_preserved() -> None:

    raw = """## Assessment
Assessment: SUPPORTED
Confidence: HIGH
Closure Status: NOT ASSESSED

## Engineering Basis
The deterministic evidence chain supports the observation.

## Evidence Summary
Executed test evidence supports the assessment. [CIT-004]

## Blocking or Missing Evidence
None recorded.

## Closure Status
Closure readiness was not requested.

## Sources
[CIT-001] [CIT-002] [CIT-003] [CIT-004]

## Authority Limitation
REQTRACE AI provides engineering decision support.
"""

    normalized = normalize_narration_output(
        raw
    )

    assert (
        "Assessment: SUPPORTED"
        in normalized
    )

    assert (
        "## Authority Limitation"
        in normalized
    )


def test_duplicate_sections_are_removed() -> None:

    raw = """## Assessment
Assessment: SUPPORTED
Confidence: HIGH
Closure Status: NOT ASSESSED

## Engineering Basis
First basis.

## Evidence Summary
Evidence. [CIT-001]

## Blocking or Missing Evidence
None recorded.

## Closure Status
Not assessed.

## Sources
[CIT-001]

## Authority Limitation
Authority text.

## Blocking or Missing Evidence
Duplicate blocker.

## Closure Status
Duplicate closure.
"""

    normalized = normalize_narration_output(
        raw
    )

    assert (
        normalized.count(
            "## Blocking or Missing Evidence"
        )
        == 1
    )

    assert (
        normalized.count(
            "## Closure Status"
        )
        == 1
    )

    assert (
        "Duplicate blocker"
        not in normalized
    )


def test_sources_keep_markers_only() -> None:

    raw = """## Assessment
Assessment: SUPPORTED
Confidence: HIGH
Closure Status: NOT ASSESSED

## Engineering Basis
Basis.

## Evidence Summary
Summary.

## Blocking or Missing Evidence
None recorded.

## Closure Status
Not assessed.

## Sources
[CIT-001] [CIT-002]
[CIT-001] | OBSERVATION | some file
[CIT-002] | CODE | some source lines

## Authority Limitation
Authority.
"""

    normalized = normalize_narration_output(
        raw
    )

    assert (
        "## Sources\n"
        "[CIT-001] [CIT-002]"
        in normalized
    )

    assert (
        "| OBSERVATION |"
        not in normalized
    )


def test_control_line_is_removed() -> None:

    raw = """## Assessment
Assessment: SUPPORTED
Confidence: HIGH
Closure Status: NOT ASSESSED

## Engineering Basis
Basis.

## Evidence Summary
Summary.

## Blocking or Missing Evidence
None recorded.

## Closure Status
Not assessed.

## Sources
[CIT-001]

## Authority Limitation
Authority text.
Distinguish expected behavior from observed behavior.
"""

    normalized = normalize_narration_output(
        raw
    )

    assert (
        "Distinguish expected behavior"
        not in normalized
    )


def test_missing_section_is_not_fabricated() -> None:
    """
    Critical safety test.

    Normalization must not manufacture Authority Limitation just to
    satisfy NV-015.
    """

    raw = """## Assessment
Assessment: SUPPORTED
Confidence: HIGH
Closure Status: NOT ASSESSED

## Engineering Basis
Basis.

## Evidence Summary
Summary.

## Blocking or Missing Evidence
None.

## Closure Status
Not assessed.

## Sources
[CIT-001]
"""

    normalized = normalize_narration_output(
        raw
    )

    assert (
        "## Authority Limitation"
        not in normalized
    )


def test_missing_mandatory_values_are_not_created() -> None:

    raw = """## Assessment

## Engineering Basis
Basis.

## Evidence Summary
Summary.

## Blocking or Missing Evidence
None.

## Closure Status
Not assessed.

## Sources
[CIT-001]

## Authority Limitation
Authority.
"""

    normalized = normalize_narration_output(
        raw
    )

    assert (
        "Assessment: SUPPORTED"
        not in normalized
    )

    assert (
        "Confidence: HIGH"
        not in normalized
    )
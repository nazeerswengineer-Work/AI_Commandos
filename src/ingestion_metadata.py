"""
Engineering metadata extraction for dynamic ingestion.

Only explicit values found in source content or filenames are returned.

No requirement, observation, test-case, result, function, section, or
document category is fabricated.
"""

from __future__ import annotations

import re

from pathlib import Path


ENGINEERING_ID_PATTERN = re.compile(
    r"\b(?P<prefix>REQ|OBS|TC)"
    r"[\s_-]?"
    r"(?P<number>\d+)\b",
    re.IGNORECASE,
)

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*$"
)

TEXT_HEADING_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9 /&()_-]{3,80}$"
)

RESULT_STATUS_PATTERN = re.compile(
    r"\b(?:result(?:\s+status)?\s*[:=-]?\s*)"
    r"(?P<status>"
    r"NOT\s+EXECUTED|"
    r"NOT\s+RUN|"
    r"BLOCKED|"
    r"PASSED?|"
    r"FAILED?"
    r")\b",
    re.IGNORECASE,
)


def ordered_unique(
    values: list[str],
) -> tuple[str, ...]:
    """Return ordered, duplicate-free values."""

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
    prefix: str,
    number: str,
) -> str:
    """Normalize an explicit engineering identifier."""

    return (
        f"{prefix.upper()}-{number}"
    )


def extract_engineering_ids(
    content: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Extract REQ, OBS and TC identifiers from text."""

    requirements: list[str] = []
    observations: list[str] = []
    test_cases: list[str] = []

    for match in ENGINEERING_ID_PATTERN.finditer(
        content
    ):
        prefix = (
            match.group(
                "prefix"
            )
            .upper()
        )

        value = normalize_engineering_id(
            prefix,
            match.group(
                "number"
            ),
        )

        if prefix == "REQ":
            requirements.append(
                value
            )

        elif prefix == "OBS":
            observations.append(
                value
            )

        elif prefix == "TC":
            test_cases.append(
                value
            )

    return (
        ordered_unique(
            requirements
        ),
        ordered_unique(
            observations
        ),
        ordered_unique(
            test_cases
        ),
    )


def extract_result_status(
    content: str,
) -> str | None:
    """Extract an explicit source-reported result status."""

    match = RESULT_STATUS_PATTERN.search(
        content
    )

    if match is None:
        return None

    raw_status = " ".join(
        match.group(
            "status"
        )
        .upper()
        .split()
    )

    normalized = {
        "PASS": "PASS",
        "PASSED": "PASS",
        "FAIL": "FAIL",
        "FAILED": "FAIL",
        "NOT RUN": "NOT EXECUTED",
        "NOT EXECUTED": "NOT EXECUTED",
        "BLOCKED": "BLOCKED",
    }

    return normalized.get(
        raw_status
    )


def detect_section_heading(
    line: str,
    *,
    markdown: bool,
) -> str | None:
    """Identify an explicit source section heading."""

    stripped = line.strip()

    if not stripped:
        return None

    if markdown:
        match = MARKDOWN_HEADING_PATTERN.match(
            line
        )

        if match is not None:
            return (
                match.group(
                    "title"
                )
                .strip()
            )

    if (
        len(stripped) <= 80
        and TEXT_HEADING_PATTERN.fullmatch(
            stripped
        )
    ):
        return stripped

    return None


def infer_document_type(
    *,
    relative_path: str,
    extension: str,
    content_sample: str = "",
) -> tuple[str, str]:
    """
    Conservatively classify an engineering source document.

    Precedence:

    1. Source-code extension
    2. Explicit directory / filename category
    3. Strong filename category hints
    4. Engineering IDs in filename
    5. Engineering IDs in content

    Important:
    A FRD commonly references test cases.
    Therefore TC-501 appearing inside an FRD must not classify
    the complete document as a Test Report.
    """

    normalized_path = (
        relative_path
        .replace("\\", "/")
        .lower()
    )

    filename = (
        Path(normalized_path)
        .name
    )

    stem = (
        Path(filename)
        .stem
        .lower()
    )

    extension = (
        extension
        .strip()
        .lower()
    )

    # ---------------------------------------------------------
    # 1. Source code
    # ---------------------------------------------------------

    code_extensions = {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".py",
    }

    if extension in code_extensions:
        return (
            "code",
            "Source Code",
        )

    # ---------------------------------------------------------
    # 2. Explicit path/category information
    # ---------------------------------------------------------

    path_parts = {
        part
        for part in normalized_path.split("/")
        if part
    }

    if (
        "observations" in path_parts
        or "observation" in path_parts
    ):
        return (
            "observation",
            "Observation Report",
        )

    if (
        "tests" in path_parts
        or "test" in path_parts
    ):
        return (
            "test",
            "Test Report",
        )

    if (
        "reviews" in path_parts
        or "review" in path_parts
    ):
        return (
            "review",
            "Engineering Review",
        )

    if (
        "frd" in path_parts
        or "requirements" in path_parts
        or "requirement" in path_parts
    ):
        return (
            "frd",
            "FRD",
        )

    # ---------------------------------------------------------
    # 3. Strong filename hints
    #
    # FRD.md must be classified as an FRD even when its content
    # refers to TC-501.
    # ---------------------------------------------------------

    observation_name_patterns = (
        "observation_report",
        "observation-report",
        "observation report",
        "obs_report",
        "obs-report",
    )

    if any(
        pattern in stem
        for pattern in observation_name_patterns
    ):
        return (
            "observation",
            "Observation Report",
        )

    test_name_patterns = (
        "test_report",
        "test-report",
        "test report",
        "verification_report",
        "verification-report",
    )

    if any(
        pattern in stem
        for pattern in test_name_patterns
    ):
        return (
            "test",
            "Test Report",
        )

    review_name_patterns = (
        "review_report",
        "review-report",
        "review report",
        "engineering_review",
        "engineering-review",
    )

    if any(
        pattern in stem
        for pattern in review_name_patterns
    ):
        return (
            "review",
            "Engineering Review",
        )

    frd_name_patterns = (
        "frd",
        "functional_requirement",
        "functional-requirement",
        "functional requirement",
        "requirements_specification",
        "requirement_specification",
    )

    if any(
        pattern in stem
        for pattern in frd_name_patterns
    ):
        return (
            "frd",
            "FRD",
        )

    # ---------------------------------------------------------
    # 4. IDs explicitly contained in filename
    #
    # Filename identity is stronger evidence of document category
    # than references appearing inside document content.
    # ---------------------------------------------------------

    (
        filename_requirements,
        filename_observations,
        filename_tests,
    ) = extract_engineering_ids(
        filename
    )

    if filename_observations:
        return (
            "observation",
            "Observation Report",
        )

    if filename_tests:
        return (
            "test",
            "Test Report",
        )

    if filename_requirements:
        return (
            "frd",
            "FRD",
        )

    # ---------------------------------------------------------
    # 5. Content fallback
    #
    # Content can contain cross-references. Therefore classify only
    # when the evidence is reasonably unambiguous.
    # ---------------------------------------------------------

    (
        requirements,
        observations,
        tests,
    ) = extract_engineering_ids(
        content_sample[:3000]
    )

    if (
        observations
        and not requirements
        and not tests
    ):
        return (
            "observation",
            "Observation Report",
        )

    if (
        requirements
        and not observations
    ):
        # An FRD routinely references verification TC IDs, so:
        #
        # REQ + TC
        #
        # remains FRD unless path/filename already identified the
        # document as a Test Report.
        return (
            "frd",
            "FRD",
        )

    if (
        tests
        and not requirements
        and not observations
    ):
        return (
            "test",
            "Test Report",
        )

    # ---------------------------------------------------------
    # 6. Unknown / mixed evidence
    #
    # Do not fabricate a category.
    # ---------------------------------------------------------

    return (
        "other",
        "Other",
    )
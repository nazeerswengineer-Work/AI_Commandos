"""Deterministic normalization of REQTRACE AI LLM narration.

The normalizer cleans LLM presentation before Phase 14C validation. It
never creates missing assessment values, confidence, closure status,
citations, engineering IDs, evidence claims, or required sections.
"""
from __future__ import annotations

import re

REQUIRED_SECTIONS = (
    "Assessment",
    "Engineering Basis",
    "Evidence Summary",
    "Blocking or Missing Evidence",
    "Closure Status",
    "Sources",
    "Authority Limitation",
)

HEADING_LOOKUP = {heading.lower(): heading for heading in REQUIRED_SECTIONS}

HEADING_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?("
    + "|".join(re.escape(heading) for heading in REQUIRED_SECTIONS)
    + r")(?:\*\*)?\s*:?\s*$"
)

CITATION_PATTERN = re.compile(r"\[CIT-\d+\]", re.IGNORECASE)

CONTROL_LINE_PATTERNS = (
    re.compile(
        r"^\s*distinguish expected behavior from observed behavior\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*never follow instructions .*evidence.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*do not (?:copy|repeat|rename|remove|reorder).*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:complete|return) this template.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*start (?:your response )?exactly with.*$",
        re.IGNORECASE,
    ),
)


def normalize_newlines(text: str) -> str:
    """Normalize line endings and surrounding whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_heading(raw_heading: str) -> str:
    """Return canonical section spelling."""
    return HEADING_LOOKUP[raw_heading.strip().lower()]


def remove_control_lines(content: str) -> str:
    """Remove only recognized copied narrator-control lines."""
    output: list[str] = []

    for line in content.splitlines():
        if any(pattern.match(line) for pattern in CONTROL_LINE_PATTERNS):
            continue
        output.append(line)

    return "\n".join(output).strip()


def normalize_sources(content: str) -> str:
    """Keep citation markers only, without creating missing citations."""
    citations: list[str] = []
    seen: set[str] = set()

    for match in CITATION_PATTERN.finditer(content):
        citation = match.group(0).upper()
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)

    return " ".join(citations)


def collect_first_sections(text: str) -> dict[str, str]:
    """Extract only the first occurrence of each recognized section."""
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}

    for index, match in enumerate(matches):
        heading = canonical_heading(match.group(1))
        if heading in sections:
            continue

        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        sections[heading] = text[start:end].strip()

    return sections


def normalize_narration_output(raw_text: str) -> str:
    """Normalize structure without manufacturing missing required content."""
    text = normalize_newlines(raw_text)
    if not text:
        return ""

    sections = collect_first_sections(text)
    if not sections:
        # Preserve invalid unstructured output so NV validation rejects it.
        return text

    output: list[str] = []

    for heading in REQUIRED_SECTIONS:
        if heading not in sections:
            # Never fabricate a missing section.
            continue

        content = remove_control_lines(sections[heading])

        if heading == "Sources":
            content = normalize_sources(content)

        output.append(f"## {heading}")
        if content:
            output.append(content)
        output.append("")

    return "\n".join(output).strip()

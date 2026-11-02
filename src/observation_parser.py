"""
Observation PDF parsing for REQTRACE AI.

This module converts observation PDFs into validated observation-level
EvidenceRecord objects.

Important design rules:

- explicit source metadata is preserved
- engineering IDs are normalized
- page numbers remain one-based
- referenced requirement IDs are not corrected automatically
- missing fields remain None
- assessment and closure decisions are not performed here
- no LLM, embeddings, retrieval, or ChromaDB calls occur here
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.evidence import (
    EvidenceRecord,
    build_observation_evidence_id,
    calculate_content_hash,
)
from src.loaders import LoadedPDF, load_pdf
from src.metadata import (
    DocumentType,
    EvidenceMetadata,
    ParsingMethod,
)
from src.normalization import (
    normalize_observation_id,
    normalize_optional_requirement_id,
    normalize_optional_test_case_id,
    normalize_revision,
    normalize_whitespace,
)


class ObservationParsingError(RuntimeError):
    """Raised when an observation document cannot be parsed reliably."""


FIELD_PATTERN_TEMPLATE = r"(?im)^{field}\s*:\s*(.+?)\s*$"

PAGE_NUMBER_PATTERN = re.compile(
    r"(?im)^Page\s+\d+\s*$"
)

SYNTHETIC_HEADER_LINES = {
    "REQTRACE AI SYNTHETIC EVIDENCE",
    "Fictional demonstration evidence only",
}

SAFETY_NOTICE_PREFIX = (
    "FICTIONAL TRAINING DATA - NOT FOR ENGINEERING APPROVAL"
)


@dataclass(frozen=True)
class ObservationParseWarning:
    """Non-fatal observation parsing issue."""

    code: str
    message: str
    document_name: str
    page_number: int | None = None


@dataclass(frozen=True)
class ParsedObservationFields:
    """Fields explicitly extracted from an observation PDF."""

    observation_id: str
    revision: str | None
    status: str | None
    severity: str | None

    requirement_id: str | None
    test_case_id: str | None

    referenced_file: str | None
    referenced_function: str | None

    description: str | None
    expected_result: str | None
    actual_result: str | None

    evidence_limitation: str | None
    revision_conflict: str | None
    assessment_constraint: str | None
    closure_limitation: str | None


@dataclass(frozen=True)
class ObservationParseResult:
    """Complete result for one observation document."""

    document_name: str
    document_revision: str | None
    observation: ParsedObservationFields
    evidence_records: tuple[EvidenceRecord, ...]
    warnings: tuple[ObservationParseWarning, ...]

    @property
    def observation_id(self) -> str:
        return self.observation.observation_id

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_records)


def remove_observation_page_noise(text: str) -> str:
    """Remove deterministic Phase 3 headers and footers."""

    retained_lines: list[str] = []
    skip_safety_notice = False

    for line in text.splitlines():
        cleaned = normalize_whitespace(line)

        if not cleaned:
            if retained_lines and retained_lines[-1] != "":
                retained_lines.append("")
            continue

        if cleaned in SYNTHETIC_HEADER_LINES:
            continue

        if PAGE_NUMBER_PATTERN.fullmatch(cleaned):
            continue

        if cleaned.startswith(SAFETY_NOTICE_PREFIX):
            skip_safety_notice = True
            continue

        if skip_safety_notice:
            upper = cleaned.upper()

            if (
                "OBSERVATION CLOSURE" in upper
                or "PRODUCTION RELEASE" in upper
            ):
                continue

            skip_safety_notice = False

        retained_lines.append(line.rstrip())

    while retained_lines and not retained_lines[0].strip():
        retained_lines.pop(0)

    while retained_lines and not retained_lines[-1].strip():
        retained_lines.pop()

    return "\n".join(retained_lines).strip()


def extract_field(
    text: str,
    field_name: str,
) -> str | None:
    """
    Extract a metadata field while tolerating PDF line wrapping.

    Supports both:

        Referenced Requirement: REQ-017

    and:

        Referenced Requirement:
        REQ-017
    """

    pattern = re.compile(
        rf"(?im)^\s*{re.escape(field_name)}\s*:\s*"
        rf"(?:\n\s*)?"
        rf"([^\n]+)"
    )

    match = pattern.search(text)

    if match is None:
        return None

    value = normalize_whitespace(
        match.group(1)
    )

    return value or None


def extract_section(
    text: str,
    section_title: str,
) -> str | None:
    """
    Extract a numbered observation section.

    Example:
        1. Observation Description
        ...
        2. Expected Result
    """

    title = re.escape(section_title)

    heading_pattern = re.compile(
        rf"(?im)^\d+(?:\.\d+)*[.)]?\s+{title}\s*$"
    )

    match = heading_pattern.search(text)

    if match is None:
        return None

    remaining_text = text[match.end():]

    next_heading = re.search(
        r"(?im)^\d+(?:\.\d+)*[.)]?\s+[A-Za-z].*$",
        remaining_text,
    )

    if next_heading is not None:
        section_text = remaining_text[:next_heading.start()]
    else:
        section_text = remaining_text

    section_text = section_text.strip()

    return section_text or None


def extract_first_available_section(
    text: str,
    section_names: tuple[str, ...],
) -> str | None:
    """Return the first existing section from alternative section titles."""

    for section_name in section_names:
        extracted = extract_section(
            text,
            section_name,
        )

        if extracted:
            return extracted

    return None


def extract_observation_revision(
    text: str,
    document_name: str,
) -> str | None:
    """Extract observation revision from text or filename."""

    raw_revision = extract_field(
        text,
        "Revision",
    )

    if raw_revision:
        try:
            return normalize_revision(
                raw_revision
            )
        except ValueError:
            pass

    filename_match = re.search(
        r"(?i)rev(?:ision)?[_\-\s]*(\d+)",
        document_name,
    )

    if filename_match:
        return normalize_revision(
            filename_match.group(1)
        )

    return None


def parse_observation_fields(
    text: str,
    document_name: str,
) -> ParsedObservationFields:
    """Extract explicitly stated observation fields."""

    raw_observation_id = extract_field(
        text,
        "Observation ID",
    )

    if raw_observation_id is None:
        filename_match = re.search(
            r"(?i)(OBS[\s_-]*\d+)",
            document_name,
        )

        if filename_match is None:
            raise ObservationParsingError(
                f"Observation ID not found in '{document_name}'."
            )

        raw_observation_id = filename_match.group(1)

    observation_id = normalize_observation_id(
        raw_observation_id
    )

    requirement_id = normalize_optional_requirement_id(
        extract_field(
            text,
            "Referenced Requirement",
        )
    )

    raw_test_case = extract_field(
        text,
        "Referenced Test",
    )

    if raw_test_case is not None:
        if raw_test_case.lower() in {
            "not provided",
            "none",
            "n/a",
            "na",
        }:
            raw_test_case = None

    test_case_id = normalize_optional_test_case_id(
        raw_test_case
    )

    referenced_file = extract_field(
        text,
        "Referenced File",
    )

    referenced_function = extract_field(
        text,
        "Referenced Function",
    )

    return ParsedObservationFields(
        observation_id=observation_id,
        revision=extract_observation_revision(
            text,
            document_name,
        ),
        status=extract_field(
            text,
            "Status",
        ),
        severity=extract_field(
            text,
            "Severity",
        ),
        requirement_id=requirement_id,
        test_case_id=test_case_id,
        referenced_file=referenced_file,
        referenced_function=referenced_function,
        description=extract_first_available_section(
            text,
            (
                "Observation Description",
                "Description",
            ),
        ),
        expected_result=extract_first_available_section(
            text,
            (
                "Expected Result",
                "Expected Behavior",
                "Expected Behaviour",
            ),
        ),
        actual_result=extract_first_available_section(
            text,
            (
                "Actual Result",
                "Reported Actual Result",
            ),
        ),
        evidence_limitation=extract_first_available_section(
            text,
            (
                "Evidence Limitation",
                "Evidence Limitations",
            ),
        ),
        revision_conflict=extract_first_available_section(
    text,
    (
        "Revision Conflict",
        "Revision Conflicts",
    ),
),

assessment_constraint=extract_first_available_section(
    text,
    (
        "Assessment Constraint",
        "Assessment Constraints",
        "Traceability Concern",
    ),
),
        closure_limitation=extract_first_available_section(
            text,
            (
                "Closure Limitation",
                "Closure Limitations",
            ),
        ),
    )


def join_document_pages(
    loaded_pdf: LoadedPDF,
) -> str:
    """
    Join cleaned observation pages while retaining a deterministic page marker.

    The marker is internal parser context and is not used as the citation
    page number.
    """

    page_texts: list[str] = []

    for page in loaded_pdf.pages:
        if page.is_empty:
            continue

        cleaned = remove_observation_page_noise(
            page.text
        )

        if cleaned:
            page_texts.append(cleaned)

    return "\n\n".join(page_texts)


def choose_primary_page(
    loaded_pdf: LoadedPDF,
    observation_id: str,
) -> int:
    """
    Identify the page containing the explicit Observation ID.

    The first page is used only when no explicit page-level ID match exists.
    """

    normalized_target = observation_id.upper()

    for page in loaded_pdf.pages:
        cleaned = remove_observation_page_noise(
            page.text
        )

        raw_id = extract_field(
            cleaned,
            "Observation ID",
        )

        if raw_id is None:
            continue

        try:
            current_id = normalize_observation_id(
                raw_id
            )
        except ValueError:
            continue

        if current_id.upper() == normalized_target:
            return page.page_number

    non_empty_pages = loaded_pdf.non_empty_pages

    if not non_empty_pages:
        raise ObservationParsingError(
            f"Observation PDF '{loaded_pdf.document_name}' "
            "contains no searchable text."
        )

    return non_empty_pages[0].page_number


def make_observation_chunk_id(
    *,
    document_name: str,
    page_number: int,
    observation_id: str,
) -> str:
    """Create a stable observation chunk ID."""

    safe_stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        Path(document_name).stem,
    ).strip("-")

    return (
        f"OBS-{safe_stem}"
        f"-PAGE-{page_number}"
        f"-{observation_id}"
    )


def create_observation_evidence(
    *,
    loaded_pdf: LoadedPDF,
    fields: ParsedObservationFields,
    content: str,
) -> EvidenceRecord:
    """Create one validated observation evidence record."""

    page_number = choose_primary_page(
        loaded_pdf,
        fields.observation_id,
    )

    evidence_id = build_observation_evidence_id(
        document_name=loaded_pdf.document_name,
        page_number=page_number,
        observation_id=fields.observation_id,
    )

    chunk_id = make_observation_chunk_id(
        document_name=loaded_pdf.document_name,
        page_number=page_number,
        observation_id=fields.observation_id,
    )

    content_hash = calculate_content_hash(
        content
    )

    metadata = EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.OBSERVATION,
        document_name=loaded_pdf.document_name,
        document_revision=fields.revision,
        page_number=page_number,
        requirement_id=fields.requirement_id,
        observation_id=fields.observation_id,
        test_case_id=fields.test_case_id,
        result_status=fields.status,
        file_name=fields.referenced_file,
        function_name=fields.referenced_function,
        file_path=str(loaded_pdf.file_path),
        chunk_id=chunk_id,
        content_hash=content_hash,
        parsing_method=ParsingMethod.OBSERVATION,
    )

    return EvidenceRecord(
        content=content,
        metadata=metadata,
    )


def parse_observation_document(
    loaded_pdf: LoadedPDF,
) -> ObservationParseResult:
    """Parse one loaded observation PDF."""

    warnings: list[ObservationParseWarning] = []

    for page in loaded_pdf.pages:
        if page.is_empty:
            warnings.append(
                ObservationParseWarning(
                    code="EMPTY_PAGE",
                    message=(
                        "Page contains no searchable text. "
                        "OCR is not performed in Phase 6."
                    ),
                    document_name=loaded_pdf.document_name,
                    page_number=page.page_number,
                )
            )

    content = join_document_pages(
        loaded_pdf
    )

    if not content:
        raise ObservationParsingError(
            f"Observation PDF '{loaded_pdf.document_name}' "
            "contains no searchable text."
        )

    fields = parse_observation_fields(
        content,
        loaded_pdf.document_name,
    )

    if fields.requirement_id is None:
        warnings.append(
            ObservationParseWarning(
                code="MISSING_REQUIREMENT_REFERENCE",
                message=(
                    "No explicit referenced requirement was found."
                ),
                document_name=loaded_pdf.document_name,
            )
        )

    if fields.test_case_id is None:
        warnings.append(
            ObservationParseWarning(
                code="MISSING_TEST_REFERENCE",
                message=(
                    "No explicit referenced test case was found."
                ),
                document_name=loaded_pdf.document_name,
            )
        )

    if fields.status is None:
        warnings.append(
            ObservationParseWarning(
                code="MISSING_STATUS",
                message=(
                    "Observation status was not explicitly found."
                ),
                document_name=loaded_pdf.document_name,
            )
        )

    evidence = create_observation_evidence(
        loaded_pdf=loaded_pdf,
        fields=fields,
        content=content,
    )

    return ObservationParseResult(
        document_name=loaded_pdf.document_name,
        document_revision=fields.revision,
        observation=fields,
        evidence_records=(evidence,),
        warnings=tuple(warnings),
    )


def parse_observation_pdf(
    path: str | Path,
) -> ObservationParseResult:
    """Load and parse one observation PDF."""

    loaded_pdf = load_pdf(
        path,
        sort_text=True,
    )

    return parse_observation_document(
        loaded_pdf
    )


def parse_observation_directory(
    directory: str | Path,
) -> tuple[ObservationParseResult, ...]:
    """Parse every observation PDF in deterministic filename order."""

    directory_path = (
        Path(directory)
        .expanduser()
        .resolve()
    )

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Observation directory does not exist: "
            f"{directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Observation path is not a directory: "
            f"{directory_path}"
        )

    pdf_paths = sorted(
        (
            path
            for path in directory_path.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".pdf"
        ),
        key=lambda item: item.name.lower(),
    )

    return tuple(
        parse_observation_pdf(path)
        for path in pdf_paths
    )
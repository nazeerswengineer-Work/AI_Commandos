"""
Requirement-aware FRD parsing for REQTRACE AI.

This module converts loaded FRD PDF pages into validated requirement-level
EvidenceRecord objects.

Rules:

- preserve the original document name
- preserve one-based page numbers
- normalize requirement IDs
- preserve document revision when available
- do not merge different requirement IDs
- do not fabricate missing metadata
- generate stable evidence IDs and content hashes
- keep empty-page and parsing warnings visible
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.evidence import (
    EvidenceRecord,
    build_frd_evidence_id,
    calculate_content_hash,
)
from src.loaders import LoadedPDF, LoadedPDFPage, load_pdf
from src.metadata import (
    DocumentType,
    EvidenceMetadata,
    ParsingMethod,
)
from src.normalization import (
    extract_requirement_ids,
    normalize_multiline_content,
    normalize_requirement_id,
    normalize_revision,
    normalize_whitespace,
)


class FRDParsingError(RuntimeError):
    """Raised when an FRD cannot be parsed reliably."""


REQUIREMENT_HEADING_PATTERN = re.compile(
r"(?im)^\s*"
r"\d+(?:\.\d+)*"
r"[.)]?\s+Requirement\s+"
r"(REQ[\s_-]*\d+)"
r"\s*$"
)

REVISION_LINE_PATTERN = re.compile(
    r"(?im)^Revision\s*:\s*"
    r"((?:Rev(?:ision)?[\\_ -]*)?\d+)\s*$"
)

FILENAME_REVISION_PATTERN = re.compile(
r"(?i)rev(?:ision)?[_\-\s]*(\d+)"
)

SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^(\d+(?:\.\d+)*)[.)]?\s+(.+?)\s*$"
)

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
class RequirementOccurrence:
    """One requirement heading detected on one PDF page."""

    requirement_id: str
    start_offset: int
    end_offset: int
    heading_text: str


@dataclass(frozen=True)
class FRDParseWarning:
    """Non-fatal issue discovered during FRD parsing."""

    code: str
    message: str
    document_name: str
    page_number: int | None = None


@dataclass(frozen=True)
class FRDParseResult:
    """Complete output of one FRD parsing operation."""

    document_name: str
    document_revision: str | None
    page_count: int
    evidence_records: tuple[EvidenceRecord, ...]
    warnings: tuple[FRDParseWarning, ...]

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        """Return ordered unique requirement IDs."""

        result: list[str] = []
        seen: set[str] = set()

        for evidence in self.evidence_records:
            requirement_id = evidence.metadata.requirement_id

            if requirement_id and requirement_id not in seen:
                seen.add(requirement_id)
                result.append(requirement_id)

        return tuple(result)

    @property
    def requirement_count(self) -> int:
        """Return number of generated requirement evidence records."""

        return len(self.evidence_records)


def remove_synthetic_page_noise(text: str) -> str:
    """
    Remove known Phase 3 synthetic headers and footers.

    This is deterministic cleanup for the demonstration dataset. It does not
    remove arbitrary engineering text based on semantic judgment.
    """

    retained_lines: list[str] = []
    skip_safety_notice = False

    for line in text.splitlines():
        cleaned_line = normalize_whitespace(line)

        if not cleaned_line:
            if retained_lines and retained_lines[-1] != "":
                retained_lines.append("")
            continue

        if cleaned_line in SYNTHETIC_HEADER_LINES:
            continue
        if PAGE_NUMBER_PATTERN.fullmatch(cleaned_line):
            continue

        if cleaned_line.startswith(SAFETY_NOTICE_PREFIX):
            skip_safety_notice = True
            continue

        if skip_safety_notice:
            upper_line = cleaned_line.upper()

            if (
                "OBSERVATION CLOSURE" in upper_line
                or "PRODUCTION RELEASE" in upper_line
            ):
                continue

            skip_safety_notice = False

        retained_lines.append(line.rstrip())

    while retained_lines and not retained_lines[0].strip():
        retained_lines.pop(0)

    while retained_lines and not retained_lines[-1].strip():
        retained_lines.pop()

    return normalize_multiline_content(
        "\n".join(retained_lines)
    )


def extract_document_revision(
    loaded_pdf: LoadedPDF,
) -> str | None:

    for page in loaded_pdf.pages:
        match = REVISION_LINE_PATTERN.search(page.text)

        if match:
            return normalize_revision(match.group(1))

    filename_match = FILENAME_REVISION_PATTERN.search(
        loaded_pdf.document_name
    )

    if filename_match:
        return normalize_revision(filename_match.group(1))

    return None



def extract_section_for_offset(
    text: str,
    offset: int,
) -> str | None:
    """Return the nearest preceding numbered section identifier."""

    candidates = [
        match
        for match in SECTION_HEADING_PATTERN.finditer(text)
        if match.start() <= offset
    ]

    if not candidates:
        return None

    nearest = candidates[-1]
    section_number = nearest.group(1)
    section_title = normalize_whitespace(nearest.group(2))

    return f"{section_number} {section_title}".strip()


def find_requirement_occurrences(
    text: str,
) -> tuple[RequirementOccurrence, ...]:

    occurrences: list[RequirementOccurrence] = []
    seen_ids: set[str] = set()

    for match in REQUIREMENT_HEADING_PATTERN.finditer(text):

        requirement_id = normalize_requirement_id(
            match.group(1)
        )

        if requirement_id in seen_ids:
            continue

        seen_ids.add(requirement_id)

        occurrences.append(
            RequirementOccurrence(
                requirement_id=requirement_id,
                start_offset=match.start(),
                end_offset=match.end(),
                heading_text=normalize_whitespace(
                    match.group(0)
                ),
            )
        )

    return tuple(occurrences)


def make_chunk_id(
    document_name: str,
    page_number: int,
    requirement_id: str,
    occurrence_number: int,
) -> str:
    """Create a deterministic requirement chunk ID."""

    safe_document_stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        Path(document_name).stem,
    ).strip("-")

    return (
        f"FRD-{safe_document_stem}"
        f"-PAGE-{page_number}"
        f"-{requirement_id}"
        f"-{occurrence_number:02d}"
    )


def build_requirement_content(
    page_text: str,
    occurrence: RequirementOccurrence,
    next_occurrence: RequirementOccurrence | None,
) -> str:
    """
    Extract one requirement without including the next requirement.

    The explicit requirement heading is retained in the evidence content.
    """

    end_offset = (
        next_occurrence.start_offset
        if next_occurrence is not None
        else len(page_text)
    )

    requirement_text = page_text[
        occurrence.start_offset:end_offset
    ]

    normalized = normalize_multiline_content(requirement_text)

    if not normalized:
        raise FRDParsingError(
            f"Requirement {occurrence.requirement_id} has empty content."
        )

    return normalized


def create_requirement_evidence(
    loaded_pdf: LoadedPDF,
    page: LoadedPDFPage,
    revision: str | None,
    requirement_id: str,
    content: str,
    section: str | None,
    occurrence_number: int,
) -> EvidenceRecord:
    """Create one validated FRD evidence record."""

    evidence_id = build_frd_evidence_id(
        document_name=loaded_pdf.document_name,
        page_number=page.page_number,
        requirement_id=requirement_id,
    )

    chunk_id = make_chunk_id(
        document_name=loaded_pdf.document_name,
        page_number=page.page_number,
        requirement_id=requirement_id,
        occurrence_number=occurrence_number,
    )

    content_hash = calculate_content_hash(content)

    metadata = EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.FRD,
        document_name=loaded_pdf.document_name,
        document_revision=revision,
        page_number=page.page_number,
        section=section,
        requirement_id=requirement_id,
        file_name=loaded_pdf.document_name,
        file_path=str(loaded_pdf.file_path),
        chunk_id=chunk_id,
        content_hash=content_hash,
        parsing_method=ParsingMethod.REQUIREMENT,
    )

    return EvidenceRecord(
        content=content,
        metadata=metadata,
    )


def parse_frd_document(
    loaded_pdf: LoadedPDF,
) -> FRDParseResult:
    """
    Parse one loaded FRD into requirement-level evidence records.

    Different explicit requirement headings always produce separate evidence
    records. A requirement ID found only in descriptive text does not create
    an evidence record.
    """

    revision = extract_document_revision(loaded_pdf)
    evidence_records: list[EvidenceRecord] = []
    warnings: list[FRDParseWarning] = []
    evidence_ids: set[str] = set()

    for page in loaded_pdf.pages:
        if page.is_empty:
            warnings.append(
                FRDParseWarning(
                    code="EMPTY_PAGE",
                    message=(
                        "The page contains no searchable text. "
                        "OCR is not performed in Phase 5."
                    ),
                    document_name=loaded_pdf.document_name,
                    page_number=page.page_number,
                )
            )
            continue

        cleaned_page_text = remove_synthetic_page_noise(page.text)
        occurrences = find_requirement_occurrences(
            cleaned_page_text
        )

        if not occurrences:
            referenced_ids = extract_requirement_ids(
                cleaned_page_text
            )

            if referenced_ids:
                warnings.append(
                    FRDParseWarning(
                        code="REFERENCE_WITHOUT_REQUIREMENT_HEADING",
                        message=(
                            "Requirement IDs were found, but no explicit "
                            "requirement heading was detected on this page. "
                            "No requirement evidence record was created."
                        ),
                        document_name=loaded_pdf.document_name,
                        page_number=page.page_number,
                    )
                )

            continue

        per_page_counts: dict[str, int] = {}

        for index, occurrence in enumerate(occurrences):
            next_occurrence = (
                occurrences[index + 1]
                if index + 1 < len(occurrences)
                else None
            )

            content = build_requirement_content(
                page_text=cleaned_page_text,
                occurrence=occurrence,
                next_occurrence=next_occurrence,
            )

            extracted_ids = extract_requirement_ids(content)

            if occurrence.requirement_id not in extracted_ids:
                raise FRDParsingError(
                    "Requirement content does not contain its "
                    f"detected ID: {occurrence.requirement_id}"
                )

            per_page_counts[occurrence.requirement_id] = (
                per_page_counts.get(
                    occurrence.requirement_id,
                    0,
                )
                + 1
            )

            occurrence_number = per_page_counts[
                occurrence.requirement_id
            ]

            section = extract_section_for_offset(
                cleaned_page_text,
                occurrence.start_offset,
            )

            evidence = create_requirement_evidence(
                loaded_pdf=loaded_pdf,
                page=page,
                revision=revision,
                requirement_id=occurrence.requirement_id,
                content=content,
                section=section,
                occurrence_number=occurrence_number,
            )

            if evidence.metadata.evidence_id in evidence_ids:
                raise FRDParsingError(
                    "Duplicate FRD evidence ID detected: "
                    f"{evidence.metadata.evidence_id}. "
                    "Multiple instances of the same requirement on one "
                    "page require a more specific source distinction."
                )

            evidence_ids.add(evidence.metadata.evidence_id)
            evidence_records.append(evidence)

    if not evidence_records:
        warnings.append(
            FRDParseWarning(
                code="NO_REQUIREMENTS_PARSED",
                message=(
                    "No explicit FRD requirement sections were parsed "
                    "from the document."
                ),
                document_name=loaded_pdf.document_name,
                page_number=None,
            )
        )

    return FRDParseResult(
        document_name=loaded_pdf.document_name,
        document_revision=revision,
        page_count=loaded_pdf.page_count,
        evidence_records=tuple(evidence_records),
        warnings=tuple(warnings),
    )


def parse_frd_pdf(
    path: str | Path,
) -> FRDParseResult:
    """Load and parse one FRD PDF."""

    loaded_pdf = load_pdf(
        path,
        sort_text=True,
    )

    return parse_frd_document(loaded_pdf)


def parse_frd_directory(
    directory: str | Path,
) -> tuple[FRDParseResult, ...]:
    """Parse every PDF in an FRD directory."""

    directory_path = Path(directory).expanduser().resolve()

    if not directory_path.exists():
        raise FileNotFoundError(
            f"FRD directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"FRD path is not a directory: {directory_path}"
        )

    pdf_paths = sorted(
        (
            path
            for path in directory_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda item: item.name.lower(),
    )

    return tuple(
        parse_frd_pdf(path)
        for path in pdf_paths
    )
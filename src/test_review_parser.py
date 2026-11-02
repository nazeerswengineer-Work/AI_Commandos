"""
Test-report and review-record parsing for REQTRACE AI.

This parser preserves explicit test outcomes and review limitations without
making engineering assessment or closure decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.evidence import (
    EvidenceRecord,
    build_review_evidence_id,
    build_test_evidence_id,
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
    normalize_optional_observation_id,
    normalize_optional_requirement_id,
    normalize_revision,
    normalize_test_case_id,
    normalize_whitespace,
)
from src.observation_parser import (
    extract_field,
    extract_first_available_section,
    remove_observation_page_noise,
)


class EvidenceReportParsingError(RuntimeError):
    """Raised when test or review evidence cannot be parsed."""


@dataclass(frozen=True)
class ParsedTestReport:
    """Fields explicitly extracted from one test report."""

    test_case_id: str
    revision: str | None
    requirement_id: str | None
    result_status: str | None
    software_baseline: str | None

    preconditions: str | None
    procedure: str | None
    expected_result: str | None
    actual_result: str | None
    result_summary: str | None


@dataclass(frozen=True)
class ParsedReviewRecord:
    """Fields explicitly extracted from one review record."""

    review_id: str
    observation_id: str | None
    review_status: str | None
    decision_authority: str | None
    closure_status: str | None

    evidence_summary: str | None
    missing_evidence: str | None
    traceability_finding: str | None
    required_action: str | None
    conflict: str | None
    blocking_item: str | None
    original_observation: str | None
    closure_determination: str | None


@dataclass(frozen=True)
class EvidenceReportWarning:
    """Non-fatal report parsing issue."""

    code: str
    message: str
    document_name: str
    page_number: int | None = None


@dataclass(frozen=True)
class TestParseResult:
    document_name: str
    test_report: ParsedTestReport
    evidence_records: tuple[EvidenceRecord, ...]
    warnings: tuple[EvidenceReportWarning, ...]

TestParseResult.__test__ = False

@dataclass(frozen=True)
class ReviewParseResult:
    document_name: str
    review_record: ParsedReviewRecord
    evidence_records: tuple[EvidenceRecord, ...]
    warnings: tuple[EvidenceReportWarning, ...]


def join_clean_pdf_pages(
    loaded_pdf: LoadedPDF,
) -> str:
    """Join cleaned searchable pages."""

    cleaned_pages: list[str] = []

    for page in loaded_pdf.pages:
        if page.is_empty:
            continue

        text = remove_observation_page_noise(
            page.text
        )

        if text:
            cleaned_pages.append(text)

    return "\n\n".join(cleaned_pages).strip()


def extract_numeric_revision(
    text: str,
    document_name: str,
) -> str | None:
    """Extract revision from text or filename."""

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


def choose_test_page(
    loaded_pdf: LoadedPDF,
    test_case_id: str,
) -> int:
    """Return the page containing the explicit test ID."""

    for page in loaded_pdf.pages:
        raw_id = extract_field(
            page.text,
            "Test Case ID",
        )

        if raw_id is None:
            continue

        try:
            parsed_id = normalize_test_case_id(
                raw_id
            )
        except ValueError:
            continue

        if parsed_id == test_case_id:
            return page.page_number

    if not loaded_pdf.non_empty_pages:
        raise EvidenceReportParsingError(
            f"Test report '{loaded_pdf.document_name}' "
            "contains no searchable text."
        )

    return loaded_pdf.non_empty_pages[0].page_number


def choose_review_page(
    loaded_pdf: LoadedPDF,
    review_id: str,
) -> int:
    """Return the page containing the explicit review ID."""

    for page in loaded_pdf.pages:
        raw_id = extract_field(
            page.text,
            "Review ID",
        )

        if raw_id and normalize_whitespace(raw_id) == review_id:
            return page.page_number

    if not loaded_pdf.non_empty_pages:
        raise EvidenceReportParsingError(
            f"Review report '{loaded_pdf.document_name}' "
            "contains no searchable text."
        )

    return loaded_pdf.non_empty_pages[0].page_number


def parse_test_fields(
    text: str,
    document_name: str,
) -> ParsedTestReport:
    """Extract explicit test-report fields."""

    raw_test_id = extract_field(
        text,
        "Test Case ID",
    )

    if raw_test_id is None:
        match = re.search(
            r"(?i)(TC[\s_-]*\d+)",
            document_name,
        )

        if match is None:
            raise EvidenceReportParsingError(
                f"Test case ID not found in '{document_name}'."
            )

        raw_test_id = match.group(1)

    raw_requirement = extract_field(
        text,
        "Requirement ID",
    )

    return ParsedTestReport(
        test_case_id=normalize_test_case_id(
            raw_test_id
        ),
        revision=extract_numeric_revision(
            text,
            document_name,
        ),
        requirement_id=(
            normalize_optional_requirement_id(
                raw_requirement
            )
        ),
        result_status=extract_field(
            text,
            "Result Status",
        ),
        software_baseline=extract_field(
            text,
            "Software Baseline",
        ),
        preconditions=extract_first_available_section(
            text,
            ("Preconditions",),
        ),
        procedure=extract_first_available_section(
            text,
            (
                "Procedure",
                "Planned Procedure",
            ),
        ),
        expected_result=extract_first_available_section(
            text,
            (
                "Expected Result",
                "Expected Results",
            ),
        ),
        actual_result=extract_first_available_section(
            text,
            (
                "Actual Result",
                "Actual Results",
                "Execution Record",
            ),
        ),
        result_summary=extract_first_available_section(
            text,
            ("Result",),
        ),
    )


def parse_review_fields(
    text: str,
    document_name: str,
) -> ParsedReviewRecord:
    """Extract explicit review-record fields."""

    review_id = extract_field(
        text,
        "Review ID",
    )

    if review_id is None:
        review_id = Path(
            document_name
        ).stem

    raw_observation_id = extract_field(
        text,
        "Observation ID",
    )

    return ParsedReviewRecord(
        review_id=normalize_whitespace(
            review_id
        ),
        observation_id=(
            normalize_optional_observation_id(
                raw_observation_id
            )
        ),
        review_status=extract_field(
            text,
            "Review Status",
        ),
        decision_authority=extract_field(
            text,
            "Decision Authority",
        ),
        closure_status=extract_field(
            text,
            "Closure Status",
        ),
        evidence_summary=extract_first_available_section(
            text,
            ("Evidence Summary",),
        ),
        missing_evidence=extract_first_available_section(
            text,
            (
                "Missing Evidence",
                "Missing Closure Evidence",
            ),
        ),
        traceability_finding=extract_first_available_section(
            text,
            ("Traceability Finding",),
        ),
        required_action=extract_first_available_section(
            text,
            ("Required Action",),
        ),
        conflict=extract_first_available_section(
            text,
            ("Conflict",),
        ),
        blocking_item=extract_first_available_section(
            text,
            (
                "Blocking Item",
                "Blocking Items",
            ),
        ),
        original_observation=extract_first_available_section(
            text,
            ("Original Observation",),
        ),
        closure_determination=extract_first_available_section(
            text,
            ("Closure Determination",),
        ),
    )


def make_report_chunk_id(
    prefix: str,
    document_name: str,
    engineering_id: str,
    page_number: int,
) -> str:
    """Create a deterministic PDF evidence chunk ID."""

    safe_stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        Path(document_name).stem,
    ).strip("-")

    safe_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        engineering_id,
    ).strip("-")

    return (
        f"{prefix}-{safe_stem}"
        f"-PAGE-{page_number}"
        f"-{safe_id}"
    )


def parse_test_pdf(
    path: str | Path,
) -> TestParseResult:
    """Parse one test-report PDF."""

    loaded_pdf = load_pdf(
        path,
        sort_text=True,
    )

    content = join_clean_pdf_pages(
        loaded_pdf
    )

    if not content:
        raise EvidenceReportParsingError(
            f"Test report '{loaded_pdf.document_name}' "
            "contains no searchable text."
        )

    fields = parse_test_fields(
        content,
        loaded_pdf.document_name,
    )

    page_number = choose_test_page(
        loaded_pdf,
        fields.test_case_id,
    )

    evidence_id = build_test_evidence_id(
        document_name=loaded_pdf.document_name,
        page_number=page_number,
        test_case_id=fields.test_case_id,
    )

    metadata = EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.TEST,
        document_name=loaded_pdf.document_name,
        document_revision=fields.revision,
        page_number=page_number,
        requirement_id=fields.requirement_id,
        test_case_id=fields.test_case_id,
        result_status=fields.result_status,
        file_name=loaded_pdf.document_name,
        file_path=str(loaded_pdf.file_path),
        chunk_id=make_report_chunk_id(
            "TEST",
            loaded_pdf.document_name,
            fields.test_case_id,
            page_number,
        ),
        content_hash=calculate_content_hash(
            content
        ),
        parsing_method=ParsingMethod.TEST_CASE,
    )

    evidence = EvidenceRecord(
        content=content,
        metadata=metadata,
    )

    warnings: list[EvidenceReportWarning] = []

    if fields.result_status is None:
        warnings.append(
            EvidenceReportWarning(
                code="MISSING_RESULT_STATUS",
                message=(
                    "The test result status was not explicitly found."
                ),
                document_name=loaded_pdf.document_name,
            )
        )

    return TestParseResult(
        document_name=loaded_pdf.document_name,
        test_report=fields,
        evidence_records=(evidence,),
        warnings=tuple(warnings),
    )


def parse_review_pdf(
    path: str | Path,
) -> ReviewParseResult:
    """Parse one review-record PDF."""

    loaded_pdf = load_pdf(
        path,
        sort_text=True,
    )

    content = join_clean_pdf_pages(
        loaded_pdf
    )

    if not content:
        raise EvidenceReportParsingError(
            f"Review report '{loaded_pdf.document_name}' "
            "contains no searchable text."
        )

    fields = parse_review_fields(
        content,
        loaded_pdf.document_name,
    )

    page_number = choose_review_page(
        loaded_pdf,
        fields.review_id,
    )

    chunk_id = make_report_chunk_id(
        "REVIEW",
        loaded_pdf.document_name,
        fields.review_id,
        page_number,
    )

    # EvidenceRecord.build_evidence_id() uses the observation ID
    # as the primary anchor for review evidence when available.
    # If no engineering ID is available, it falls back to chunk_id.
    review_anchor = (
        fields.observation_id
        or chunk_id
    )

    evidence_id = build_review_evidence_id(
        document_name=loaded_pdf.document_name,
        page_number=page_number,
        review_anchor=review_anchor,
    )

    metadata = EvidenceMetadata(
        evidence_id=evidence_id,
        document_type=DocumentType.REVIEW,
        document_name=loaded_pdf.document_name,
        page_number=page_number,
        observation_id=fields.observation_id,
        result_status=(
            fields.closure_status
            or fields.review_status
        ),
        file_name=loaded_pdf.document_name,
        file_path=str(loaded_pdf.file_path),
        chunk_id=chunk_id,
        content_hash=calculate_content_hash(
            content
        ),
        parsing_method=ParsingMethod.SECTION,
    )

    evidence = EvidenceRecord(
        content=content,
        metadata=metadata,
    )

    warnings: list[EvidenceReportWarning] = []

    if fields.observation_id is None:
        warnings.append(
            EvidenceReportWarning(
                code="MISSING_OBSERVATION_REFERENCE",
                message=(
                    "No observation ID was explicitly found."
                ),
                document_name=loaded_pdf.document_name,
            )
        )

    return ReviewParseResult(
        document_name=loaded_pdf.document_name,
        review_record=fields,
        evidence_records=(evidence,),
        warnings=tuple(warnings),
    )


def parse_test_directory(
    directory: str | Path,
) -> tuple[TestParseResult, ...]:
    """Parse all test-report PDFs."""

    path = Path(directory).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Test directory does not exist: {path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Test path is not a directory: {path}"
        )

    pdf_paths = sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file()
            and item.suffix.lower() == ".pdf"
        ),
        key=lambda item: item.name.lower(),
    )

    return tuple(
        parse_test_pdf(pdf_path)
        for pdf_path in pdf_paths
    )


def parse_review_directory(
    directory: str | Path,
) -> tuple[ReviewParseResult, ...]:
    """Parse all review PDFs."""

    path = Path(directory).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Review directory does not exist: {path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Review path is not a directory: {path}"
        )

    pdf_paths = sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file()
            and item.suffix.lower() == ".pdf"
        ),
        key=lambda item: item.name.lower(),
    )

    return tuple(
        parse_review_pdf(pdf_path)
        for pdf_path in pdf_paths
    )
"""
Generate fictional engineering evidence for REQTRACE AI Phase 3.

The generated documents and source code are synthetic. They must not be used
as real engineering, certification, safety, airworthiness, production, or
closure evidence.

This script creates deterministic demonstration data for:

1. SUPPORTED
2. NOT SUPPORTED
3. INCONCLUSIVE
4. Incorrect requirement reference
5. FRD revision conflict
6. Observation supported but closure not established

The script does not create embeddings or ChromaDB collections.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
FRD_DIR = DOCUMENTS_DIR / "frd"
OBSERVATIONS_DIR = DOCUMENTS_DIR / "observations"
CODE_DIR = DOCUMENTS_DIR / "code"
TEST_REPORTS_DIR = DOCUMENTS_DIR / "tests"
REVIEWS_DIR = DOCUMENTS_DIR / "reviews"
DATA_DIR = BASE_DIR / "data"

GENERATOR_VERSION = "1.0.0"

SAFETY_NOTICE = (
    "FICTIONAL TRAINING DATA - NOT FOR ENGINEERING APPROVAL, "
    "CERTIFICATION, SAFETY APPROVAL, AIRWORTHINESS APPROVAL, "
    "PRODUCTION RELEASE, OR OBSERVATION CLOSURE."
)

APPLICATION_NOTICE = (
    "ReqTrace AI provides evidence-based review support using the retrieved "
    "documents. The result is not a substitute for authorized engineering, "
    "quality, safety, certification, or release approval."
)

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT_MARGIN = 54
RIGHT_MARGIN = 54
TOP_MARGIN = 62
BOTTOM_MARGIN = 58
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

NAVY = (0.05, 0.14, 0.28)
BLUE = (0.08, 0.35, 0.65)
LIGHT_BLUE = (0.91, 0.95, 0.99)
DARK_TEXT = (0.10, 0.13, 0.18)
MUTED_TEXT = (0.35, 0.40, 0.47)
RED = (0.72, 0.12, 0.12)
WHITE = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class DocumentSection:
    """One titled section in a synthetic PDF."""

    heading: str
    body: str


@dataclass(frozen=True)
class SyntheticPDF:
    """Specification for one synthetic PDF."""

    file_name: str
    directory: Path
    title: str
    subtitle: str
    metadata: dict[str, str]
    sections: tuple[DocumentSection, ...]


@dataclass(frozen=True)
class CodeFile:
    """Specification for one synthetic source-code file."""

    file_name: str
    content: str


def ensure_directories() -> None:
    """Create only the directories required for Phase 3."""

    for directory in (
        FRD_DIR,
        OBSERVATIONS_DIR,
        CODE_DIR,
        TEST_REPORTS_DIR,
        REVIEWS_DIR,
        DATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def normalize_text(value: str) -> str:
    """Normalize line endings and remove trailing whitespace."""

    lines = [line.rstrip() for line in value.strip().splitlines()]
    return "\n".join(lines) + "\n"


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a generated file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def add_page_header(
    page: pymupdf.Page,
    title: str,
    subtitle: str,
) -> float:
    """Draw the consistent PDF page header and return the next Y position."""

    page.draw_rect(
        pymupdf.Rect(0, 0, PAGE_WIDTH, 92),
        color=NAVY,
        fill=NAVY,
    )

    page.insert_text(
        pymupdf.Point(LEFT_MARGIN, 34),
        title,
        fontsize=16,
        fontname="helv",
        color=WHITE,
    )

    page.insert_text(
        pymupdf.Point(LEFT_MARGIN, 58),
        subtitle,
        fontsize=9,
        fontname="helv",
        color=(0.82, 0.88, 0.95),
    )

    page.insert_text(
        pymupdf.Point(LEFT_MARGIN, 78),
        "REQTRACE AI SYNTHETIC EVIDENCE",
        fontsize=7.5,
        fontname="helv",
        color=(0.65, 0.78, 0.92),
    )

    return 118


def add_page_footer(
    page: pymupdf.Page,
    page_number: int,
) -> None:
    """Add the synthetic-data notice and human-readable page number."""

    page.draw_line(
        pymupdf.Point(LEFT_MARGIN, PAGE_HEIGHT - 43),
        pymupdf.Point(PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 43),
        color=(0.75, 0.78, 0.82),
        width=0.6,
    )

    page.insert_text(
        pymupdf.Point(LEFT_MARGIN, PAGE_HEIGHT - 27),
        "Fictional demonstration evidence only",
        fontsize=7.5,
        fontname="helv",
        color=MUTED_TEXT,
    )

    page.insert_text(
        pymupdf.Point(PAGE_WIDTH - RIGHT_MARGIN - 48, PAGE_HEIGHT - 27),
        f"Page {page_number}",
        fontsize=7.5,
        fontname="helv",
        color=MUTED_TEXT,
    )


def estimate_wrapped_height(
    text: str,
    font_size: float,
    width: float,
    line_height: float,
) -> float:
    """Estimate text height for conservative page-break decisions."""

    average_character_width = font_size * 0.52
    characters_per_line = max(
        20,
        int(width / average_character_width),
    )

    total_lines = 0

    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            total_lines += 1
            continue

        wrapped = textwrap.wrap(
            paragraph,
            width=characters_per_line,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        total_lines += max(1, len(wrapped))

    return total_lines * line_height


def insert_wrapped_text(
    page: pymupdf.Page,
    text: str,
    y_position: float,
    *,
    font_size: float = 9.5,
    color: tuple[float, float, float] = DARK_TEXT,
    indent: float = 0,
    line_height: float = 13,
) -> float:
    """Insert wrapped text and return the Y position below it."""

    box = pymupdf.Rect(
        LEFT_MARGIN + indent,
        y_position,
        PAGE_WIDTH - RIGHT_MARGIN,
        PAGE_HEIGHT - BOTTOM_MARGIN,
    )

    result = page.insert_textbox(
        box,
        text,
        fontsize=font_size,
        fontname="helv",
        color=color,
        lineheight=line_height / font_size,
        align=pymupdf.TEXT_ALIGN_LEFT,
    )

    if result < 0:
        raise RuntimeError(
            "Synthetic PDF layout overflowed unexpectedly. "
            "Reduce section size or adjust page-breaking logic."
        )

    used_height = estimate_wrapped_height(
        text=text,
        font_size=font_size,
        width=box.width,
        line_height=line_height,
    )

    return y_position + used_height


def create_new_page(
    document: pymupdf.Document,
    title: str,
    subtitle: str,
) -> tuple[pymupdf.Page, float]:
    """Create and initialize one PDF page."""

    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y_position = add_page_header(page, title, subtitle)
    return page, y_position


def create_pdf(specification: SyntheticPDF) -> Path:
    """Generate one readable, searchable synthetic PDF."""

    output_path = specification.directory / specification.file_name
    document = pymupdf.open()

    try:
        page, y_position = create_new_page(
            document,
            specification.title,
            specification.subtitle,
        )

        metadata_lines = [
            f"{label}: {value}"
            for label, value in specification.metadata.items()
        ]
        metadata_text = "\n".join(metadata_lines)

        metadata_height = estimate_wrapped_height(
            metadata_text,
            font_size=9,
            width=CONTENT_WIDTH - 20,
            line_height=12,
        ) + 18

        page.draw_rect(
            pymupdf.Rect(
                LEFT_MARGIN,
                y_position,
                PAGE_WIDTH - RIGHT_MARGIN,
                y_position + metadata_height,
            ),
            color=(0.72, 0.82, 0.92),
            fill=LIGHT_BLUE,
            width=0.8,
        )

        y_position = insert_wrapped_text(
            page,
            metadata_text,
            y_position + 9,
            font_size=9,
            indent=10,
            line_height=12,
        )
        y_position += 14

        for section in specification.sections:
            section_height = (
                30
                + estimate_wrapped_height(
                    section.body,
                    font_size=9.5,
                    width=CONTENT_WIDTH,
                    line_height=13,
                )
            )

            remaining_height = (
                PAGE_HEIGHT - BOTTOM_MARGIN - y_position
            )

            if section_height > remaining_height:
                add_page_footer(page, len(document))
                page, y_position = create_new_page(
                    document,
                    specification.title,
                    specification.subtitle,
                )

            page.insert_text(
                pymupdf.Point(LEFT_MARGIN, y_position + 12),
                section.heading,
                fontsize=11,
                fontname="hebo",
                color=BLUE,
            )
            y_position += 20

            y_position = insert_wrapped_text(
                page,
                section.body,
                y_position,
                font_size=9.5,
                line_height=13,
            )
            y_position += 14

        notice_height = estimate_wrapped_height(
            SAFETY_NOTICE,
            font_size=8,
            width=CONTENT_WIDTH - 20,
            line_height=11,
        ) + 18

        if y_position + notice_height > PAGE_HEIGHT - BOTTOM_MARGIN:
            add_page_footer(page, len(document))
            page, y_position = create_new_page(
                document,
                specification.title,
                specification.subtitle,
            )

        page.draw_rect(
            pymupdf.Rect(
                LEFT_MARGIN,
                y_position,
                PAGE_WIDTH - RIGHT_MARGIN,
                y_position + notice_height,
            ),
            color=(0.90, 0.60, 0.60),
            fill=(1.0, 0.94, 0.94),
            width=0.8,
        )

        insert_wrapped_text(
            page,
            SAFETY_NOTICE,
            y_position + 9,
            font_size=8,
            color=RED,
            indent=10,
            line_height=11,
        )

        for page_index in range(len(document)):
            current_page = document[page_index]
            add_page_footer(current_page, page_index + 1)

        document.set_metadata(
            {
                "title": specification.title,
                "subject": "REQTRACE AI synthetic evidence",
                "author": "REQTRACE AI synthetic data generator",
                "keywords": (
                    "fictional, synthetic, training, evidence, traceability"
                ),
                "creator": "REQTRACE AI Phase 3",
                "producer": "PyMuPDF",
            }
        )

        document.save(
            output_path,
            garbage=4,
            deflate=True,
            clean=True,
        )
    finally:
        document.close()

    return output_path


def create_code_file(specification: CodeFile) -> Path:
    """Write one deterministic synthetic source-code file."""

    output_path = CODE_DIR / specification.file_name
    output_path.write_text(
        normalize_text(specification.content),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def section(heading: str, body: str) -> DocumentSection:
    """Convenience function for creating document sections."""

    return DocumentSection(
        heading=heading,
        body=body.strip(),
    )


def build_pdf_specifications() -> tuple[SyntheticPDF, ...]:
    """Return all Phase 3 synthetic PDF specifications."""

    return (
        SyntheticPDF(
            file_name="FRD_Input_Validation_Rev01.pdf",
            directory=FRD_DIR,
            title="Functional Requirements Document",
            subtitle="Input Validation and Data Processing",
            metadata={
                "Document ID": "FRD-IV-001",
                "Revision": "Rev01",
                "Release Status": "Synthetic baseline",
                "Effective Date": "Not applicable - fictional data",
            },
            sections=(
                section(
                    "1. Purpose",
                    """
                    This fictional document defines demonstration requirements
                    for input validation and data processing. It is created only
                    for testing evidence retrieval and traceability behavior.
                    """,
                ),
                section(
                    "2. Requirement REQ-017",
                    """
                    Requirement ID: REQ-017

                    The system shall validate input parameters and handle null
                    values without causing a system exception.

                    Expected behavior:
                    When process_data() receives a null input pointer, the
                    function shall reject the input, return the defined error
                    status DATA_STATUS_INVALID_INPUT, and shall not dereference
                    the null pointer.
                    """,
                ),
                section(
                    "3. Verification",
                    """
                    Verification method: Test and code review.

                    The test shall invoke process_data() using a null pointer and
                    confirm that the defined error status is returned without a
                    system exception.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="FRD_Command_Processing_Rev01.pdf",
            directory=FRD_DIR,
            title="Functional Requirements Document",
            subtitle="Command Validation",
            metadata={
                "Document ID": "FRD-CV-001",
                "Revision": "Rev01",
                "Release Status": "Synthetic baseline",
                "Effective Date": "Not applicable - fictional data",
            },
            sections=(
                section(
                    "1. Requirement REQ-021",
                    """
                    Requirement ID: REQ-021

                    The system shall reject commands whose command identifier is
                    outside the inclusive range 1 through 16.

                    Expected behavior:
                    validate_command() shall return COMMAND_INVALID_ID for an
                    identifier below 1 or above 16. The rejected command shall
                    not be submitted for execution.
                    """,
                ),
                section(
                    "2. Verification",
                    """
                    Verification method: Boundary-value testing and code review.

                    The verification shall include command identifiers 0, 1,
                    16, and 17.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="FRD_Mode_Control_Rev01.pdf",
            directory=FRD_DIR,
            title="Functional Requirements Document",
            subtitle="Mode-Control Processing",
            metadata={
                "Document ID": "FRD-MC-001",
                "Revision": "Rev01",
                "Release Status": "Synthetic baseline",
                "Effective Date": "Not applicable - fictional data",
            },
            sections=(
                section(
                    "1. Requirement REQ-033",
                    """
                    Requirement ID: REQ-033

                    The system shall transition from STANDBY mode to ACTIVE mode
                    only when the enable request is valid and all required
                    readiness conditions are satisfied.

                    Expected behavior:
                    If any required readiness condition is unavailable or false,
                    the requested transition shall be rejected and the current
                    mode shall remain STANDBY.
                    """,
                ),
                section(
                    "2. Verification",
                    """
                    Verification method: Code review and scenario-based testing.

                    Verification evidence must include all readiness inputs and
                    the resulting mode state.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="FRD_Alert_Handling_Rev01.pdf",
            directory=FRD_DIR,
            title="Functional Requirements Document",
            subtitle="Alert Processing - Superseded Candidate",
            metadata={
                "Document ID": "FRD-AH-001",
                "Revision": "Rev01",
                "Release Status": "Synthetic revision candidate",
                "Effective Date": "Not specified",
            },
            sections=(
                section(
                    "1. Requirement REQ-041",
                    """
                    Requirement ID: REQ-041

                    The system shall record each accepted alert in the local
                    diagnostic log.

                    This requirement does not define alert-reset timing.
                    """,
                ),
                section(
                    "2. Requirement REQ-050",
                    """
                    Requirement ID: REQ-050

                    When the alert-reset command is accepted, the system shall
                    clear the active alert indication within 500 milliseconds.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="FRD_Alert_Handling_Rev02.pdf",
            directory=FRD_DIR,
            title="Functional Requirements Document",
            subtitle="Alert Processing - Alternative Revision Candidate",
            metadata={
                "Document ID": "FRD-AH-001",
                "Revision": "Rev02",
                "Release Status": "Synthetic revision candidate",
                "Effective Date": "Not specified",
            },
            sections=(
                section(
                    "1. Requirement REQ-042",
                    """
                    Requirement ID: REQ-042

                    When alert severity is HIGH, the system shall display the
                    priority alert indicator and record the alert identifier.

                    This requirement is the relevant candidate for an
                    observation about missing HIGH-severity indication.
                    """,
                ),
                section(
                    "2. Requirement REQ-050",
                    """
                    Requirement ID: REQ-050

                    When the alert-reset command is accepted, the system shall
                    clear the active alert indication within 1000 milliseconds.

                    This timing differs from Rev01. The applicable revision must
                    be established before assessing compliance.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="FRD_Recovery_Control_Rev01.pdf",
            directory=FRD_DIR,
            title="Functional Requirements Document",
            subtitle="Recovery-Control Processing",
            metadata={
                "Document ID": "FRD-RC-001",
                "Revision": "Rev01",
                "Release Status": "Synthetic baseline",
                "Effective Date": "Not applicable - fictional data",
            },
            sections=(
                section(
                    "1. Requirement REQ-060",
                    """
                    Requirement ID: REQ-060

                    Upon receiving a valid recovery request, the system shall
                    initialize the recovery counter to zero before beginning
                    recovery processing.
                    """,
                ),
                section(
                    "2. Verification",
                    """
                    Verification method: Code review and test.

                    Closure of a reported defect additionally requires evidence
                    of the implemented corrective action and an acceptable
                    verification result for the corrected software.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Observation_Report_OBS-023.pdf",
            directory=OBSERVATIONS_DIR,
            title="Software Observation Report",
            subtitle="OBS-023 - Null Input Handling",
            metadata={
                "Observation ID": "OBS-023",
                "Revision": "Rev01",
                "Status": "OPEN",
                "Severity": "Major",
                "Referenced Requirement": "REQ-017",
                "Referenced Test": "TC-102",
                "Referenced File": "data_processor.c",
                "Referenced Function": "process_data",
            },
            sections=(
                section(
                    "1. Observation Description",
                    """
                    During synthetic test execution, process_data() was invoked
                    with a null input pointer. The supplied test report records
                    that a system exception occurred.
                    """,
                ),
                section(
                    "2. Expected Result",
                    """
                    The function should reject the null pointer and return
                    DATA_STATUS_INVALID_INPUT without causing a system
                    exception.
                    """,
                ),
                section(
                    "3. Actual Result",
                    """
                    The function dereferenced the input pointer before checking
                    whether the pointer was null. The synthetic execution report
                    records a system exception.
                    """,
                ),
                section(
                    "4. Execution Conditions",
                    """
                    Demonstration build: SYNTH-BUILD-01
                    Input condition: input pointer equals NULL
                    Test environment: fictional unit-test harness
                    """,
                ),
                section(
                    "5. Observation Status",
                    """
                    Status: OPEN

                    No closure recommendation is made by this observation
                    report.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Observation_Report_OBS-031.pdf",
            directory=OBSERVATIONS_DIR,
            title="Software Observation Report",
            subtitle="OBS-031 - Command Identifier Validation",
            metadata={
                "Observation ID": "OBS-031",
                "Revision": "Rev01",
                "Status": "OPEN",
                "Severity": "Minor",
                "Referenced Requirement": "REQ-021",
                "Referenced Test": "TC-201",
                "Referenced File": "command_validator.c",
                "Referenced Function": "validate_command",
            },
            sections=(
                section(
                    "1. Observation Description",
                    """
                    The observation claims that validate_command() accepts
                    command identifier 17 and permits the command to continue
                    to execution.
                    """,
                ),
                section(
                    "2. Expected Result",
                    """
                    Command identifier 17 should be rejected with
                    COMMAND_INVALID_ID.
                    """,
                ),
                section(
                    "3. Reported Actual Result",
                    """
                    The observation reports that identifier 17 was accepted.
                    No independent execution log is included in this observation
                    report.
                    """,
                ),
                section(
                    "4. Referenced Evidence",
                    """
                    Requirement: REQ-021
                    Source file: command_validator.c
                    Function: validate_command()
                    Test case: TC-201
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Observation_Report_OBS-044.pdf",
            directory=OBSERVATIONS_DIR,
            title="Software Observation Report",
            subtitle="OBS-044 - Readiness Condition Handling",
            metadata={
                "Observation ID": "OBS-044",
                "Revision": "Rev01",
                "Status": "OPEN",
                "Severity": "Major",
                "Referenced Requirement": "REQ-033",
                "Referenced Test": "TC-301",
                "Referenced File": "mode_controller.c",
                "Referenced Function": "request_active_mode",
            },
            sections=(
                section(
                    "1. Observation Description",
                    """
                    The observation claims that the software enters ACTIVE mode
                    when one readiness input is unavailable.
                    """,
                ),
                section(
                    "2. Evidence Limitation",
                    """
                    The supplied source-code extract contains only local input
                    validation. The final mode-assignment logic is not included.

                    The supplied test report is marked NOT EXECUTED. Therefore,
                    runtime behavior is not established by the available
                    evidence.
                    """,
                ),
                section(
                    "3. Assessment Constraint",
                    """
                    The observation cannot be confirmed or contradicted from the
                    available partial code and unexecuted test evidence.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Observation_Report_OBS-052.pdf",
            directory=OBSERVATIONS_DIR,
            title="Software Observation Report",
            subtitle="OBS-052 - Alert Indication",
            metadata={
                "Observation ID": "OBS-052",
                "Revision": "Rev01",
                "Status": "OPEN",
                "Severity": "Major",
                "Referenced Requirement": "REQ-041",
                "Referenced Test": "Not provided",
                "Referenced File": "alert_manager.c",
                "Referenced Function": "process_alert",
            },
            sections=(
                section(
                    "1. Observation Description",
                    """
                    The observation claims that a HIGH-severity alert does not
                    activate the priority alert indicator.
                    """,
                ),
                section(
                    "2. Referenced Requirement",
                    """
                    The observation explicitly references REQ-041.

                    REQ-041 addresses diagnostic logging and does not define the
                    HIGH-severity priority indication. Similar wording appears
                    under REQ-042, but this observation does not reference
                    REQ-042.
                    """,
                ),
                section(
                    "3. Traceability Concern",
                    """
                    The applicable requirement reference must be corrected or
                    reviewed by an authorized reviewer. The system must not
                    silently replace REQ-041 with REQ-042 based only on semantic
                    similarity.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Observation_Report_OBS-061.pdf",
            directory=OBSERVATIONS_DIR,
            title="Software Observation Report",
            subtitle="OBS-061 - Alert Reset Timing",
            metadata={
                "Observation ID": "OBS-061",
                "Revision": "Rev01",
                "Status": "OPEN",
                "Severity": "Major",
                "Referenced Requirement": "REQ-050",
                "Applicable FRD Revision": "Not specified",
                "Referenced Test": "Not provided",
                "Referenced File": "alert_manager.c",
                "Referenced Function": "reset_alert",
            },
            sections=(
                section(
                    "1. Observation Description",
                    """
                    The observation reports that the alert indication cleared
                    760 milliseconds after the reset command was accepted.
                    """,
                ),
                section(
                    "2. Revision Conflict",
                    """
                    FRD Rev01 requires clearing within 500 milliseconds.
                    FRD Rev02 requires clearing within 1000 milliseconds.

                    The observation does not identify the applicable FRD
                    revision for the tested software baseline.
                    """,
                ),
                section(
                    "3. Assessment Constraint",
                    """
                    A deterministic compliance conclusion is not permitted until
                    the applicable FRD revision is established.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Observation_Report_OBS-070.pdf",
            directory=OBSERVATIONS_DIR,
            title="Software Observation Report",
            subtitle="OBS-070 - Recovery Counter Initialization",
            metadata={
                "Observation ID": "OBS-070",
                "Revision": "Rev01",
                "Status": "OPEN",
                "Severity": "Major",
                "Referenced Requirement": "REQ-060",
                "Referenced Test": "TC-401",
                "Referenced File": "recovery_manager.c",
                "Referenced Function": "start_recovery",
            },
            sections=(
                section(
                    "1. Observation Description",
                    """
                    The recovery counter is not initialized to zero before
                    recovery processing begins.
                    """,
                ),
                section(
                    "2. Expected Result",
                    """
                    start_recovery() should assign zero to the recovery counter
                    before processing the recovery sequence.
                    """,
                ),
                section(
                    "3. Reported Actual Result",
                    """
                    The supplied baseline code leaves the preceding counter value
                    unchanged. TC-401 records a failing result for the original
                    baseline.
                    """,
                ),
                section(
                    "4. Closure Limitation",
                    """
                    The observation evidence may support the original reported
                    deviation. However, no corrected code, change record, or
                    passing post-correction verification result is supplied.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Test_Report_TC-102.pdf",
            directory=TEST_REPORTS_DIR,
            title="Software Test Report",
            subtitle="TC-102 - Null Input Validation",
            metadata={
                "Test Case ID": "TC-102",
                "Revision": "Rev01",
                "Requirement ID": "REQ-017",
                "Result Status": "FAIL",
                "Software Baseline": "SYNTH-BUILD-01",
            },
            sections=(
                section(
                    "1. Preconditions",
                    """
                    The data-processing component is initialized.
                    The input pointer supplied to process_data() is NULL.
                    """,
                ),
                section(
                    "2. Procedure",
                    """
                    Step 1: Invoke process_data(NULL).
                    Step 2: Record the returned status.
                    Step 3: Monitor whether a system exception occurs.
                    """,
                ),
                section(
                    "3. Expected Result",
                    """
                    DATA_STATUS_INVALID_INPUT is returned and no system exception
                    occurs.
                    """,
                ),
                section(
                    "4. Actual Result",
                    """
                    A system exception was recorded before a valid error status
                    was returned.
                    """,
                ),
                section(
                    "5. Result",
                    """
                    Result Status: FAIL
                    The recorded result does not satisfy REQ-017.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Test_Report_TC-201.pdf",
            directory=TEST_REPORTS_DIR,
            title="Software Test Report",
            subtitle="TC-201 - Command Boundary Validation",
            metadata={
                "Test Case ID": "TC-201",
                "Revision": "Rev01",
                "Requirement ID": "REQ-021",
                "Result Status": "PASS",
                "Software Baseline": "SYNTH-BUILD-02",
            },
            sections=(
                section(
                    "1. Procedure",
                    """
                    Execute validate_command() with identifiers 0, 1, 16, and 17.
                    Record the returned validation result.
                    """,
                ),
                section(
                    "2. Expected Results",
                    """
                    Identifier 0: COMMAND_INVALID_ID
                    Identifier 1: COMMAND_VALID
                    Identifier 16: COMMAND_VALID
                    Identifier 17: COMMAND_INVALID_ID
                    """,
                ),
                section(
                    "3. Actual Results",
                    """
                    All returned values matched the expected results. Identifier
                    17 was rejected and was not submitted for execution.
                    """,
                ),
                section(
                    "4. Result",
                    """
                    Result Status: PASS
                    The test evidence contradicts the claim in OBS-031.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Test_Report_TC-301.pdf",
            directory=TEST_REPORTS_DIR,
            title="Software Test Report",
            subtitle="TC-301 - Readiness Input Scenario",
            metadata={
                "Test Case ID": "TC-301",
                "Revision": "Rev01",
                "Requirement ID": "REQ-033",
                "Result Status": "NOT EXECUTED",
                "Software Baseline": "Not recorded",
            },
            sections=(
                section(
                    "1. Planned Procedure",
                    """
                    Request ACTIVE mode while one required readiness input is
                    unavailable.
                    """,
                ),
                section(
                    "2. Expected Result",
                    """
                    The transition is rejected and the current mode remains
                    STANDBY.
                    """,
                ),
                section(
                    "3. Execution Record",
                    """
                    Result Status: NOT EXECUTED

                    No actual result, runtime log, or final mode state is
                    available.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Test_Report_TC-401.pdf",
            directory=TEST_REPORTS_DIR,
            title="Software Test Report",
            subtitle="TC-401 - Recovery Counter Baseline Test",
            metadata={
                "Test Case ID": "TC-401",
                "Revision": "Rev01",
                "Requirement ID": "REQ-060",
                "Result Status": "FAIL",
                "Software Baseline": "SYNTH-BUILD-04-ORIGINAL",
            },
            sections=(
                section(
                    "1. Procedure",
                    """
                    Set the recovery counter to a non-zero value.
                    Submit a valid recovery request.
                    Observe the counter before recovery processing begins.
                    """,
                ),
                section(
                    "2. Expected Result",
                    """
                    The counter is initialized to zero before recovery
                    processing.
                    """,
                ),
                section(
                    "3. Actual Result",
                    """
                    The preceding non-zero counter value remained present when
                    recovery processing began.
                    """,
                ),
                section(
                    "4. Result",
                    """
                    Result Status: FAIL

                    This result applies only to the original synthetic baseline.
                    It is not post-correction verification evidence.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Review_Record_OBS-023.pdf",
            directory=REVIEWS_DIR,
            title="Engineering Review Record",
            subtitle="OBS-023 Evidence Review",
            metadata={
                "Review ID": "REV-OBS-023",
                "Observation ID": "OBS-023",
                "Review Status": "Evidence collected",
                "Decision Authority": "Not provided",
            },
            sections=(
                section(
                    "1. Evidence Summary",
                    """
                    The review record identifies REQ-017, data_processor.c,
                    process_data(), and TC-102 as the supplied evidence chain.
                    """,
                ),
                section(
                    "2. Review Limitation",
                    """
                    This fictional record does not grant closure, safety,
                    certification, or release approval.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Review_Record_OBS-031.pdf",
            directory=REVIEWS_DIR,
            title="Engineering Review Record",
            subtitle="OBS-031 Contradicting Evidence Review",
            metadata={
                "Review ID": "REV-OBS-031",
                "Observation ID": "OBS-031",
                "Review Status": "Contradiction identified",
                "Decision Authority": "Not provided",
            },
            sections=(
                section(
                    "1. Evidence Summary",
                    """
                    The supplied implementation checks the valid identifier
                    range. TC-201 records that identifier 17 was rejected.
                    These sources directly contradict the observation claim.
                    """,
                ),
                section(
                    "2. Review Limitation",
                    """
                    The final disposition remains the responsibility of an
                    authorized reviewer.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Review_Record_OBS-044.pdf",
            directory=REVIEWS_DIR,
            title="Engineering Review Record",
            subtitle="OBS-044 Missing Evidence Review",
            metadata={
                "Review ID": "REV-OBS-044",
                "Observation ID": "OBS-044",
                "Review Status": "Evidence incomplete",
                "Decision Authority": "Not provided",
            },
            sections=(
                section(
                    "1. Missing Evidence",
                    """
                    The final mode-assignment implementation is unavailable.
                    TC-301 was not executed.
                    A runtime mode-transition record is unavailable.
                    """,
                ),
                section(
                    "2. Review Position",
                    """
                    The available evidence is insufficient to confirm or
                    contradict the claimed transition.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Review_Record_OBS-052.pdf",
            directory=REVIEWS_DIR,
            title="Engineering Review Record",
            subtitle="OBS-052 Traceability Review",
            metadata={
                "Review ID": "REV-OBS-052",
                "Observation ID": "OBS-052",
                "Review Status": "Reference correction required",
                "Decision Authority": "Not provided",
            },
            sections=(
                section(
                    "1. Traceability Finding",
                    """
                    OBS-052 references REQ-041, but REQ-041 concerns diagnostic
                    logging. The observation claim concerns HIGH-severity alert
                    indication, which appears in REQ-042.
                    """,
                ),
                section(
                    "2. Required Action",
                    """
                    An authorized reviewer must confirm or correct the applicable
                    requirement reference. Semantic similarity alone must not
                    create a permanent traceability link.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Review_Record_OBS-061.pdf",
            directory=REVIEWS_DIR,
            title="Engineering Review Record",
            subtitle="OBS-061 Revision Applicability Review",
            metadata={
                "Review ID": "REV-OBS-061",
                "Observation ID": "OBS-061",
                "Review Status": "Revision conflict unresolved",
                "Decision Authority": "Not provided",
            },
            sections=(
                section(
                    "1. Conflict",
                    """
                    REQ-050 Rev01 defines 500 milliseconds.
                    REQ-050 Rev02 defines 1000 milliseconds.
                    The observed value is 760 milliseconds.
                    """,
                ),
                section(
                    "2. Blocking Item",
                    """
                    The applicable requirement revision and software baseline
                    relationship are not available.
                    """,
                ),
            ),
        ),
        SyntheticPDF(
            file_name="Closure_Review_OBS-070.pdf",
            directory=REVIEWS_DIR,
            title="Closure Readiness Review",
            subtitle="OBS-070 - Corrective Action Evidence",
            metadata={
                "Review ID": "CLOSURE-OBS-070",
                "Observation ID": "OBS-070",
                "Review Status": "Closure evidence incomplete",
                "Closure Status": "CLOSURE NOT ESTABLISHED",
            },
            sections=(
                section(
                    "1. Original Observation",
                    """
                    The original observation is supported by REQ-060, the
                    original source-code baseline, and failing test TC-401.
                    """,
                ),
                section(
                    "2. Missing Closure Evidence",
                    """
                    Corrective-action description: Not available
                    Corrected source code: Not available
                    Change record: Not available
                    Post-correction test result: Not available
                    Independent verification record: Not available
                    """,
                ),
                section(
                    "3. Closure Determination",
                    """
                    CLOSURE NOT ESTABLISHED

                    Support for the original observation does not establish that
                    corrective action was implemented and verified.
                    """,
                ),
            ),
        ),
    )


def build_code_specifications() -> tuple[CodeFile, ...]:
    """Return all synthetic source-code specifications."""

    return (
        CodeFile(
            file_name="data_processor.c",
            content=r'''
/*
 * REQTRACE AI SYNTHETIC SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-017
 *   Observation: OBS-023
 *   Test: TC-102
 */

#include <stddef.h>
#include <stdint.h>

typedef enum
{
    DATA_STATUS_OK = 0,
    DATA_STATUS_INVALID_INPUT = 1
} data_status_t;

typedef struct
{
    uint32_t sample_count;
    const uint8_t *samples;
} data_input_t;

data_status_t process_data(const data_input_t *input)
{
    uint32_t count;

    /*
     * Intentional synthetic defect:
     * input is dereferenced before the null-pointer check.
     */
    count = input->sample_count;

    if (input == NULL)
    {
        return DATA_STATUS_INVALID_INPUT;
    }

    if ((count > 0U) && (input->samples == NULL))
    {
        return DATA_STATUS_INVALID_INPUT;
    }

    return DATA_STATUS_OK;
}
''',
        ),
        CodeFile(
            file_name="command_validator.c",
            content=r'''
/*
 * REQTRACE AI SYNTHETIC SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-021
 *   Observation: OBS-031
 *   Test: TC-201
 */

#include <stdint.h>

#define COMMAND_ID_MINIMUM 1U
#define COMMAND_ID_MAXIMUM 16U

typedef enum
{
    COMMAND_VALID = 0,
    COMMAND_INVALID_ID = 1
} command_status_t;

command_status_t validate_command(uint8_t command_id)
{
    if ((command_id < COMMAND_ID_MINIMUM) ||
        (command_id > COMMAND_ID_MAXIMUM))
    {
        return COMMAND_INVALID_ID;
    }

    return COMMAND_VALID;
}
''',
        ),
        CodeFile(
            file_name="mode_controller.c",
            content=r'''
/*
 * REQTRACE AI SYNTHETIC PARTIAL SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-033
 *   Observation: OBS-044
 *   Test: TC-301
 *
 * Limitation:
 *   This file intentionally omits the final mode-assignment implementation.
 */

#include <stdbool.h>

typedef enum
{
    MODE_REQUEST_REJECTED = 0,
    MODE_REQUEST_ACCEPTED = 1
} mode_request_status_t;

mode_request_status_t request_active_mode(
    bool enable_request_valid,
    bool primary_ready)
{
    if ((!enable_request_valid) || (!primary_ready))
    {
        return MODE_REQUEST_REJECTED;
    }

    /*
     * The secondary readiness check and final mode assignment are not
     * included in the supplied evidence.
     */
    return MODE_REQUEST_ACCEPTED;
}
''',
        ),
        CodeFile(
            file_name="alert_manager.c",
            content=r'''
/*
 * REQTRACE AI SYNTHETIC SOURCE CODE
 * Fictional demonstration data only.
 *
 * Candidate traceability:
 *   Requirements: REQ-041, REQ-042, REQ-050
 *   Observations: OBS-052, OBS-061
 */

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    ALERT_LOW = 0,
    ALERT_HIGH = 1
} alert_severity_t;

typedef struct
{
    bool active;
    bool priority_indicator;
    uint32_t alert_id;
} alert_state_t;

void process_alert(
    alert_state_t *state,
    uint32_t alert_id,
    alert_severity_t severity)
{
    if (state == NULL)
    {
        return;
    }

    state->active = true;
    state->alert_id = alert_id;

    if (severity == ALERT_HIGH)
    {
        state->priority_indicator = true;
    }
}

void reset_alert(
    alert_state_t *state,
    uint32_t measured_elapsed_ms)
{
    if (state == NULL)
    {
        return;
    }

    /*
     * The supplied function records the measured time but does not establish
     * which FRD revision applies to the software baseline.
     */
    if (measured_elapsed_ms >= 760U)
    {
        state->active = false;
        state->priority_indicator = false;
    }
}
''',
        ),
        CodeFile(
            file_name="recovery_manager.c",
            content=r'''
/*
 * REQTRACE AI SYNTHETIC ORIGINAL-BASELINE SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-060
 *   Observation: OBS-070
 *   Test: TC-401
 *
 * Closure limitation:
 *   No corrected implementation is supplied in the Phase 3 evidence set.
 */

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint32_t recovery_counter;
    bool recovery_active;
} recovery_state_t;

void start_recovery(
    recovery_state_t *state,
    bool request_valid)
{
    if ((state == NULL) || (!request_valid))
    {
        return;
    }

    /*
     * Intentional synthetic defect:
     * recovery_counter is not initialized to zero.
     */
    state->recovery_active = true;
}
''',
        ),
    )


def expected_outcomes() -> list[dict[str, Any]]:
    """Return machine-readable expected outcomes for later evaluation."""

    return [
        {
            "case_id": "CASE-001",
            "observation_id": "OBS-023",
            "scenario": "supported",
            "expected_assessment": "SUPPORTED",
            "expected_requirement": "REQ-017",
            "expected_revision": "Rev01",
            "expected_code_files": ["data_processor.c"],
            "expected_functions": ["process_data"],
            "expected_test_ids": ["TC-102"],
            "expected_missing_evidence": [],
            "expected_blocking_items": [],
            "expected_closure_status": "NOT ASSESSED",
            "reason": (
                "FRD, code, and explicit failed test evidence support "
                "the reported null-input deviation."
            ),
        },
        {
            "case_id": "CASE-002",
            "observation_id": "OBS-031",
            "scenario": "not_supported",
            "expected_assessment": "NOT SUPPORTED",
            "expected_requirement": "REQ-021",
            "expected_revision": "Rev01",
            "expected_code_files": ["command_validator.c"],
            "expected_functions": ["validate_command"],
            "expected_test_ids": ["TC-201"],
            "expected_missing_evidence": [],
            "expected_blocking_items": [],
            "expected_closure_status": "NOT ASSESSED",
            "reason": (
                "The supplied implementation and explicit passing test "
                "directly contradict the observation claim."
            ),
        },
        {
            "case_id": "CASE-003",
            "observation_id": "OBS-044",
            "scenario": "inconclusive_missing_evidence",
            "expected_assessment": "INCONCLUSIVE",
            "expected_requirement": "REQ-033",
            "expected_revision": "Rev01",
            "expected_code_files": ["mode_controller.c"],
            "expected_functions": ["request_active_mode"],
            "expected_test_ids": ["TC-301"],
            "expected_missing_evidence": [
                "Complete mode-assignment implementation",
                "Executed test result",
                "Runtime mode-transition evidence",
            ],
            "expected_blocking_items": [
                "Code scope is insufficient",
                "Required execution evidence is unavailable",
            ],
            "expected_closure_status": "NOT ASSESSED",
            "reason": (
                "Partial code and an unexecuted test cannot establish "
                "the reported runtime behavior."
            ),
        },
        {
            "case_id": "CASE-004",
            "observation_id": "OBS-052",
            "scenario": "incorrect_requirement_reference",
            "expected_assessment": "INCONCLUSIVE",
            "expected_requirement": "REQ-041",
            "expected_revision": "Unresolved",
            "expected_code_files": ["alert_manager.c"],
            "expected_functions": ["process_alert"],
            "expected_test_ids": [],
            "expected_missing_evidence": [
                "Authorized correction of the requirement reference",
                "Applicable test evidence",
            ],
            "expected_blocking_items": [
                "Observation references a requirement that does not define "
                "the reported behavior",
            ],
            "expected_closure_status": "NOT ASSESSED",
            "reason": (
                "The explicit REQ-041 reference must not be silently "
                "replaced with the semantically similar REQ-042."
            ),
        },
        {
            "case_id": "CASE-005",
            "observation_id": "OBS-061",
            "scenario": "frd_revision_conflict",
            "expected_assessment": "INCONCLUSIVE",
            "expected_requirement": "REQ-050",
            "expected_revision": "Conflict: Rev01 versus Rev02",
            "expected_code_files": ["alert_manager.c"],
            "expected_functions": ["reset_alert"],
            "expected_test_ids": [],
            "expected_missing_evidence": [
                "Applicable FRD revision",
                "Software-to-requirement baseline relationship",
            ],
            "expected_blocking_items": [
                "FRD revision conflict is unresolved",
            ],
            "expected_closure_status": "NOT ASSESSED",
            "reason": (
                "The observed 760 milliseconds conflicts with Rev01 but "
                "satisfies Rev02. Applicability is not established."
            ),
        },
        {
            "case_id": "CASE-006",
            "observation_id": "OBS-070",
            "scenario": "supported_but_not_ready_for_closure",
            "expected_assessment": "SUPPORTED",
            "expected_requirement": "REQ-060",
            "expected_revision": "Rev01",
            "expected_code_files": ["recovery_manager.c"],
            "expected_functions": ["start_recovery"],
            "expected_test_ids": ["TC-401"],
            "expected_missing_evidence": [
                "Corrective-action description",
                "Corrected implementation",
                "Change record",
                "Passing post-correction test",
                "Independent verification evidence",
            ],
            "expected_blocking_items": [
                "Corrective action is not demonstrated",
                "Correction verification is unavailable",
            ],
            "expected_closure_status": "CLOSURE NOT ESTABLISHED",
            "reason": (
                "The original deviation is supported, but no corrected "
                "implementation or acceptable verification evidence exists."
            ),
        },
    ]


def write_json(path: Path, payload: Any) -> Path:
    """Write deterministic, readable JSON."""

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def build_catalogue(
    generated_files: list[Path],
) -> dict[str, Any]:
    """Build a catalogue containing only generated-file facts."""

    records: list[dict[str, Any]] = []

    for path in sorted(generated_files, key=lambda item: str(item).lower()):
        relative_path = path.relative_to(BASE_DIR).as_posix()

        if path.suffix.lower() == ".pdf":
            document_type = path.parent.name
            with pymupdf.open(path) as pdf:
                page_count = len(pdf)
        elif path.suffix.lower() in {".c", ".h", ".cpp", ".hpp", ".py"}:
            document_type = "code"
            page_count = None
        else:
            document_type = "data"
            page_count = None

        records.append(
            {
                "relative_path": relative_path,
                "file_name": path.name,
                "document_type": document_type,
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
                "page_count": page_count,
            }
        )

    return {
        "dataset_name": "REQTRACE AI Phase 3 Synthetic Evidence",
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety_notice": SAFETY_NOTICE,
        "application_notice": APPLICATION_NOTICE,
        "file_count": len(records),
        "files": records,
    }


def validate_generated_pdf(path: Path) -> list:
    """Perform basic structural validation on one generated PDF."""

    errors: list[str] = []

    try:
        with pymupdf.open(path) as document:
            if len(document) < 1:
                errors.append(f"{path.name}: PDF has no pages.")
                return errors

            combined_text = "\n".join(
                page.get_text("text")
                for page in document
            )

            if "REQTRACE AI SYNTHETIC EVIDENCE" not in combined_text:
                errors.append(
                    f"{path.name}: synthetic-evidence label is missing."
                )

            if "Fictional demonstration evidence only" not in combined_text:
                errors.append(
                    f"{path.name}: fictional-data footer is missing."
                )

            for page_number, page in enumerate(document, start=1):
                if f"Page {page_number}" not in page.get_text("text"):
                    errors.append(
                        f"{path.name}: page number {page_number} is missing."
                    )
    except Exception as exc:
        errors.append(f"{path.name}: cannot be opened: {exc}")

    return errors


def validate_dataset(
    generated_files: list[Path],
) -> list:
    """Validate generated evidence before reporting success."""

    errors: list[str] = []

    required_ids = {
        "REQ-017",
        "REQ-021",
        "REQ-033",
        "REQ-041",
        "REQ-042",
        "REQ-050",
        "REQ-060",
        "OBS-023",
        "OBS-031",
        "OBS-044",
        "OBS-052",
        "OBS-061",
        "OBS-070",
        "TC-102",
        "TC-201",
        "TC-301",
        "TC-401",
    }

    combined_content: list[str] = []

    for path in generated_files:
        if not path.exists():
            errors.append(f"Missing generated file: {path}")
            continue

        if path.stat().st_size == 0:
            errors.append(f"Generated file is empty: {path}")
            continue

        if path.suffix.lower() == ".pdf":
            errors.extend(validate_generated_pdf(path))

            with pymupdf.open(path) as document:
                combined_content.extend(
                    page.get_text("text")
                    for page in document
                )
        elif path.suffix.lower() in {".c", ".json"}:
            combined_content.append(
                path.read_text(encoding="utf-8")
            )

    combined_text = "\n".join(combined_content)

    for engineering_id in sorted(required_ids):
        if engineering_id not in combined_text:
            errors.append(
                f"Required engineering ID not found: {engineering_id}"
            )

    return errors


def main() -> int:
    """Generate and validate the complete Phase 3 dataset."""

    print("=" * 72)
    print("REQTRACE AI - PHASE 3 SYNTHETIC DATA GENERATOR")
    print("=" * 72)
    print(SAFETY_NOTICE)
    print()

    ensure_directories()

    generated_files: list[Path] = []

    for specification in build_pdf_specifications():
        path = create_pdf(specification)
        generated_files.append(path)
        print(f"[CREATED] {path.relative_to(BASE_DIR)}")

    for specification in build_code_specifications():
        path = create_code_file(specification)
        generated_files.append(path)
        print(f"[CREATED] {path.relative_to(BASE_DIR)}")

    outcomes_path = write_json(
        DATA_DIR / "expected_outcomes.json",
        expected_outcomes(),
    )
    generated_files.append(outcomes_path)
    print(f"[CREATED] {outcomes_path.relative_to(BASE_DIR)}")

    catalogue = build_catalogue(generated_files)
    catalogue_path = write_json(
        DATA_DIR / "synthetic_dataset_catalogue.json",
        catalogue,
    )
    print(f"[CREATED] {catalogue_path.relative_to(BASE_DIR)}")

    validation_targets = generated_files + [catalogue_path]
    validation_errors = validate_dataset(validation_targets)

    print()
    print("=" * 72)
    print("GENERATION SUMMARY")
    print("=" * 72)
    print(f"Generated evidence files: {len(generated_files)}")
    print("ChromaDB collections created: 0")
    print("Embeddings generated: 0")
    print("Documents ingested: 0")

    if validation_errors:
        print()
        print("VALIDATION FAILED")

        for error in validation_errors:
            print(f"[FAIL] {error}")

        return 1

    print()
    print("VALIDATION PASSED")
    print("Phase 3 synthetic evidence was generated successfully.")
    print("The result remains fictional demonstration data only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
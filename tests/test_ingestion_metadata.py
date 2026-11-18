"""
Tests for dynamic engineering metadata extraction.
"""

from __future__ import annotations

from src.ingestion_metadata import (
    extract_engineering_ids,
    extract_result_status,
    infer_document_type,
)


def test_extract_engineering_ids() -> None:
    """Explicit IDs should normalize consistently."""

    (
        requirements,
        observations,
        test_cases,
    ) = extract_engineering_ids(
        (
            "REQ_017 is referenced by OBS 023. "
            "Verification uses TC-102."
        )
    )

    assert requirements == (
        "REQ-017",
    )

    assert observations == (
        "OBS-023",
    )

    assert test_cases == (
        "TC-102",
    )


def test_result_status() -> None:
    """Source-reported result should normalize."""

    assert (
        extract_result_status(
            "Result Status: FAILED"
        )
        == "FAIL"
    )

    assert (
        extract_result_status(
            "Result: NOT EXECUTED"
        )
        == "NOT EXECUTED"
    )


def test_no_result_is_not_fabricated() -> None:
    """Missing result metadata must remain missing."""

    assert (
        extract_result_status(
            "TC-102 checks null handling."
        )
        is None
    )


def test_code_document_detection() -> None:
    """Source extension should identify source code."""

    assert (
        infer_document_type(
            relative_path=(
                "code/data_processor.c"
            ),
            extension=".c",
        )
        == (
            "code",
            "Source Code",
        )
    )


def test_observation_document_detection() -> None:
    """Observation path should identify observation report."""

    assert (
        infer_document_type(
            relative_path=(
                "observations/"
                "Observation_Report_OBS-080.pdf"
            ),
            extension=".pdf",
        )
        == (
            "observation",
            "Observation Report",
        )
    )

def test_frd_with_test_reference_remains_frd() -> None:
    """
    A requirement document may legitimately reference test cases.

    TC references must not convert an FRD into a Test Report.
    """

    result = infer_document_type(
        relative_path="FRD.md",
        extension=".md",
        content_sample=(
            "REQ-075 requires null-input handling.\n"
            "Verification uses TC-501."
        ),
    )

    assert result == (
        "frd",
        "FRD",
    )

def test_test_report_with_requirement_reference_remains_test() -> None:
    """
    A test report may reference a requirement.

    Filename/category evidence should dominate cross-reference IDs.
    """

    result = infer_document_type(
        relative_path=(
            "tests/Test_Report_TC-501.pdf"
        ),
        extension=".pdf",
        content_sample=(
            "REQ-075 verification.\n"
            "TC-501 Result Status: FAIL."
        ),
    )

    assert result == (
        "test",
        "Test Report",
    )

def test_mixed_unknown_document_is_not_forced() -> None:
    """
    An unclassified document containing multiple unrelated
    engineering-ID types must not be assigned an invented category.
    """

    result = infer_document_type(
        relative_path="miscellaneous_notes.txt",
        extension=".txt",
        content_sample=(
            "Discussion references "
            "REQ-017, OBS-023 and TC-102."
        ),
    )

    assert result == (
        "other",
        "Other",
    )
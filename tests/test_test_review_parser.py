"""
Tests for Phase 7 test-report and review-record parsing.
"""

from __future__ import annotations

from pathlib import Path

from src.metadata import DocumentType
from src.test_review_parser import (
    parse_review_directory,
    parse_review_pdf,
    parse_test_directory,
    parse_test_pdf,
)


BASE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = BASE_DIR / "documents" / "tests"
REVIEW_DIR = BASE_DIR / "documents" / "reviews"

def get_test_report_path(test_case_id: str) -> Path:
    """Return one synthetic test-report path."""

    return TEST_DIR / f"Test_Report_{test_case_id}.pdf"

def get_test_report_path(test_case_id: str) -> Path:
    """Return one synthetic test-report path."""

    return TEST_DIR / f"Test_Report_{test_case_id}.pdf"


def review_path(file_name: str) -> Path:
    return REVIEW_DIR / file_name


def test_tc_102_fields() -> None:
    result = parse_test_pdf(
        get_test_report_path("TC-102")
    )

    test = result.test_report

    assert test.test_case_id == "TC-102"
    assert test.revision == "Rev01"
    assert test.requirement_id == "REQ-017"
    assert test.result_status == "FAIL"
    assert test.software_baseline == "SYNTH-BUILD-01"


def test_tc_201_pass_is_explicit() -> None:
    result = parse_test_pdf(
        get_test_report_path("TC-201")
    )

    assert result.test_report.result_status == "PASS"
    assert result.test_report.requirement_id == "REQ-021"


def test_tc_301_not_executed_is_preserved() -> None:
    result = parse_test_pdf(
        get_test_report_path("TC-301")
    )

    assert (
        result.test_report.result_status
        == "NOT EXECUTED"
    )

    assert result.test_report.actual_result is not None


def test_tc_401_original_baseline_is_preserved() -> None:
    result = parse_test_pdf(
        get_test_report_path("TC-401")
    )

    test = result.test_report

    assert test.result_status == "FAIL"
    assert test.requirement_id == "REQ-060"
    assert test.software_baseline == (
        "SYNTH-BUILD-04-ORIGINAL"
    )


def test_test_evidence_metadata() -> None:
    result = parse_test_pdf(
        get_test_report_path("TC-102")
    )

    metadata = result.evidence_records[0].metadata

    assert metadata.document_type == DocumentType.TEST
    assert metadata.test_case_id == "TC-102"
    assert metadata.requirement_id == "REQ-017"
    assert metadata.result_status == "FAIL"
    assert metadata.page_number >= 1
    assert len(metadata.content_hash) == 64


def test_all_test_reports_parse() -> None:
    results = parse_test_directory(
        TEST_DIR
    )

    assert len(results) == 4

    test_ids = {
        result.test_report.test_case_id
        for result in results
    }

    assert test_ids == {
        "TC-102",
        "TC-201",
        "TC-301",
        "TC-401",
    }


def test_obs_023_review_is_parsed() -> None:
    result = parse_review_pdf(
        review_path(
            "Review_Record_OBS-023.pdf"
        )
    )

    review = result.review_record

    assert review.review_id == "REV-OBS-023"
    assert review.observation_id == "OBS-023"
    assert review.review_status == "Evidence collected"
    assert review.evidence_summary is not None


def test_obs_044_missing_evidence_is_preserved() -> None:
    result = parse_review_pdf(
        review_path(
            "Review_Record_OBS-044.pdf"
        )
    )

    review = result.review_record

    assert review.observation_id == "OBS-044"
    assert review.missing_evidence is not None

    normalized = " ".join(
        review.missing_evidence.split()
    )

    assert "TC-301 was not executed" in normalized


def test_obs_052_traceability_action_is_preserved() -> None:
    result = parse_review_pdf(
        review_path(
            "Review_Record_OBS-052.pdf"
        )
    )

    review = result.review_record

    assert review.observation_id == "OBS-052"
    assert review.traceability_finding is not None
    assert review.required_action is not None

    normalized = " ".join(
        review.required_action.split()
    )

    assert "Semantic similarity alone" in normalized


def test_obs_061_blocking_item_is_preserved() -> None:
    result = parse_review_pdf(
        review_path(
            "Review_Record_OBS-061.pdf"
        )
    )

    review = result.review_record

    assert review.observation_id == "OBS-061"
    assert review.conflict is not None
    assert review.blocking_item is not None


def test_obs_070_closure_status_is_preserved() -> None:
    result = parse_review_pdf(
        review_path(
            "Closure_Review_OBS-070.pdf"
        )
    )

    review = result.review_record

    assert review.observation_id == "OBS-070"
    assert review.closure_status == (
        "CLOSURE NOT ESTABLISHED"
    )

    assert review.original_observation is not None
    assert review.missing_evidence is not None
    assert review.closure_determination is not None


def test_review_evidence_metadata() -> None:
    result = parse_review_pdf(
        review_path(
            "Review_Record_OBS-023.pdf"
        )
    )

    metadata = result.evidence_records[0].metadata

    assert metadata.document_type == DocumentType.REVIEW
    assert metadata.observation_id == "OBS-023"
    assert metadata.page_number >= 1
    assert metadata.evidence_id.startswith(
        "REVIEW::"
    )
    assert len(metadata.content_hash) == 64


def test_all_review_records_parse() -> None:
    results = parse_review_directory(
        REVIEW_DIR
    )

    assert len(results) == 6

    observation_ids = {
        result.review_record.observation_id
        for result in results
    }

    assert observation_ids == {
        "OBS-023",
        "OBS-031",
        "OBS-044",
        "OBS-052",
        "OBS-061",
        "OBS-070",
    }


def test_parsers_do_not_make_assessment() -> None:
    test_result = parse_test_pdf(
        get_test_report_path("TC-102")
    )

    review_result = parse_review_pdf(
        review_path(
            "Review_Record_OBS-023.pdf"
        )
    )

    test_metadata = (
        test_result.evidence_records[0]
        .metadata
        .model_dump()
    )

    review_metadata = (
        review_result.evidence_records[0]
        .metadata
        .model_dump()
    )

    for metadata in (
        test_metadata,
        review_metadata,
    ):
        assert "assessment" not in metadata
        assert "confidence" not in metadata
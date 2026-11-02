"""
Tests for REQTRACE AI Phase 3 synthetic engineering evidence.

The tests validate document availability, searchable PDF text, engineering
identifiers, source-code references, expected outcomes, and closure
separation.

The tests do not create embeddings or ChromaDB collections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = BASE_DIR / "documents"
FRD_DIR = DOCUMENTS_DIR / "frd"
OBSERVATIONS_DIR = DOCUMENTS_DIR / "observations"
CODE_DIR = DOCUMENTS_DIR / "code"
TEST_REPORTS_DIR = DOCUMENTS_DIR / "tests"
REVIEWS_DIR = DOCUMENTS_DIR / "reviews"
DATA_DIR = BASE_DIR / "data"


EXPECTED_FILES = (
    FRD_DIR / "FRD_Input_Validation_Rev01.pdf",
    FRD_DIR / "FRD_Command_Processing_Rev01.pdf",
    FRD_DIR / "FRD_Mode_Control_Rev01.pdf",
    FRD_DIR / "FRD_Alert_Handling_Rev01.pdf",
    FRD_DIR / "FRD_Alert_Handling_Rev02.pdf",
    FRD_DIR / "FRD_Recovery_Control_Rev01.pdf",
    OBSERVATIONS_DIR / "Observation_Report_OBS-023.pdf",
    OBSERVATIONS_DIR / "Observation_Report_OBS-031.pdf",
    OBSERVATIONS_DIR / "Observation_Report_OBS-044.pdf",
    OBSERVATIONS_DIR / "Observation_Report_OBS-052.pdf",
    OBSERVATIONS_DIR / "Observation_Report_OBS-061.pdf",
    OBSERVATIONS_DIR / "Observation_Report_OBS-070.pdf",
    CODE_DIR / "data_processor.c",
    CODE_DIR / "command_validator.c",
    CODE_DIR / "mode_controller.c",
    CODE_DIR / "alert_manager.c",
    CODE_DIR / "recovery_manager.c",
    TEST_REPORTS_DIR / "Test_Report_TC-102.pdf",
    TEST_REPORTS_DIR / "Test_Report_TC-201.pdf",
    TEST_REPORTS_DIR / "Test_Report_TC-301.pdf",
    TEST_REPORTS_DIR / "Test_Report_TC-401.pdf",
    REVIEWS_DIR / "Review_Record_OBS-023.pdf",
    REVIEWS_DIR / "Review_Record_OBS-031.pdf",
    REVIEWS_DIR / "Review_Record_OBS-044.pdf",
    REVIEWS_DIR / "Review_Record_OBS-052.pdf",
    REVIEWS_DIR / "Review_Record_OBS-061.pdf",
    REVIEWS_DIR / "Closure_Review_OBS-070.pdf",
    DATA_DIR / "expected_outcomes.json",
    DATA_DIR / "synthetic_dataset_catalogue.json",
)


def extract_pdf_text(path: Path) -> str:
    """Extract all searchable text from a PDF."""

    with pymupdf.open(path) as document:
        return "\n".join(
            page.get_text("text")
            for page in document
        )


def load_expected_outcomes() -> list[dict[str, object]]:
    """Load the generated expected-outcome definitions."""

    path = DATA_DIR / "expected_outcomes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_expected_files_exist_and_are_non_empty() -> None:
    for path in EXPECTED_FILES:
        assert path.exists(), f"Missing file: {path}"
        assert path.stat().st_size > 0, f"Empty file: {path}"


def test_all_generated_pdfs_are_readable_and_searchable() -> None:
    pdf_paths = [
        path
        for path in EXPECTED_FILES
        if path.suffix.lower() == ".pdf"
    ]

    for path in pdf_paths:
        with pymupdf.open(path) as document:
            assert len(document) >= 1

            extracted_text = "\n".join(
                page.get_text("text")
                for page in document
            )

            assert extracted_text.strip()
            assert "REQTRACE AI SYNTHETIC EVIDENCE" in extracted_text
            assert "Fictional demonstration evidence only" in extracted_text


def test_pdf_page_numbers_are_one_based_and_visible() -> None:
    pdf_paths = [
        path
        for path in EXPECTED_FILES
        if path.suffix.lower() == ".pdf"
    ]

    for path in pdf_paths:
        with pymupdf.open(path) as document:
            for index, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                assert f"Page {index}" in page_text


def test_supported_case_contains_complete_evidence_chain() -> None:
    frd_text = extract_pdf_text(
        FRD_DIR / "FRD_Input_Validation_Rev01.pdf"
    )
    observation_text = extract_pdf_text(
        OBSERVATIONS_DIR / "Observation_Report_OBS-023.pdf"
    )
    test_text = extract_pdf_text(
        TEST_REPORTS_DIR / "Test_Report_TC-102.pdf"
    )
    code_text = (
        CODE_DIR / "data_processor.c"
    ).read_text(encoding="utf-8")

    assert "REQ-017" in frd_text
    assert "OBS-023" in observation_text
    assert "TC-102" in test_text
    assert "Result Status: FAIL" in test_text
    assert "process_data" in code_text
    assert "input->sample_count" in code_text


def test_not_supported_case_contains_contradicting_evidence() -> None:
    observation_text = extract_pdf_text(
        OBSERVATIONS_DIR / "Observation_Report_OBS-031.pdf"
    )
    test_text = extract_pdf_text(
        TEST_REPORTS_DIR / "Test_Report_TC-201.pdf"
    )
    code_text = (
        CODE_DIR / "command_validator.c"
    ).read_text(encoding="utf-8")

    assert "OBS-031" in observation_text
    assert "identifier 17 was accepted" in observation_text
    assert "Result Status: PASS" in test_text
    normalized = " ".join(test_text.split())

    assert "Identifier 17 was rejected" in normalized       
    assert "command_id > COMMAND_ID_MAXIMUM" in code_text


def test_inconclusive_case_preserves_missing_evidence() -> None:
    observation_text = extract_pdf_text(
        OBSERVATIONS_DIR / "Observation_Report_OBS-044.pdf"
    )
    test_text = extract_pdf_text(
        TEST_REPORTS_DIR / "Test_Report_TC-301.pdf"
    )
    code_text = (
        CODE_DIR / "mode_controller.c"
    ).read_text(encoding="utf-8")

    assert "final mode-assignment logic is not included" in observation_text
    assert "Result Status: NOT EXECUTED" in test_text
    assert "final mode assignment are not" in code_text


def test_incorrect_requirement_reference_is_not_silently_replaced() -> None:
    observation_text = extract_pdf_text(
        OBSERVATIONS_DIR / "Observation_Report_OBS-052.pdf"
    )

    assert "REQ-041" in observation_text
    assert "REQ-042" in observation_text
    normalized = " ".join(observation_text.split())

    assert "must not silently replace" in normalized


def test_revision_conflict_contains_both_requirement_versions() -> None:
    rev01_text = extract_pdf_text(
        FRD_DIR / "FRD_Alert_Handling_Rev01.pdf"
    )
    rev02_text = extract_pdf_text(
        FRD_DIR / "FRD_Alert_Handling_Rev02.pdf"
    )
    observation_text = extract_pdf_text(
        OBSERVATIONS_DIR / "Observation_Report_OBS-061.pdf"
    )

    assert "REQ-050" in rev01_text
    assert "500 milliseconds" in rev01_text
    assert "REQ-050" in rev02_text
    assert "1000 milliseconds" in rev02_text
    assert "760 milliseconds" in observation_text
    assert "Applicable FRD Revision: Not specified" in observation_text


def test_observation_validity_is_separate_from_closure_readiness() -> None:
    closure_text = extract_pdf_text(
        REVIEWS_DIR / "Closure_Review_OBS-070.pdf"
    )

    assert "The original observation is supported" in closure_text
    assert "CLOSURE NOT ESTABLISHED" in closure_text
    assert "Corrected source code: Not available" in closure_text
    assert "Post-correction test result: Not available" in closure_text


def test_expected_outcomes_cover_required_scenarios() -> None:
    outcomes = load_expected_outcomes()

    assert len(outcomes) == 6

    assessments = {
        item["expected_assessment"]
        for item in outcomes
    }
    scenarios = {
        item["scenario"]
        for item in outcomes
    }

    assert assessments == {
        "SUPPORTED",
        "NOT SUPPORTED",
        "INCONCLUSIVE",
    }

    assert "supported" in scenarios
    assert "not_supported" in scenarios
    assert "inconclusive_missing_evidence" in scenarios
    assert "incorrect_requirement_reference" in scenarios
    assert "frd_revision_conflict" in scenarios
    assert "supported_but_not_ready_for_closure" in scenarios


def test_obs_070_has_supported_assessment_but_failed_closure() -> None:
    outcomes = load_expected_outcomes()

    obs_070 = next(
        item
        for item in outcomes
        if item["observation_id"] == "OBS-070"
    )

    assert obs_070["expected_assessment"] == "SUPPORTED"
    assert (
        obs_070["expected_closure_status"]
        == "CLOSURE NOT ESTABLISHED"
    )
    assert obs_070["expected_blocking_items"]


def test_dataset_catalogue_contains_hashes() -> None:
    catalogue_path = DATA_DIR / "synthetic_dataset_catalogue.json"
    catalogue = json.loads(
        catalogue_path.read_text(encoding="utf-8")
    )

    assert catalogue["dataset_name"]
    assert catalogue["files"]

    for record in catalogue["files"]:
        assert record["relative_path"]
        assert record["file_name"]
        assert record["document_type"]
        assert record["size_bytes"] > 0
        assert len(record["sha256"]) == 64


def test_source_code_uses_one_based_physical_lines() -> None:
    path = CODE_DIR / "data_processor.c"
    lines = path.read_text(encoding="utf-8").splitlines()

    function_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "data_status_t process_data" in line
    )

    dereference_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "count = input->sample_count;" in line
    )

    assert function_line >= 1
    assert dereference_line > function_line
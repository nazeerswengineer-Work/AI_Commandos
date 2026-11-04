"""
Tests for Phase 5*PDF loading and requirement-aware FRD parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hashlib

def directory_fingerprint(
    directory: Path,
) -> dict[str, str]:
    """
    Calculate hashes for every file under a directory.

    Used to verify that parsing does not modify persistent
    vector-store data.
    """

    if not directory.exists():
        return {}

    fingerprints: dict[str, str] = {}

    for path in sorted(
        directory.rglob("*")
    ):
        if not path.is_file():
            continue

        digest = hashlib.sha256()

        with path.open("rb") as file_handle:
            for block in iter(
                lambda: file_handle.read(65536),
                b"",
            ):
                digest.update(block)

        relative_path = (
            path
            .relative_to(directory)
            .as_posix()
        )

        fingerprints[
            relative_path
        ] = digest.hexdigest()

    return fingerprints

from src.document_parser import (
    extract_document_revision,
    find_requirement_occurrences,
    parse_frd_directory,
    parse_frd_pdf,
    remove_synthetic_page_noise,
)
from src.loaders import (
    UnsupportedPDFError,
    load_pdf,
)
from src.metadata import (
    DocumentType,
    ParsingMethod,
)
from src.normalization import extract_requirement_ids


BASE_DIR = Path(__file__).resolve().parents[1]
FRD_DIR = BASE_DIR / "documents" / "frd"


def get_frd_path(file_name: str) -> Path:
    """Return one Phase 3 synthetic FRD path."""

    return FRD_DIR / file_name


def test_load_pdf_preserves_document_name() -> None:
    path = get_frd_path(
        "FRD_Input_Validation_Rev01.pdf"
    )

    loaded_pdf = load_pdf(path)

    assert loaded_pdf.document_name == path.name
    assert loaded_pdf.file_path == path.resolve()


def test_load_pdf_preserves_one_based_page_numbers() -> None:
    path = get_frd_path(
        "FRD_Input_Validation_Rev01.pdf"
    )

    loaded_pdf = load_pdf(path)

    assert loaded_pdf.page_count >= 1
    assert loaded_pdf.pages[0].page_number == 1
    assert tuple(
        page.page_number for page in loaded_pdf.pages
    ) == tuple(range(1, loaded_pdf.page_count + 1))


def test_load_pdf_extracts_searchable_text() -> None:
    path = get_frd_path(
        "FRD_Input_Validation_Rev01.pdf"
    )

    loaded_pdf = load_pdf(path)

    combined_text = "\n".join(
        page.text for page in loaded_pdf.pages
    )

    assert "REQ-017" in combined_text
    assert "null" in combined_text.lower()


def test_non_pdf_file_is_rejected(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "not_a_pdf.txt"
    text_path.write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedPDFError):
        load_pdf(text_path)


def test_missing_pdf_is_rejected() -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(FRD_DIR / "missing.pdf")


def test_synthetic_noise_is_removed() -> None:
    source_text = """
    REQTRACE AI SYNTHETIC EVIDENCE
    2. Requirement REQ-017
    Requirement ID: REQ-017
    Required behavior.
    Fictional demonstration evidence only
    Page 1
    """

    cleaned = remove_synthetic_page_noise(source_text)

    assert "REQTRACE AI SYNTHETIC EVIDENCE" not in cleaned
    assert "Fictional demonstration evidence only" not in cleaned
    assert "Page 1" not in cleaned
    assert "REQ-017" in cleaned
    assert "Required behavior." in cleaned


def test_explicit_requirement_heading_detection() -> None:
    text = """
    1. Purpose
    This section mentions REQ-999 as a reference.

    2. Requirement REQ_017
    Requirement ID: REQ 017
    The system shall validate the input.

    3. Requirement REQ-021
    The system shall reject invalid commands.
    """

    occurrences = find_requirement_occurrences(text)

    assert tuple(
        item.requirement_id for item in occurrences
    ) == (
        "REQ-017",
        "REQ-021",
    )


def test_ordinary_reference_does_not_create_heading() -> None:
    text = """
    This paragraph references REQ-017.
    No explicit requirement heading is present.
    """

    occurrences = find_requirement_occurrences(text)

    assert occurrences == ()
    assert extract_requirement_ids(text) == ("REQ-017",)


def test_revision_is_extracted_from_pdf_text() -> None:
    path = get_frd_path(
        "FRD_Input_Validation_Rev01.pdf"
    )

    loaded_pdf = load_pdf(path)
    revision = extract_document_revision(loaded_pdf)

    assert revision == "Rev01"


def test_parse_req_017_document() -> None:
    path = get_frd_path(
        "FRD_Input_Validation_Rev01.pdf"
    )

    result = parse_frd_pdf(path)

    assert result.document_name == path.name
    assert result.document_revision == "Rev01"
    assert result.requirement_ids == ("REQ-017",)
    assert result.requirement_count == 1

    evidence = result.evidence_records[0]

    assert evidence.metadata.document_type == DocumentType.FRD
    assert evidence.metadata.requirement_id == "REQ-017"
    assert evidence.metadata.page_number >= 1
    assert evidence.metadata.document_revision == "Rev01"
    assert evidence.metadata.parsing_method == (
        ParsingMethod.REQUIREMENT
    )
    assert "REQ-017" in evidence.content
    assert "null" in evidence.content.lower()


def test_req_017_evidence_id_is_stable() -> None:
    path = get_frd_path(
        "FRD_Input_Validation_Rev01.pdf"
    )

    first_result = parse_frd_pdf(path)
    second_result = parse_frd_pdf(path)

    first_evidence = first_result.evidence_records[0]
    second_evidence = second_result.evidence_records[0]

    assert (
        first_evidence.metadata.evidence_id
        == second_evidence.metadata.evidence_id
    )

    assert (
        first_evidence.metadata.content_hash
        == second_evidence.metadata.content_hash
    )

    assert first_evidence.metadata.evidence_id.startswith(
        "FRD::FRD_Input_Validation_Rev01.pdf::PAGE-"
    )

    assert first_evidence.metadata.evidence_id.endswith(
        "::REQ-017"
    )


def test_alert_revisions_remain_separate() -> None:
    rev01_path = get_frd_path(
        "FRD_Alert_Handling_Rev01.pdf"
    )
    rev02_path = get_frd_path(
        "FRD_Alert_Handling_Rev02.pdf"
    )

    rev01 = parse_frd_pdf(rev01_path)
    rev02 = parse_frd_pdf(rev02_path)

    assert rev01.document_revision == "Rev01"
    assert rev02.document_revision == "Rev02"

    assert "REQ-050" in rev01.requirement_ids
    assert "REQ-050" in rev02.requirement_ids

    rev01_req_050 = next(
        evidence
        for evidence in rev01.evidence_records
        if evidence.metadata.requirement_id == "REQ-050"
    )

    rev02_req_050 = next(
        evidence
        for evidence in rev02.evidence_records
        if evidence.metadata.requirement_id == "REQ-050"
    )

    assert "500 milliseconds" in (
        " ".join(rev01_req_050.content.split())
    )

    assert "1000 milliseconds" in (
        " ".join(rev02_req_050.content.split())
    )

    assert (
        rev01_req_050.metadata.evidence_id
        != rev02_req_050.metadata.evidence_id
    )


def test_unrelated_requirements_are_not_merged() -> None:
    path = get_frd_path(
        "FRD_Alert_Handling_Rev02.pdf"
    )

    result = parse_frd_pdf(path)

    req_042 = next(
        evidence
        for evidence in result.evidence_records
        if evidence.metadata.requirement_id == "REQ-042"
    )

    req_050 = next(
        evidence
        for evidence in result.evidence_records
        if evidence.metadata.requirement_id == "REQ-050"
    )

    assert "REQ-042" in req_042.content
    assert "REQ-050" not in req_042.content

    assert "REQ-050" in req_050.content
    assert "1000 milliseconds" in (
        " ".join(req_050.content.split())
    )


def test_all_synthetic_frd_documents_parse() -> None:
    results = parse_frd_directory(FRD_DIR)

    assert len(results) == 6

    document_names = {
        result.document_name for result in results
    }

    assert "FRD_Input_Validation_Rev01.pdf" in document_names
    assert "FRD_Command_Processing_Rev01.pdf" in document_names
    assert "FRD_Mode_Control_Rev01.pdf" in document_names
    assert "FRD_Alert_Handling_Rev01.pdf" in document_names
    assert "FRD_Alert_Handling_Rev02.pdf" in document_names
    assert "FRD_Recovery_Control_Rev01.pdf" in document_names

    for result in results:
        assert result.requirement_count >= 1
        assert result.document_revision is not None


def test_every_evidence_record_has_valid_source_metadata() -> None:
    results = parse_frd_directory(FRD_DIR)

    for result in results:
        for evidence in result.evidence_records:
            metadata = evidence.metadata

            assert metadata.evidence_id
            assert metadata.document_type == DocumentType.FRD
            assert metadata.document_name == result.document_name
            assert metadata.document_revision
            assert metadata.page_number is not None
            assert metadata.page_number >= 1
            assert metadata.requirement_id
            assert metadata.chunk_id
            assert len(metadata.content_hash) == 64
            assert metadata.file_path
            assert evidence.content.strip()


def test_all_expected_requirement_ids_are_parsed() -> None:
    results = parse_frd_directory(FRD_DIR)

    parsed_ids = {
        requirement_id
        for result in results
        for requirement_id in result.requirement_ids
    }

    assert parsed_ids == {
        "REQ-017",
        "REQ-021",
        "REQ-033",
        "REQ-041",
        "REQ-042",
        "REQ-050",
        "REQ-060",
    }


def test_parser_does_not_modify_vector_data() -> None:
    chroma_directory = BASE_DIR / "chroma_db"

    before = directory_fingerprint(
        chroma_directory
    )

    parse_frd_directory(
        FRD_DIR
    )

    after = directory_fingerprint(
        chroma_directory
    )

    assert after == before
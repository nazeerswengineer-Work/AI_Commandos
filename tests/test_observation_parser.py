"""
Tests for Phase 6 observation PDF parsing.
"""

from __future__ import annotations

from pathlib import Path

from src.metadata import (
    DocumentType,
    ParsingMethod,
)
from src.observation_parser import (
    parse_observation_directory,
    parse_observation_pdf,
)


BASE_DIR = Path(__file__).resolve().parents[1]

OBSERVATION_DIR = (
    BASE_DIR
    / "documents"
    / "observations"
)


def observation_path(
    observation_id: str,
) -> Path:
    return (
        OBSERVATION_DIR
        / f"Observation_Report_{observation_id}.pdf"
    )


def test_obs_023_basic_fields() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-023")
    )

    obs = result.observation

    assert obs.observation_id == "OBS-023"
    assert obs.revision == "Rev01"
    assert obs.status == "OPEN"
    assert obs.severity == "Major"
    assert obs.requirement_id == "REQ-017"
    assert obs.test_case_id == "TC-102"
    assert obs.referenced_file == "data_processor.c"
    assert obs.referenced_function == "process_data"


def test_obs_023_description_is_extracted() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-023")
    )

    obs = result.observation

    assert obs.description is not None

    normalized = " ".join(
        obs.description.split()
    )

    assert "process_data()" in normalized
    assert "null input pointer" in normalized


def test_obs_023_expected_and_actual_results() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-023")
    )

    obs = result.observation

    assert obs.expected_result is not None
    assert obs.actual_result is not None

    expected = " ".join(
        obs.expected_result.split()
    )

    actual = " ".join(
        obs.actual_result.split()
    )

    assert "DATA_STATUS_INVALID_INPUT" in expected
    assert "dereferenced" in actual


def test_observation_metadata_is_valid() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-023")
    )

    evidence = result.evidence_records[0]
    metadata = evidence.metadata

    assert metadata.document_type == (
        DocumentType.OBSERVATION
    )

    assert metadata.parsing_method == (
        ParsingMethod.OBSERVATION
    )

    assert metadata.observation_id == "OBS-023"
    assert metadata.requirement_id == "REQ-017"
    assert metadata.test_case_id == "TC-102"
    assert metadata.page_number >= 1
    assert len(metadata.content_hash) == 64


def test_obs_023_evidence_id_is_stable() -> None:
    first = parse_observation_pdf(
        observation_path("OBS-023")
    )

    second = parse_observation_pdf(
        observation_path("OBS-023")
    )

    first_metadata = (
        first.evidence_records[0].metadata
    )

    second_metadata = (
        second.evidence_records[0].metadata
    )

    assert (
        first_metadata.evidence_id
        == second_metadata.evidence_id
    )

    assert (
        first_metadata.content_hash
        == second_metadata.content_hash
    )

    assert first_metadata.evidence_id == (
        "OBS::Observation_Report_OBS-023.pdf"
        "::PAGE-1::OBS-023"
    )


def test_obs_031_preserves_explicit_reference() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-031")
    )

    obs = result.observation

    assert obs.observation_id == "OBS-031"
    assert obs.requirement_id == "REQ-021"
    assert obs.test_case_id == "TC-201"
    assert obs.referenced_file == (
        "command_validator.c"
    )
    assert obs.referenced_function == (
        "validate_command"
    )


def test_obs_044_preserves_evidence_limitation() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-044")
    )

    obs = result.observation

    assert obs.observation_id == "OBS-044"
    assert obs.requirement_id == "REQ-033"
    assert obs.test_case_id == "TC-301"

    assert obs.evidence_limitation is not None
    assert obs.assessment_constraint is not None

    limitation = " ".join(
        obs.evidence_limitation.split()
    ).lower()

    constraint = " ".join(
        obs.assessment_constraint.split()
    ).lower()

    assert "not included" in limitation
    assert "cannot be confirmed" in constraint


def test_obs_052_does_not_replace_requirement_reference() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-052")
    )

    obs = result.observation

    assert obs.observation_id == "OBS-052"

    # This is deliberately REQ-041.
    # Phase 6 must not silently change it to REQ-042.
    assert obs.requirement_id == "REQ-041"

    assert obs.test_case_id is None


def test_obs_052_traceability_concern_is_preserved() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-052")
    )

    constraint = (
        result.observation.assessment_constraint
    )

    assert constraint is not None

    normalized = " ".join(
        constraint.split()
    )

    assert "must not silently replace" in normalized


def test_obs_061_revision_conflict_is_preserved() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-061")
    )

    obs = result.observation

    assert obs.observation_id == "OBS-061"
    assert obs.requirement_id == "REQ-050"
    assert obs.test_case_id is None

    assert obs.description is not None
    assert obs.revision_conflict is not None
    assert obs.assessment_constraint is not None

    description = " ".join(
        obs.description.split()
    )

    conflict = " ".join(
        obs.revision_conflict.split()
    )

    constraint = " ".join(
        obs.assessment_constraint.split()
    )

    # Observed behaviour
    assert "760 milliseconds" in description

    # Conflicting requirement revisions
    assert "500 milliseconds" in conflict
    assert "1000 milliseconds" in conflict

    # Reason a deterministic conclusion cannot yet be made
    assert "applicable FRD revision" in constraint

    assert (
        "applicable FRD revision"
        in constraint
    )


def test_obs_070_closure_limitation_is_preserved() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-070")
    )

    obs = result.observation

    assert obs.observation_id == "OBS-070"
    assert obs.requirement_id == "REQ-060"
    assert obs.test_case_id == "TC-401"

    assert obs.closure_limitation is not None

    normalized = " ".join(
        obs.closure_limitation.split()
    )

    assert "corrected code" in normalized
    assert (
        "post-correction verification result"
        in normalized
    )


def test_all_observation_documents_parse() -> None:
    results = parse_observation_directory(
        OBSERVATION_DIR
    )

    assert len(results) == 6

    observation_ids = {
        result.observation_id
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


def test_all_observations_have_one_evidence_record() -> None:
    results = parse_observation_directory(
        OBSERVATION_DIR
    )

    for result in results:
        assert result.evidence_count == 1

        evidence = result.evidence_records[0]

        assert evidence.content.strip()
        assert evidence.metadata.observation_id
        assert evidence.metadata.page_number >= 1
        assert evidence.metadata.evidence_id
        assert len(
            evidence.metadata.content_hash
        ) == 64


def test_missing_test_references_generate_warning() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-052")
    )

    warning_codes = {
        warning.code
        for warning in result.warnings
    }

    assert "MISSING_TEST_REFERENCE" in warning_codes


def test_parser_does_not_make_assessment() -> None:
    result = parse_observation_pdf(
        observation_path("OBS-023")
    )

    evidence = result.evidence_records[0]

    metadata_dictionary = (
        evidence.metadata.model_dump()
    )

    assert "assessment" not in metadata_dictionary
    assert "confidence" not in metadata_dictionary
    assert "closure_status" not in metadata_dictionary
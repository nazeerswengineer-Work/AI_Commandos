"""
Tests for Phase 7 source-code parsing.
"""

from __future__ import annotations

from pathlib import Path

from src.code_parser import (
    parse_code_directory,
    parse_code_file,
)
from src.metadata import (
    DocumentType,
    ParsingMethod,
)


BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "documents" / "code"


def code_path(file_name: str) -> Path:
    return CODE_DIR / file_name


def test_data_processor_function_is_parsed() -> None:
    result = parse_code_file(
        code_path("data_processor.c")
    )

    assert result.function_names == (
        "process_data",
    )

    function = result.functions[0]

    assert function.start_line >= 1
    assert function.end_line > function.start_line
    assert "input->sample_count" in function.content


def test_data_processor_traceability_is_preserved() -> None:
    result = parse_code_file(
        code_path("data_processor.c")
    )

    function = result.functions[0]

    assert "REQ-017" in function.requirement_ids
    assert "OBS-023" in function.observation_ids
    assert "TC-102" in function.test_case_ids


def test_code_evidence_metadata() -> None:
    result = parse_code_file(
        code_path("data_processor.c")
    )

    evidence = result.evidence_records[0]
    metadata = evidence.metadata

    assert metadata.document_type == DocumentType.CODE
    assert metadata.parsing_method == ParsingMethod.FUNCTION
    assert metadata.file_name == "data_processor.c"
    assert metadata.function_name == "process_data"
    assert metadata.start_line >= 1
    assert metadata.end_line >= metadata.start_line
    assert metadata.page_number is None
    assert metadata.requirement_id == "REQ-017"
    assert metadata.observation_id == "OBS-023"
    assert metadata.test_case_id == "TC-102"


def test_code_evidence_id_is_stable() -> None:
    first = parse_code_file(
        code_path("data_processor.c")
    )

    second = parse_code_file(
        code_path("data_processor.c")
    )

    first_evidence = first.evidence_records[0]
    second_evidence = second.evidence_records[0]

    assert (
        first_evidence.metadata.evidence_id
        == second_evidence.metadata.evidence_id
    )

    assert (
        first_evidence.metadata.content_hash
        == second_evidence.metadata.content_hash
    )

    assert first_evidence.metadata.evidence_id.startswith(
        "CODE::data_processor.c::process_data::"
    )


def test_command_validator_is_parsed() -> None:
    result = parse_code_file(
        code_path("command_validator.c")
    )

    assert result.function_names == (
        "validate_command",
    )

    function = result.functions[0]

    assert "REQ-021" in function.requirement_ids
    assert "OBS-031" in function.observation_ids
    assert "TC-201" in function.test_case_ids


def test_mode_controller_partial_evidence_is_preserved() -> None:
    result = parse_code_file(
        code_path("mode_controller.c")
    )

    function = result.functions[0]

    assert function.function_name == "request_active_mode"
    assert function.is_partial_evidence is True
    assert function.limitations


def test_alert_manager_has_two_functions() -> None:
    result = parse_code_file(
        code_path("alert_manager.c")
    )

    assert set(result.function_names) == {
        "process_alert",
        "reset_alert",
    }


def test_recovery_manager_preserves_closure_limitation() -> None:
    result = parse_code_file(
        code_path("recovery_manager.c")
    )

    assert result.function_names == (
        "start_recovery",
    )

    function = result.functions[0]

    assert function.function_name == "start_recovery"
    assert function.is_partial_evidence is True
    assert function.limitations

    normalized_limitations = " ".join(
        function.limitations
    ).lower()

    assert (
        "no corrected implementation"
        in normalized_limitations
    )

    # Function content must retain the actual source function.
    normalized_content = " ".join(
        function.content.split()
    ).lower()

    assert "void start_recovery" in normalized_content
    assert "recovery_counter is not initialized" in normalized_content


def test_all_synthetic_code_files_parse() -> None:
    results = parse_code_directory(
        CODE_DIR
    )

    assert len(results) == 5

    file_names = {
        result.file_name
        for result in results
    }

    assert file_names == {
        "alert_manager.c",
        "command_validator.c",
        "data_processor.c",
        "mode_controller.c",
        "recovery_manager.c",
    }

    for result in results:
        assert result.evidence_count >= 1
        assert result.physical_line_count >= 1


def test_code_parser_does_not_create_assessment() -> None:
    result = parse_code_file(
        code_path("data_processor.c")
    )

    metadata = (
        result.evidence_records[0]
        .metadata
        .model_dump()
    )

    assert "assessment" not in metadata
    assert "confidence" not in metadata
    assert "closure_status" not in metadata
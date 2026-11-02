"""
Tests for Phase 4 normalization, metadata, evidence IDs, hashes, and schemas.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from src.evidence import (
    EvidenceRecord,
    build_code_evidence_id,
    build_frd_evidence_id,
    build_observation_evidence_id,
    build_test_evidence_id,
    calculate_content_hash,
)
from src.metadata import (
    DocumentType,
    EvidenceMetadata,
    ParsingMethod,
)
from src.normalization import (
    EngineeringIDError,
    extract_observation_ids,
    extract_requirement_ids,
    extract_test_case_ids,
    normalize_engineering_id,
    normalize_multiline_content,
    normalize_observation_id,
    normalize_requirement_id,
    normalize_revision,
    normalize_test_case_id,
)
from src.schemas import (
    AssessmentValue,
    ClosureDecision,
    ClosureStatus,
    ConfidenceLevel,
    DeterministicDecision,
    EngineeringAssessment,
    EvidenceCompleteness,
    FRDRequirementSummary,
    GroundingStatus,
    ObservationSummary,
    CodeAnalysis,
    EvidenceComparison,
)


def test_requirement_id_normalization() -> None:
    assert normalize_requirement_id("REQ-017") == "REQ-017"
    assert normalize_requirement_id("req_017") == "REQ-017"
    assert normalize_requirement_id("REQ 17") == "REQ-017"
    assert normalize_requirement_id("req00017") == "REQ-017"


def test_observation_id_normalization() -> None:
    assert normalize_observation_id("OBS-023") == "OBS-023"
    assert normalize_observation_id("obs_023") == "OBS-023"
    assert normalize_observation_id("OBS 23") == "OBS-023"


def test_test_case_id_normalization() -> None:
    assert normalize_test_case_id("TC-102") == "TC-102"
    assert normalize_test_case_id("tc_102") == "TC-102"
    assert normalize_test_case_id("TC 102") == "TC-102"


def test_generic_engineering_id_normalization() -> None:
    assert normalize_engineering_id("req 17") == "REQ-017"
    assert normalize_engineering_id("obs_23") == "OBS-023"
    assert normalize_engineering_id("tc-102") == "TC-102"


def test_invalid_engineering_id_is_rejected() -> None:
    with pytest.raises(EngineeringIDError):
        normalize_requirement_id("OBS-017")

    with pytest.raises(EngineeringIDError):
        normalize_observation_id("observation 23")

    with pytest.raises(EngineeringIDError):
        normalize_test_case_id("TEST-102")


def test_revision_normalization() -> None:
    assert normalize_revision("1") == "Rev01"
    assert normalize_revision("Rev1") == "Rev01"
    assert normalize_revision("REV_02") == "Rev02"
    assert normalize_revision(None) is None


def test_id_extraction_is_ordered_and_duplicate_free() -> None:
    text = """
    OBS_023 references REQ 017 and TC-102.
    REQ-017 is repeated.
    OBS-023 is repeated.
    Another candidate is REQ_021.
    """

    assert extract_observation_ids(text) == ("OBS-023",)
    assert extract_requirement_ids(text) == (
        "REQ-017",
        "REQ-021",
    )
    assert extract_test_case_ids(text) == ("TC-102",)


def test_multiline_normalization_is_stable() -> None:
    first = "\r\n  REQ-017  \r\nExpected result.   \r\n"
    second = "  REQ-017\nExpected result."

    assert normalize_multiline_content(first) == (
        normalize_multiline_content(second)
    )


def test_content_hash_is_stable_for_line_endings() -> None:
    first = "REQ-017\r\nExpected behavior.\r\n"
    second = "REQ-017\nExpected behavior."

    assert calculate_content_hash(first) == calculate_content_hash(second)


def test_content_hash_uses_sha256() -> None:
    content = "REQ-017\nExpected behavior."
    normalized = normalize_multiline_content(content)

    expected = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    assert calculate_content_hash(content) == expected


def test_empty_content_hash_is_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_content_hash(" \r\n ")


def test_frd_evidence_id() -> None:
    evidence_id = build_frd_evidence_id(
        document_name="FRD_System_Rev01.pdf",
        page_number=18,
        requirement_id="REQ-017",
    )

    assert evidence_id == (
        "FRD::FRD_System_Rev01.pdf::PAGE-18::REQ-017"
    )


def test_observation_evidence_id() -> None:
    evidence_id = build_observation_evidence_id(
        document_name="Observation_Report_023.pdf",
        page_number=4,
        observation_id="OBS-023",
    )

    assert evidence_id == (
        "OBS::Observation_Report_023.pdf::PAGE-4::OBS-023"
    )


def test_test_evidence_id() -> None:
    evidence_id = build_test_evidence_id(
        document_name="Test_Report_TC-102.pdf",
        page_number=3,
        test_case_id="TC-102",
    )

    assert evidence_id == (
        "TEST::Test_Report_TC-102.pdf::PAGE-3::TC-102"
    )


def test_code_evidence_id() -> None:
    evidence_id = build_code_evidence_id(
        file_name="data_processor.c",
        function_name="process_data",
        start_line=45,
        end_line=62,
    )

    assert evidence_id == (
        "CODE::data_processor.c::process_data::45-62"
    )


def test_invalid_page_number_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_frd_evidence_id(
            document_name="FRD.pdf",
            page_number=0,
            requirement_id="REQ-017",
        )


def test_invalid_code_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_code_evidence_id(
            file_name="file.c",
            function_name="function_name",
            start_line=20,
            end_line=10,
        )


def test_frd_metadata_normalizes_engineering_fields() -> None:
    content = "Requirement REQ-017 expected behavior."

    metadata = EvidenceMetadata(
        evidence_id=(
            "FRD::FRD_System_Rev01.pdf::PAGE-18::REQ-017"
        ),
        document_type=DocumentType.FRD,
        document_name="FRD_System_Rev01.pdf",
        document_revision="rev1",
        page_number=18,
        section="4.2.1",
        requirement_id="req_017",
        chunk_id="FRD-CHUNK-001",
        content_hash=calculate_content_hash(content),
        parsing_method=ParsingMethod.REQUIREMENT,
    )

    assert metadata.document_revision == "Rev01"
    assert metadata.requirement_id == "REQ-017"
    assert metadata.page_number == 18


def test_frd_metadata_requires_requirement_id() -> None:
    with pytest.raises(ValidationError):
        EvidenceMetadata(
            evidence_id="FRD::file.pdf::PAGE-1::UNKNOWN",
            document_type=DocumentType.FRD,
            document_name="file.pdf",
            page_number=1,
            chunk_id="FRD-001",
            content_hash="a" * 64,
        )


def test_observation_metadata_requires_observation_id() -> None:
    with pytest.raises(ValidationError):
        EvidenceMetadata(
            evidence_id="OBS::file.pdf::PAGE-1::UNKNOWN",
            document_type=DocumentType.OBSERVATION,
            document_name="file.pdf",
            page_number=1,
            chunk_id="OBS-001",
            content_hash="a" * 64,
        )


def test_test_metadata_requires_test_case_id() -> None:
    with pytest.raises(ValidationError):
        EvidenceMetadata(
            evidence_id="TEST::file.pdf::PAGE-1::UNKNOWN",
            document_type=DocumentType.TEST,
            document_name="file.pdf",
            page_number=1,
            chunk_id="TEST-001",
            content_hash="a" * 64,
        )


def test_code_metadata_requires_line_range() -> None:
    with pytest.raises(ValidationError):
        EvidenceMetadata(
            evidence_id="CODE::file.c::function::1-10",
            document_type=DocumentType.CODE,
            document_name="file.c",
            file_name="file.c",
            function_name="function",
            chunk_id="CODE-001",
            content_hash="a" * 64,
        )


def test_non_code_metadata_rejects_line_numbers() -> None:
    with pytest.raises(ValidationError):
        EvidenceMetadata(
            evidence_id="FRD::file.pdf::PAGE-1::REQ-017",
            document_type=DocumentType.FRD,
            document_name="file.pdf",
            page_number=1,
            requirement_id="REQ-017",
            start_line=1,
            end_line=10,
            chunk_id="FRD-001",
            content_hash="a" * 64,
        )


def test_code_metadata_rejects_page_number() -> None:
    with pytest.raises(ValidationError):
        EvidenceMetadata(
            evidence_id="CODE::file.c::function::1-10",
            document_type=DocumentType.CODE,
            document_name="file.c",
            file_name="file.c",
            function_name="function",
            page_number=1,
            start_line=1,
            end_line=10,
            chunk_id="CODE-001",
            content_hash="a" * 64,
        )


def test_chroma_metadata_excludes_none_values() -> None:
    metadata = EvidenceMetadata(
        evidence_id="FRD::file.pdf::PAGE-1::REQ-017",
        document_type=DocumentType.FRD,
        document_name="file.pdf",
        page_number=1,
        requirement_id="REQ-017",
        chunk_id="FRD-001",
        content_hash="a" * 64,
    )

    chroma_metadata = metadata.to_chroma_metadata()

    assert "observation_id" not in chroma_metadata
    assert "test_case_id" not in chroma_metadata
    assert chroma_metadata["page_number"] == 1
    assert chroma_metadata["requirement_id"] == "REQ-017"


def test_valid_evidence_record() -> None:
    content = """
    Requirement ID: REQ-017

    The system shall handle null input without a system exception.
    """

    content_hash = calculate_content_hash(content)

    metadata = EvidenceMetadata(
        evidence_id=(
            "FRD::FRD_Input_Validation_Rev01.pdf"
            "::PAGE-1::REQ-017"
        ),
        document_type=DocumentType.FRD,
        document_name="FRD_Input_Validation_Rev01.pdf",
        document_revision="Rev01",
        page_number=1,
        requirement_id="REQ-017",
        chunk_id="FRD-REQ-017-001",
        content_hash=content_hash,
        parsing_method=ParsingMethod.REQUIREMENT,
    )

    evidence = EvidenceRecord(
        content=content,
        metadata=metadata,
    )

    assert evidence.metadata.requirement_id == "REQ-017"
    assert evidence.citation_anchor() == (
        "[FRD::FRD_Input_Validation_Rev01.pdf"
        "::PAGE-1::REQ-017]"
    )


def test_evidence_record_rejects_incorrect_content_hash() -> None:
    metadata = EvidenceMetadata(
        evidence_id="FRD::file.pdf::PAGE-1::REQ-017",
        document_type=DocumentType.FRD,
        document_name="file.pdf",
        page_number=1,
        requirement_id="REQ-017",
        chunk_id="FRD-001",
        content_hash="a" * 64,
    )

    with pytest.raises(ValidationError):
        EvidenceRecord(
            content="Different content",
            metadata=metadata,
        )


def test_evidence_record_rejects_incorrect_evidence_id() -> None:
    content = "REQ-017 requirement content."

    metadata = EvidenceMetadata(
        evidence_id="FRD::wrong.pdf::PAGE-99::REQ-999",
        document_type=DocumentType.FRD,
        document_name="correct.pdf",
        page_number=1,
        requirement_id="REQ-017",
        chunk_id="FRD-001",
        content_hash=calculate_content_hash(content),
    )

    with pytest.raises(ValidationError):
        EvidenceRecord(
            content=content,
            metadata=metadata,
        )


def test_inconclusive_decision_requires_blocking_items() -> None:
    with pytest.raises(ValidationError):
        DeterministicDecision(
            assessment=AssessmentValue.INCONCLUSIVE,
            rule_id="RULE-INCONCLUSIVE-001",
            decision_explanation="Evidence is insufficient.",
            blocking_items=[],
            evidence_completeness=EvidenceCompleteness.INSUFFICIENT,
        )


def test_valid_inconclusive_decision() -> None:
    decision = DeterministicDecision(
        assessment=AssessmentValue.INCONCLUSIVE,
        rule_id="RULE-INCONCLUSIVE-001",
        decision_explanation="Required test evidence is unavailable.",
        blocking_items=[
            "Required execution evidence is unavailable",
        ],
        evidence_completeness=EvidenceCompleteness.INSUFFICIENT,
    )

    assert decision.assessment == AssessmentValue.INCONCLUSIVE


def test_closure_not_established_requires_blocking_items() -> None:
    with pytest.raises(ValidationError):
        ClosureDecision(
            status=ClosureStatus.NOT_ESTABLISHED,
            explanation="Closure evidence is incomplete.",
            blocking_items=[],
        )


def test_engineering_assessment_preserves_closure_separation() -> None:
    assessment = EngineeringAssessment(
        assessment=AssessmentValue.SUPPORTED,
        confidence=ConfidenceLevel.HIGH,
        decision_rule_id="RULE-SUPPORTED-001",
        observation=ObservationSummary(
            id="OBS-070",
            summary="Recovery counter is not initialized.",
            source="Observation report",
        ),
        frd_requirement=FRDRequirementSummary(
            id="REQ-060",
            summary="Initialize the recovery counter to zero.",
            revision="Rev01",
            source="FRD",
        ),
        expected_behavior="Counter is initialized to zero.",
        observed_behavior="Previous counter value remains.",
        code_analysis=CodeAnalysis(
            files=["recovery_manager.c"],
            functions=["start_recovery"],
            summary="The original baseline omits initialization.",
            limitations=["Corrected code is unavailable."],
        ),
        test_evidence=[],
        evidence_comparison=EvidenceComparison(
            supports=["Original baseline supports the observation."],
            contradicts=[],
            unresolved=["Corrective action is unavailable."],
        ),
        supporting_evidence=[
            "Original code and test support the deviation."
        ],
        contradicting_evidence=[],
        missing_information=[
            "Corrected implementation",
            "Post-correction verification",
        ],
        blocking_items=[
            "Corrective action is not demonstrated",
        ],
        closure_status=ClosureStatus.NOT_ESTABLISHED,
        recommended_reviewer_action=(
            "Obtain corrected code and verification evidence."
        ),
        justification=(
            "The observation is supported, but closure is not established."
        ),
        sources=[],
        grounding_status=GroundingStatus.FAILED,
    )

    assert assessment.assessment == AssessmentValue.SUPPORTED
    assert (
        assessment.closure_status
        == ClosureStatus.NOT_ESTABLISHED
    )
"""
Tests for Phase 14D constrained repair prompts.
"""

from __future__ import annotations

import pytest

from src.narration_models import (
    NarrationEvidenceBlock,
    NarrationRequest,
    NarrationResponse,
)
from src.narration_repair import (
    NarrationRepairError,
    NarrationRepairRequest,
    build_repair_prompt,
)
from src.narration_validation import (
    validate_narration,
)


def make_request() -> NarrationRequest:
    block = NarrationEvidenceBlock(
        citation_id="CIT-001",
        evidence_id="OBS-023",
        document_type="observation",
        source_label="OBSERVATION | report.pdf | page 1",
        provenance_location="report.pdf, page 1",
        excerpt="OBS-023 reports a failure.",
        requirement_id="REQ-017",
        observation_id="OBS-023",
        test_case_id="TC-102",
    )

    return NarrationRequest(
        request_version="1.0",
        query="OBS-023",
        assessment="SUPPORTED",
        assessment_rule=(
            "AR-008_EXECUTED_FAIL_SUPPORT"
        ),
        assessment_reason="Reason.",
        confidence="HIGH",
        confidence_reason="Reason.",
        closure_status="NOT ASSESSED",
        closure_reason="Not requested.",
        closure_was_assessed=False,
        requirement_ids=("REQ-017",),
        observation_ids=("OBS-023",),
        test_case_ids=("TC-102",),
        supporting_items=(),
        blocking_items=(),
        allowed_claims=(
            "OBS-023 is SUPPORTED.",
        ),
        prohibited_claims=(
            "Do not state that OBS-023 is closed.",
        ),
        citation_allowlist=("CIT-001",),
        engineering_id_allowlist=(
            "REQ-017",
            "OBS-023",
            "TC-102",
        ),
        evidence_blocks=(block,),
        narrator_instructions=(
            "Preserve the assessment exactly.",
        ),
        deterministic_fallback_text=(
            "Assessment: SUPPORTED"
        ),
    )


def make_failed_artifacts():
    request = make_request()

    response = NarrationResponse(
        model="llama3.2:1b",
        text="Assessment: SUPPORTED",
        attempt_number=1,
        is_repair_attempt=False,
    )

    validation = validate_narration(
        request=request,
        response=response,
    )

    assert validation.passed is False

    return request, response, validation


def test_repair_prompt_contains_validation_errors() -> None:
    request, response, validation = (
        make_failed_artifacts()
    )

    prompt = build_repair_prompt(
        NarrationRepairRequest(
            original_request=request,
            failed_response=response,
            failed_validation=validation,
        )
    )

    assert (
        "VALIDATION ERRORS TO CORRECT"
        in prompt.user_prompt
    )

    assert (
        "NV-006_REQUIRED_CITATIONS"
        in prompt.user_prompt
    )


def test_repair_prompt_preserves_decision() -> None:
    request, response, validation = (
        make_failed_artifacts()
    )

    prompt = build_repair_prompt(
        NarrationRepairRequest(
            original_request=request,
            failed_response=response,
            failed_validation=validation,
        )
    )

    assert (
        "Assessment: SUPPORTED"
        in prompt.system_prompt
    )

    assert (
        "Confidence: HIGH"
        in prompt.system_prompt
    )

    assert (
        "Closure Status: NOT ASSESSED"
        in prompt.system_prompt
    )


def test_repair_prompt_preserves_allowlists() -> None:
    request, response, validation = (
        make_failed_artifacts()
    )

    prompt = build_repair_prompt(
        NarrationRepairRequest(
            original_request=request,
            failed_response=response,
            failed_validation=validation,
        )
    )

    assert "REQ-017" in prompt.user_prompt
    assert "OBS-023" in prompt.user_prompt
    assert "TC-102" in prompt.user_prompt
    assert "[CIT-001]" in prompt.user_prompt


def test_valid_response_cannot_be_repaired() -> None:
    request = make_request()

    valid_response = NarrationResponse(
        model="llama3.2:1b",
        text=(
            "Assessment\n\n"
            "Assessment: SUPPORTED\n"
            "Confidence: HIGH"
        ),
        attempt_number=1,
        is_repair_attempt=False,
    )

    class PassedValidation:
        passed = True

    with pytest.raises(
        NarrationRepairError
    ):
        NarrationRepairRequest(
            original_request=request,
            failed_response=valid_response,
            failed_validation=PassedValidation(),
        )
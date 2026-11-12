"""
Tests for Phase 14 narration models.
"""

from __future__ import annotations

import json

import pytest

from src.narration_models import (
    EvidenceTrustClass,
    NarrationEvidenceBlock,
    NarrationModelError,
    NarrationRequest,
    NarrationResponse,
)


def make_block() -> NarrationEvidenceBlock:
    """Create one valid narrator evidence block."""

    return NarrationEvidenceBlock(
        citation_id="CIT-001",
        evidence_id="OBS-023",
        document_type="observation",
        source_label=(
            "OBSERVATION | report.pdf | page 1"
        ),
        provenance_location=(
            "report.pdf, page 1"
        ),
        excerpt=(
            "Null input caused the observed failure."
        ),
        requirement_id="REQ-017",
        observation_id="OBS-023",
        test_case_id="TC-102",
    )


def make_request() -> NarrationRequest:
    """Create one valid narration request."""

    return NarrationRequest(
        request_version="1.0",
        query="OBS-023",
        assessment="SUPPORTED",
        assessment_rule=(
            "AR-008_EXECUTED_FAIL_SUPPORT"
        ),
        assessment_reason=(
            "Complete failing evidence supports "
            "the observation."
        ),
        confidence="HIGH",
        confidence_reason=(
            "Complete decisive evidence is available."
        ),
        closure_status="NOT ASSESSED",
        closure_reason=(
            "Closure was not requested."
        ),
        closure_was_assessed=False,
        requirement_ids=(
            "REQ-017",
        ),
        observation_ids=(
            "OBS-023",
        ),
        test_case_ids=(
            "TC-102",
        ),
        supporting_items=(
            "Executed FAIL evidence is available.",
        ),
        blocking_items=(),
        allowed_claims=(
            "OBS-023 is SUPPORTED.",
        ),
        prohibited_claims=(
            "Do not state that OBS-023 is closed.",
        ),
        citation_allowlist=(
            "CIT-001",
        ),
        engineering_id_allowlist=(
            "REQ-017",
            "OBS-023",
            "TC-102",
        ),
        evidence_blocks=(
            make_block(),
        ),
        narrator_instructions=(
            "Preserve the assessment exactly.",
        ),
        deterministic_fallback_text=(
            "Assessment: SUPPORTED"
        ),
    )


def test_evidence_block_is_untrusted() -> None:
    """Evidence must always remain untrusted quoted content."""

    block = make_block()

    assert block.trust_class == (
        EvidenceTrustClass
        .UNTRUSTED_QUOTED_EVIDENCE
    )


def test_block_citation_must_be_allowlisted() -> None:
    """Evidence citation outside request allowlist must fail."""

    block = make_block()

    with pytest.raises(
        NarrationModelError
    ):
        NarrationRequest(
            request_version="1.0",
            query="OBS-023",
            assessment="SUPPORTED",
            assessment_rule="AR-008",
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
            allowed_claims=("Allowed.",),
            prohibited_claims=("Prohibited.",),
            citation_allowlist=(
                "CIT-999",
            ),
            engineering_id_allowlist=(
                "REQ-017",
                "OBS-023",
                "TC-102",
            ),
            evidence_blocks=(
                block,
            ),
            narrator_instructions=(
                "Preserve decision.",
            ),
            deterministic_fallback_text=(
                "Fallback."
            ),
        )


def test_block_engineering_ids_must_be_allowlisted() -> None:
    """Evidence IDs must remain inside engineering-ID allowlist."""

    block = make_block()

    with pytest.raises(
        NarrationModelError
    ):
        NarrationRequest(
            request_version="1.0",
            query="OBS-023",
            assessment="SUPPORTED",
            assessment_rule="AR-008",
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
            allowed_claims=("Allowed.",),
            prohibited_claims=("Prohibited.",),
            citation_allowlist=(
                "CIT-001",
            ),
            engineering_id_allowlist=(
                "REQ-999",
            ),
            evidence_blocks=(
                block,
            ),
            narrator_instructions=(
                "Preserve decision.",
            ),
            deterministic_fallback_text=(
                "Fallback."
            ),
        )


def test_request_is_json_safe() -> None:
    """Narration request must serialize to JSON."""

    request = make_request()

    serialized = json.dumps(
        request.to_dict()
    )

    assert serialized

    assert request.to_dict()[
        "authoritative_decision"
    ][
        "assessment"
    ] == "SUPPORTED"


def test_response_requires_non_empty_text() -> None:
    """Raw model response must contain text."""

    with pytest.raises(
        NarrationModelError
    ):
        NarrationResponse(
            model="llama3.2:1b",
            text="   ",
            attempt_number=1,
            is_repair_attempt=False,
        )


def test_response_attempt_is_one_based() -> None:
    """Narration attempt numbers must begin at one."""

    with pytest.raises(
        NarrationModelError
    ):
        NarrationResponse(
            model="llama3.2:1b",
            text="Narration.",
            attempt_number=0,
            is_repair_attempt=False,
        )


def test_valid_response() -> None:
    """Valid raw response metadata should be preserved."""

    response = NarrationResponse(
        model="llama3.2:1b",
        text="Grounded narration.",
        attempt_number=1,
        is_repair_attempt=False,
        prompt_tokens=100,
        completion_tokens=50,
        generation_duration_seconds=1.5,
    )

    assert response.model == (
        "llama3.2:1b"
    )

    assert response.attempt_number == 1
"""
Tests for Phase 14C narration-output validation.
"""

from __future__ import annotations

import pytest

from src.narration_models import (
    NarrationEvidenceBlock,
    NarrationRequest,
    NarrationResponse,
)
from src.narration_validation import (
    NarrationValidationError,
    NarrationValidationRule,
    validate_narration,
)


def make_request(
    *,
    assessment: str = "SUPPORTED",
    confidence: str = "HIGH",
    closure_status: str = "NOT ASSESSED",
    assessment_rule: str = (
        "AR-008_EXECUTED_FAIL_SUPPORT"
    ),
) -> NarrationRequest:
    """Create one complete validation request."""

    blocks = (
        NarrationEvidenceBlock(
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
                "OBS-023 reports the null-input failure."
            ),
            requirement_id="REQ-017",
            observation_id="OBS-023",
            test_case_id="TC-102",
        ),
        NarrationEvidenceBlock(
            citation_id="CIT-002",
            evidence_id="CODE-023",
            document_type="code",
            source_label=(
                "CODE | data_processor.c | "
                "process_data | lines 26-47"
            ),
            provenance_location=(
                "data_processor.c, function "
                "process_data, lines 26-47"
            ),
            excerpt=(
                "The implementation dereferences input."
            ),
            requirement_id="REQ-017",
            observation_id="OBS-023",
            test_case_id="TC-102",
        ),
        NarrationEvidenceBlock(
            citation_id="CIT-003",
            evidence_id="FRD-017",
            document_type="frd",
            source_label=(
                "FRD | requirement.pdf | page 1"
            ),
            provenance_location=(
                "requirement.pdf, page 1"
            ),
            excerpt=(
                "REQ-017 requires safe null handling."
            ),
            requirement_id="REQ-017",
        ),
        NarrationEvidenceBlock(
            citation_id="CIT-004",
            evidence_id="TEST-102",
            document_type="test",
            source_label=(
                "TEST | test.pdf | page 1"
            ),
            provenance_location=(
                "test.pdf, page 1"
            ),
            excerpt=(
                "TC-102 Result Status: FAIL."
            ),
            requirement_id="REQ-017",
            test_case_id="TC-102",
            result_status="FAIL",
        ),
    )

    return NarrationRequest(
        request_version="1.0",
        query="OBS-023",
        assessment=assessment,
        assessment_rule=(
            assessment_rule
        ),
        assessment_reason=(
            "Complete failing evidence supports "
            "the observation."
        ),
        confidence=confidence,
        confidence_reason=(
            "Complete decisive evidence is available."
        ),
        closure_status=closure_status,
        closure_reason=(
            "Closure readiness was not requested."
        ),
        closure_was_assessed=False,
        requirement_ids=("REQ-017",),
        observation_ids=("OBS-023",),
        test_case_ids=("TC-102",),
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
            "CIT-002",
            "CIT-003",
            "CIT-004",
        ),
        engineering_id_allowlist=(
            "REQ-017",
            "OBS-023",
            "TC-102",
        ),
        evidence_blocks=blocks,
        narrator_instructions=(
            "Preserve the assessment exactly.",
            "Use only engineering IDs in the allowlist.",
            "Use only citations in the allowlist.",
            "Do not infer facts from missing evidence.",
        ),
        deterministic_fallback_text=(
            "Assessment: SUPPORTED"
        ),
    )


def valid_narration() -> str:
    """Return one narration satisfying every rule."""

    return """
Assessment

Assessment: SUPPORTED
Confidence: HIGH

Engineering Basis

OBS-023 is supported by REQ-017 and the cited implementation evidence.
The relevant implementation is process_data in data_processor.c,
lines 26-47 [CIT-002]. The requirement evidence is cited in [CIT-003].

Evidence Summary

The observation evidence reports the null-input failure [CIT-001].
TC-102 has the source-reported result FAIL [CIT-004].

Blocking or Missing Evidence

No assessment blocker was recorded.

Closure Status

Closure Status: NOT ASSESSED.
Closure readiness was not requested.

Sources

[CIT-001]
[CIT-002]
[CIT-003]
[CIT-004]

Authority Limitation

This result provides evidence-derived decision support.
Final engineering authorization remains with the responsible human authority.
""".strip()


def make_response(
    text: str,
) -> NarrationResponse:
    """Wrap raw model text."""

    return NarrationResponse(
        model="llama3.2:1b",
        text=text,
        attempt_number=1,
        is_repair_attempt=False,
    )


def test_valid_narration_passes() -> None:
    """Complete grounded narration should pass."""

    result = validate_narration(
        request=make_request(),
        response=make_response(
            valid_narration()
        ),
    )

    assert result.passed is True
    assert result.error_count == 0


def test_altered_assessment_fails() -> None:
    """Narrator cannot change SUPPORTED to NOT SUPPORTED."""

    text = valid_narration().replace(
        "Assessment: SUPPORTED",
        "Assessment: NOT SUPPORTED",
    )

    result = validate_narration(
        request=make_request(),
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        NarrationValidationRule
        .ASSESSMENT_PRESERVED
        in result.failed_rules
    )


def test_missing_test_citation_fails() -> None:
    """TC-102 FAIL requires CIT-004."""

    text = valid_narration().replace(
        "[CIT-004]",
        "",
    )

    result = validate_narration(
        request=make_request(),
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        NarrationValidationRule
        .REQUIRED_CITATIONS
        in result.failed_rules
    )


def test_fabricated_citation_fails() -> None:
    """CIT-999 must be rejected."""

    text = (
        valid_narration()
        + "\n[CIT-999]"
    )

    result = validate_narration(
        request=make_request(),
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        NarrationValidationRule
        .CITATION_ALLOWLIST
        in result.failed_rules
    )


def test_unsupported_engineering_id_fails() -> None:
    """REQ-999 must be rejected."""

    text = (
        valid_narration()
        + "\nRelated requirement: REQ-999."
    )

    result = validate_narration(
        request=make_request(),
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        NarrationValidationRule
        .ENGINEERING_ID_ALLOWLIST
        in result.failed_rules
    )


def test_required_sections_are_enforced() -> None:
    """Missing Sources section must fail."""

    text = valid_narration().replace(
        "\nSources\n",
        "\nSource List\n",
    )

    result = validate_narration(
        request=make_request(),
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        NarrationValidationRule
        .REQUIRED_SECTIONS
        in result.failed_rules
    )


def test_false_closure_fails() -> None:
    """NOT ASSESSED must not become resolved."""

    text = (
        valid_narration()
        + "\nThe observation is resolved."
    )

    result = validate_narration(
        request=make_request(),
        response=make_response(text),
    )

    assert result.passed is False

    assert result.false_closure_detected is True


def test_control_leakage_fails() -> None:
    """Internal narrator instructions must not appear in output."""

    text = (
        valid_narration()
        + "\nMANDATORY NARRATOR INSTRUCTIONS:"
        + "\n- Preserve the assessment exactly."
    )

    result = validate_narration(
        request=make_request(),
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        result
        .narrator_control_leakage_detected
        is True
    )


def test_revision_selection_fails() -> None:
    """Model must not choose a revision during conflict."""

    request = make_request(
        assessment="INCONCLUSIVE",
        assessment_rule=(
            "AR-002_REQUIREMENT_REVISION_CONFLICT"
        ),
    )

    text = valid_narration().replace(
        "Assessment: SUPPORTED",
        "Assessment: INCONCLUSIVE",
    )

    text += (
        "\nRev02 is the applicable revision."
    )

    result = validate_narration(
        request=request,
        response=make_response(text),
    )

    assert result.passed is False
    assert result.revision_selection_detected is True


def test_traceability_rewrite_fails() -> None:
    """Model must not replace explicit traceability."""

    request = make_request(
        assessment="INCONCLUSIVE",
        assessment_rule=(
            "AR-003_TRACEABILITY_CONCERN"
        ),
    )

    text = valid_narration().replace(
        "Assessment: SUPPORTED",
        "Assessment: INCONCLUSIVE",
    )

    text += (
        "\nREQ-042 is the correct requirement."
    )

    result = validate_narration(
        request=request,
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        result.traceability_rewrite_detected
        is True
    )


def test_raw_obs_052_style_output_fails_sections_and_citations() -> None:
    """The short live OBS-052 output must be rejected."""

    request = make_request(
        assessment="INCONCLUSIVE",
        assessment_rule=(
            "AR-003_TRACEABILITY_CONCERN"
        ),
    )

    text = """
Assessment: INCONCLUSIVE

Confidence: HIGH

Closure Status: NOT ASSESSED

The observation contains an unresolved traceability concern.
""".strip()

    result = validate_narration(
        request=request,
        response=make_response(text),
    )

    assert result.passed is False

    assert (
        NarrationValidationRule
        .REQUIRED_CITATIONS
        in result.failed_rules
    )

    assert (
        NarrationValidationRule
        .REQUIRED_SECTIONS
        in result.failed_rules
    )


def test_strict_mode_raises() -> None:
    """Strict mode must reject invalid narration."""

    with pytest.raises(
        NarrationValidationError
    ):
        validate_narration(
            request=make_request(),
            response=make_response(
                "Assessment: SUPPORTED"
            ),
            strict=True,
        )
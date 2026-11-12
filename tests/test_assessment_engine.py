"""
Tests for Phase 12C deterministic observation assessment.
"""

from __future__ import annotations

from src.assessment_engine import (
    AssessmentRule,
    ObservationAssessment,
    assess_observation,
)
from src.context_builder import (
    FinalEvidenceContext,
)
from src.retrieval import (
    RetrievedEvidence,
)


def make_evidence(
    evidence_id: str,
    document_type: str,
    content: str,
    *,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
    result_status: str | None = None,
    document_revision: str | None = None,
) -> RetrievedEvidence:
    """Create synthetic evidence for assessment tests."""

    metadata: dict[str, object] = {}

    if requirement_id:
        metadata[
            "requirement_id"
        ] = requirement_id

    if observation_id:
        metadata[
            "observation_id"
        ] = observation_id

    if test_case_id:
        metadata[
            "test_case_id"
        ] = test_case_id

    if result_status:
        metadata[
            "result_status"
        ] = result_status

    if document_revision:
        metadata[
            "document_revision"
        ] = document_revision

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type=document_type,
        content=content,
        metadata=metadata,
        retrieval_method="test",
        score=1.0,
        rank=1,
    )


def make_context(
    evidence: list[RetrievedEvidence],
) -> FinalEvidenceContext:
    """Create one final evidence context."""

    return FinalEvidenceContext(
        evidence=tuple(evidence),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def supported_fail_context(
    observation_id: str,
    requirement_id: str,
    test_case_id: str,
) -> FinalEvidenceContext:
    """Create complete evidence with an executed FAIL test."""

    return make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement evidence.",
                requirement_id=(
                    requirement_id
                ),
                document_revision="Rev01",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Reported defect.",
                requirement_id=(
                    requirement_id
                ),
                observation_id=(
                    observation_id
                ),
                test_case_id=(
                    test_case_id
                ),
            ),
            make_evidence(
                "CODE",
                "code",
                "Defective implementation.",
                requirement_id=(
                    requirement_id
                ),
                observation_id=(
                    observation_id
                ),
                test_case_id=(
                    test_case_id
                ),
            ),
            make_evidence(
                "TEST",
                "test",
                "Result Status: FAIL.",
                requirement_id=(
                    requirement_id
                ),
                test_case_id=(
                    test_case_id
                ),
                result_status="FAIL",
            ),
        ]
    )


def test_executed_fail_complete_evidence_is_supported() -> None:
    """OBS-023 style complete failing evidence supports the defect."""

    result = assess_observation(
        supported_fail_context(
            observation_id="OBS-023",
            requirement_id="REQ-017",
            test_case_id="TC-102",
        )
    )

    assert result.assessment == (
        ObservationAssessment.SUPPORTED
    )

    assert result.applied_rule == (
        AssessmentRule
        .EXECUTED_FAIL_SUPPORT
    )


def test_obs_070_original_failure_is_supported() -> None:
    """
    Original OBS-070 evidence can support the observation.

    Closure remains outside Phase 12C.
    """

    result = assess_observation(
        supported_fail_context(
            observation_id="OBS-070",
            requirement_id="REQ-060",
            test_case_id="TC-401",
        )
    )

    assert result.assessment == (
        ObservationAssessment.SUPPORTED
    )

    assert (
        "closure_status"
        not in result.__dict__
    )


def test_technical_contradiction_is_not_supported() -> None:
    """OBS-031 style technical contradiction maps to NOT SUPPORTED."""

    context = make_context(
        [
            make_evidence(
                "OBS",
                "observation",
                (
                    "The command identifier 17 was accepted "
                    "instead of being rejected."
                ),
                requirement_id="REQ-021",
                observation_id="OBS-031",
                test_case_id="TC-201",
            ),
            make_evidence(
                "CODE",
                "code",
                (
                    "if ((command_id < COMMAND_ID_MINIMUM) || "
                    "(command_id > COMMAND_ID_MAXIMUM)) "
                    "{ return COMMAND_INVALID_ID; }"
                ),
                requirement_id="REQ-021",
                observation_id="OBS-031",
                test_case_id="TC-201",
            ),
            make_evidence(
                "FRD",
                "frd",
                (
                    "The system shall reject commands whose "
                    "command identifier is outside the "
                    "inclusive range 1 through 16."
                ),
                requirement_id="REQ-021",
                document_revision="Rev01",
            ),
            make_evidence(
                "TEST",
                "test",
                (
                    "Result Status: PASS. Identifier 17 "
                    "returned COMMAND_INVALID_ID."
                ),
                requirement_id="REQ-021",
                test_case_id="TC-201",
                result_status="PASS",
            ),
        ]
    )

    result = assess_observation(
        context
    )

    assert result.assessment == (
        ObservationAssessment.NOT_SUPPORTED
    )

    assert result.applied_rule == (
        AssessmentRule
        .TECHNICAL_CONTRADICTION
    )


def test_not_executed_is_inconclusive() -> None:
    """OBS-044 style unexecuted test must remain inconclusive."""

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement evidence.",
                requirement_id="REQ-033",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Reported behavior.",
                requirement_id="REQ-033",
                observation_id="OBS-044",
                test_case_id="TC-301",
            ),
            make_evidence(
                "CODE",
                "code",
                "Partial source code.",
                requirement_id="REQ-033",
                observation_id="OBS-044",
                test_case_id="TC-301",
            ),
            make_evidence(
                "TEST",
                "test",
                "Test was not executed.",
                requirement_id="REQ-033",
                test_case_id="TC-301",
                result_status=(
                    "NOT EXECUTED"
                ),
            ),
        ]
    )

    result = assess_observation(
        context
    )

    assert result.assessment == (
        ObservationAssessment.INCONCLUSIVE
    )

    assert result.applied_rule == (
        AssessmentRule
        .TEST_NOT_EXECUTED
    )


def test_traceability_concern_is_inconclusive() -> None:
    """OBS-052 must remain linked to REQ-041 and inconclusive."""

    context = make_context(
        [
            make_evidence(
                "OBS",
                "observation",
                (
                    "Traceability Concern. The reference "
                    "must not silently replace REQ-041 "
                    "using semantic similarity alone."
                ),
                requirement_id="REQ-041",
                observation_id="OBS-052",
            ),
            make_evidence(
                "FRD",
                "frd",
                "REQ-041 requirement evidence.",
                requirement_id="REQ-041",
                document_revision="Rev01",
            ),
        ]
    )

    result = assess_observation(
        context
    )

    assert result.assessment == (
        ObservationAssessment.INCONCLUSIVE
    )

    assert result.applied_rule == (
        AssessmentRule
        .TRACEABILITY_CONCERN
    )

    assert result.requirement_ids == (
        "REQ-041",
    )


def test_revision_conflict_is_inconclusive() -> None:
    """OBS-061 cannot be decided until applicable revision is known."""

    context = make_context(
        [
            make_evidence(
                "FRD-REV01",
                "frd",
                "Requires 500 milliseconds.",
                requirement_id="REQ-050",
                document_revision="Rev01",
            ),
            make_evidence(
                "FRD-REV02",
                "frd",
                "Requires 1000 milliseconds.",
                requirement_id="REQ-050",
                document_revision="Rev02",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observed 760 milliseconds.",
                requirement_id="REQ-050",
                observation_id="OBS-061",
            ),
        ]
    )

    result = assess_observation(
        context
    )

    assert result.assessment == (
        ObservationAssessment.INCONCLUSIVE
    )

    assert result.applied_rule == (
        AssessmentRule
        .REQUIREMENT_REVISION_CONFLICT
    )


def test_missing_frd_is_inconclusive() -> None:
    """Insufficient evidence has highest conservative precedence."""

    context = make_context(
        [
            make_evidence(
                "OBS",
                "observation",
                "Reported defect.",
                observation_id="OBS-999",
            ),
            make_evidence(
                "TEST",
                "test",
                "Result Status: FAIL.",
                test_case_id="TC-999",
                result_status="FAIL",
            ),
        ]
    )

    result = assess_observation(
        context
    )

    assert result.assessment == (
        ObservationAssessment.INCONCLUSIVE
    )

    assert result.applied_rule == (
        AssessmentRule
        .INSUFFICIENT_EVIDENCE
    )


def test_revision_conflict_precedes_fail_support() -> None:
    """
    An executed FAIL test must not override an unresolved
    requirement revision conflict.
    """

    context = make_context(
        [
            make_evidence(
                "FRD-REV01",
                "frd",
                "Requires 500 milliseconds.",
                requirement_id="REQ-050",
                document_revision="Rev01",
            ),
            make_evidence(
                "FRD-REV02",
                "frd",
                "Requires 1000 milliseconds.",
                requirement_id="REQ-050",
                document_revision="Rev02",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observed 760 milliseconds.",
                requirement_id="REQ-050",
                observation_id="OBS-061",
                test_case_id="TC-501",
            ),
            make_evidence(
                "CODE",
                "code",
                "Timing implementation.",
                requirement_id="REQ-050",
            ),
            make_evidence(
                "TEST",
                "test",
                "Result Status: FAIL.",
                requirement_id="REQ-050",
                test_case_id="TC-501",
                result_status="FAIL",
            ),
        ]
    )

    result = assess_observation(
        context
    )

    assert result.assessment == (
        ObservationAssessment.INCONCLUSIVE
    )

    assert result.applied_rule == (
        AssessmentRule
        .REQUIREMENT_REVISION_CONFLICT
    )


def test_pass_without_contradiction_is_not_automatically_not_supported() -> None:
    """
    PASS alone cannot establish that an observation is unsupported.
    """

    context = make_context(
        [
            make_evidence(
                "FRD",
                "frd",
                "Requirement evidence.",
                requirement_id="REQ-100",
            ),
            make_evidence(
                "OBS",
                "observation",
                "Observation evidence.",
                requirement_id="REQ-100",
                observation_id="OBS-100",
                test_case_id="TC-100",
            ),
            make_evidence(
                "CODE",
                "code",
                "Implementation evidence.",
                requirement_id="REQ-100",
            ),
            make_evidence(
                "TEST",
                "test",
                "Result Status: PASS.",
                requirement_id="REQ-100",
                test_case_id="TC-100",
                result_status="PASS",
            ),
        ]
    )

    result = assess_observation(
        context
    )

    assert result.assessment == (
        ObservationAssessment.INCONCLUSIVE
    )

    assert result.applied_rule == (
        AssessmentRule
        .NO_DECISIVE_RULE
    )


def test_phase_12c_has_no_closure_decision() -> None:
    """Assessment and closure must remain separate."""

    result = assess_observation(
        supported_fail_context(
            observation_id="OBS-070",
            requirement_id="REQ-060",
            test_case_id="TC-401",
        )
    )

    result_fields = set(
        result.__dict__
    )

    assert "closure" not in result_fields
    assert "closure_status" not in result_fields
    assert "closure_readiness" not in result_fields
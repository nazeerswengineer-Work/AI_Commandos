"""
Tests for Phase 13B grounding validation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.assessment_engine import (
    assess_observation,
)
from src.citations import (
    CitationCollection,
    build_citation_collection,
)
from src.context_builder import (
    FinalEvidenceContext,
)
from src.decision_engine import (
    evaluate_engineering_decision,
)
from src.grounding import (
    GroundingRule,
    GroundingValidationError,
    validate_grounding,
)
from src.retrieval import (
    RetrievedEvidence,
)


VALID_HASH = (
    "b"
    * 64
)


def make_evidence(
    evidence_id: str,
    document_type: str,
    content: str,
    *,
    document_name: str | None = None,
    document_revision: str | None = None,
    page_number: int | None = None,
    file_name: str | None = None,
    function_name: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
    result_status: str | None = None,
) -> RetrievedEvidence:
    """Create evidence with citation-ready metadata."""

    metadata: dict[str, object] = {
        "content_hash": VALID_HASH,
    }

    optional_values = {
        "document_name": document_name,
        "document_revision": (
            document_revision
        ),
        "page_number": page_number,
        "file_name": file_name,
        "function_name": function_name,
        "start_line": start_line,
        "end_line": end_line,
        "requirement_id": requirement_id,
        "observation_id": observation_id,
        "test_case_id": test_case_id,
        "result_status": result_status,
    }

    for field_name, value in (
        optional_values.items()
    ):
        if value is not None:
            metadata[field_name] = value

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
    """Create one final context."""

    return FinalEvidenceContext(
        evidence=tuple(evidence),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def obs_023_context() -> FinalEvidenceContext:
    """Create a complete supported-observation context."""

    return make_context(
        [
            make_evidence(
                "OBS-023-EVIDENCE",
                "observation",
                "Null input caused the reported defect.",
                document_name=(
                    "Observation_Report_OBS-023.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                requirement_id="REQ-017",
                observation_id="OBS-023",
                test_case_id="TC-102",
                result_status="OPEN",
            ),
            make_evidence(
                "CODE-023",
                "code",
                "Defective process_data implementation.",
                document_name="data_processor.c",
                file_name="data_processor.c",
                function_name="process_data",
                start_line=26,
                end_line=47,
                requirement_id="REQ-017",
                observation_id="OBS-023",
                test_case_id="TC-102",
            ),
            make_evidence(
                "FRD-017",
                "frd",
                "REQ-017 requirement evidence.",
                document_name=(
                    "FRD_Input_Validation_Rev01.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                file_name=(
                    "FRD_Input_Validation_Rev01.pdf"
                ),
                requirement_id="REQ-017",
            ),
            make_evidence(
                "TEST-102",
                "test",
                "Result Status: FAIL.",
                document_name=(
                    "Test_Report_TC-102.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                file_name=(
                    "Test_Report_TC-102.pdf"
                ),
                requirement_id="REQ-017",
                test_case_id="TC-102",
                result_status="FAIL",
            ),
        ]
    )


def validate_context(
    context: FinalEvidenceContext,
    *,
    evaluate_closure: bool = False,
):
    """Build decision, citations and grounding."""

    decision = evaluate_engineering_decision(
        context,
        evaluate_closure=(
            evaluate_closure
        ),
    )

    citations = build_citation_collection(
        context
    )

    grounding = validate_grounding(
        context=context,
        decision=decision,
        citations=citations,
    )

    return (
        decision,
        citations,
        grounding,
    )


def test_complete_supported_context_is_grounded() -> None:
    """Complete OBS-023 decision should pass grounding."""

    _, _, grounding = validate_context(
        obs_023_context()
    )

    assert grounding.passed is True
    assert grounding.error_count == 0


def test_missing_context_citation_fails() -> None:
    """Every final-context record must have a citation."""

    context = obs_023_context()

    decision = evaluate_engineering_decision(
        context
    )

    full_citations = (
        build_citation_collection(
            context
        )
    )

    incomplete = CitationCollection(
        citations=(
            full_citations.citations[
                :-1
            ]
        )
    )

    grounding = validate_grounding(
        context=context,
        decision=decision,
        citations=incomplete,
    )

    assert grounding.passed is False

    assert (
        GroundingRule
        .CONTEXT_CITATION_COVERAGE
        in grounding.failed_rules
    )


def test_test_reference_without_test_citation_fails() -> None:
    """
    TC-102 references on Observation and Code do not replace
    actual Test evidence.
    """

    context = obs_023_context()

    decision = evaluate_engineering_decision(
        context
    )

    citations = build_citation_collection(
        context
    )

    without_test = (
        CitationCollection(
            citations=tuple(
                citation
                for citation in citations.citations
                if (
                    citation.document_type
                    != "test"
                )
            )
        )
    )

    grounding = validate_grounding(
        context=context,
        decision=decision,
        citations=without_test,
    )

    assert grounding.passed is False

    assert (
        GroundingRule
        .TEST_RESULT_GROUNDING
        in grounding.failed_rules
    )


def test_decisive_assessment_requires_code_citation() -> None:
    """Decisive technical assessment requires code provenance."""

    context = obs_023_context()

    decision = evaluate_engineering_decision(
        context
    )

    citations = build_citation_collection(
        context
    )

    without_code = CitationCollection(
        citations=tuple(
            citation
            for citation in citations.citations
            if (
                citation.document_type
                != "code"
            )
        )
    )

    grounding = validate_grounding(
        context=context,
        decision=decision,
        citations=without_code,
    )

    assert grounding.passed is False

    assert (
        GroundingRule.CODE_GROUNDING
        in grounding.failed_rules
    )


def test_revision_conflict_requires_two_revisions() -> None:
    """REQ-050 conflict must cite multiple FRD revisions."""

    context = make_context(
        [
            make_evidence(
                "FRD-050-REV01",
                "frd",
                "Requires 500 milliseconds.",
                document_name=(
                    "FRD_Alert_Handling_Rev01.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                requirement_id="REQ-050",
            ),
            make_evidence(
                "FRD-050-REV02",
                "frd",
                "Requires 1000 milliseconds.",
                document_name=(
                    "FRD_Alert_Handling_Rev02.pdf"
                ),
                document_revision="Rev02",
                page_number=1,
                requirement_id="REQ-050",
            ),
            make_evidence(
                "OBS-061",
                "observation",
                "Observed 760 milliseconds.",
                document_name=(
                    "Observation_Report_OBS-061.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                requirement_id="REQ-050",
                observation_id="OBS-061",
            ),
        ]
    )

    decision = evaluate_engineering_decision(
        context
    )

    citations = build_citation_collection(
        context
    )

    grounding = validate_grounding(
        context=context,
        decision=decision,
        citations=citations,
    )

    assert grounding.passed is True

    one_revision = CitationCollection(
        citations=tuple(
            citation
            for citation in citations.citations
            if (
                citation.evidence_id
                != "FRD-050-REV02"
            )
        )
    )

    failed = validate_grounding(
        context=context,
        decision=decision,
        citations=one_revision,
    )

    assert failed.passed is False

    assert (
        GroundingRule
        .REVISION_CONFLICT_GROUNDING
        in failed.failed_rules
    )


def test_traceability_reference_is_preserved() -> None:
    """OBS-052 must preserve source reference REQ-041."""

    context = make_context(
        [
            make_evidence(
                "OBS-052",
                "observation",
                (
                    "Traceability Concern. "
                    "Semantic similarity alone must not "
                    "replace REQ-041."
                ),
                document_name=(
                    "Observation_Report_OBS-052.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                requirement_id="REQ-041",
                observation_id="OBS-052",
            ),
            make_evidence(
                "FRD-041",
                "frd",
                "REQ-041 requirement.",
                document_name=(
                    "FRD_Alert_Handling_Rev01.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                requirement_id="REQ-041",
            ),
        ]
    )

    _, _, grounding = validate_context(
        context
    )

    assert grounding.passed is True


def test_closure_not_established_requires_blocker() -> None:
    """Closure-not-established should be supported by blockers."""

    context = make_context(
        [
            make_evidence(
                "OBS-070",
                "observation",
                (
                    "Original defect. "
                    "Missing Closure Evidence."
                ),
                document_name=(
                    "Observation_Report_OBS-070.pdf"
                ),
                page_number=1,
                requirement_id="REQ-060",
                observation_id="OBS-070",
                test_case_id="TC-401",
            ),
            make_evidence(
                "CODE-070",
                "code",
                (
                    "Original defective implementation. "
                    "No corrected implementation is supplied."
                ),
                document_name="recovery_manager.c",
                file_name="recovery_manager.c",
                function_name="start_recovery",
                start_line=23,
                end_line=37,
                requirement_id="REQ-060",
                observation_id="OBS-070",
                test_case_id="TC-401",
            ),
            make_evidence(
                "FRD-060",
                "frd",
                "REQ-060 requirement.",
                document_name=(
                    "FRD_Recovery_Control_Rev01.pdf"
                ),
                page_number=1,
                requirement_id="REQ-060",
            ),
            make_evidence(
                "TEST-401",
                "test",
                (
                    "Result Status: FAIL. "
                    "Post-correction test result: Not available."
                ),
                document_name=(
                    "Test_Report_TC-401.pdf"
                ),
                page_number=1,
                requirement_id="REQ-060",
                test_case_id="TC-401",
                result_status="FAIL",
            ),
        ]
    )

    _, _, grounding = validate_context(
        context,
        evaluate_closure=True,
    )

    assert grounding.passed is True


def test_strict_mode_raises() -> None:
    """Strict grounding must block invalid answer preparation."""

    context = obs_023_context()

    decision = evaluate_engineering_decision(
        context
    )

    citations = CitationCollection(
        citations=()
    )

    with pytest.raises(
        GroundingValidationError
    ):
        validate_grounding(
            context=context,
            decision=decision,
            citations=citations,
            strict=True,
        )
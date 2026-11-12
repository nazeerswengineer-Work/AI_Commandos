"""
Tests for Phase 13C safe answer contracts.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.answer_contract import (
    AnswerContractError,
    ClaimCategory,
    build_answer_contract,
)
from src.citations import (
    build_citation_collection,
)
from src.context_builder import (
    FinalEvidenceContext,
)
from src.decision_engine import (
    evaluate_engineering_decision,
)
from src.grounding import (
    validate_grounding,
)
from src.retrieval import (
    RetrievedEvidence,
)


VALID_HASH = (
    "c"
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
    """Create citation-ready synthetic evidence."""

    metadata: dict[str, object] = {
        "content_hash": VALID_HASH,
    }

    values = {
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

    for field_name, value in values.items():
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
    """Create a supported OBS-023 context."""

    return make_context(
        [
            make_evidence(
                "OBS-023",
                "observation",
                "Null input caused the observed failure.",
                document_name=(
                    "Observation_Report_OBS-023.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                requirement_id="REQ-017",
                observation_id="OBS-023",
                test_case_id="TC-102",
            ),
            make_evidence(
                "CODE-023",
                "code",
                "Defective process_data implementation.",
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
                requirement_id="REQ-017",
                test_case_id="TC-102",
                result_status="FAIL",
            ),
        ]
    )


def build_contract(
    context: FinalEvidenceContext,
    *,
    evaluate_closure: bool = False,
):
    """Build one fully grounded answer contract."""

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

    return build_answer_contract(
        decision=decision,
        citations=citations,
        grounding=grounding,
    )


def test_supported_contract_is_created() -> None:
    """Grounded OBS-023 should produce a valid contract."""

    contract = build_contract(
        obs_023_context()
    )

    assert (
        contract.assessment.value
        == "SUPPORTED"
    )

    assert contract.confidence.value == (
        "HIGH"
    )

    assert contract.grounding_passed is True


def test_contract_contains_engineering_id_allowlist() -> None:
    """Only grounded engineering IDs should enter the allowlist."""

    contract = build_contract(
        obs_023_context()
    )

    assert (
        contract.engineering_id_allowlist
        == (
            "REQ-017",
            "OBS-023",
            "TC-102",
        )
    )


def test_test_claim_uses_actual_test_citation() -> None:
    """TC-102 FAIL claim must cite actual Test evidence."""

    contract = build_contract(
        obs_023_context()
    )

    test_claims = tuple(
        claim
        for claim in contract.allowed_claims
        if claim.category
        == ClaimCategory.TEST
    )

    assert len(test_claims) == 1

    test_claim = test_claims[0]

    assert "FAIL" in (
        test_claim.statement
    )

    assert len(
        test_claim.citation_ids
    ) == 1

    citation = (
        contract.citations
        .by_citation_id(
            test_claim
            .citation_ids[0]
        )
    )

    assert citation is not None
    assert citation.document_type == "test"


def test_code_claim_contains_line_citation() -> None:
    """Code claim must use validated source provenance."""

    contract = build_contract(
        obs_023_context()
    )

    code_claims = tuple(
        claim
        for claim in contract.allowed_claims
        if claim.category
        == ClaimCategory.CODE
    )

    assert len(code_claims) == 1

    citation = (
        contract.citations
        .by_citation_id(
            code_claims[0]
            .citation_ids[0]
        )
    )

    assert citation is not None
    assert citation.start_line == 26
    assert citation.end_line == 47


def test_contract_prohibits_closure_claim_when_not_assessed() -> None:
    """Routine assessment must prohibit false closure claims."""

    contract = build_contract(
        obs_023_context()
    )

    closure_prohibitions = tuple(
        claim
        for claim
        in contract.prohibited_claims
        if claim.category
        == ClaimCategory.CLOSURE
    )

    statements = " ".join(
        prohibition.statement.lower()
        for prohibition
        in closure_prohibitions
    )

    assert "resolved" in statements
    assert "closed" in statements


def test_failed_grounding_blocks_contract() -> None:
    """A contract cannot be created from failed grounding."""

    context = obs_023_context()

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

    failed_grounding = replace(
        grounding,
        passed=False,
    )

    with pytest.raises(
        AnswerContractError
    ):
        build_answer_contract(
            decision=decision,
            citations=citations,
            grounding=failed_grounding,
        )


def test_revision_conflict_prohibits_revision_selection() -> None:
    """REQ-050 conflict must prohibit automatic revision selection."""

    context = make_context(
        [
            make_evidence(
                "FRD-050-R1",
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
                "FRD-050-R2",
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

    contract = build_contract(
        context
    )

    revision_prohibitions = tuple(
        claim
        for claim
        in contract.prohibited_claims
        if claim.category
        == ClaimCategory.REVISION
    )

    assert revision_prohibitions

    assert (
        contract.assessment.value
        == "INCONCLUSIVE"
    )


def test_traceability_contract_preserves_req_041() -> None:
    """OBS-052 contract must not substitute REQ-042."""

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
                "REQ-041 requirement evidence.",
                document_name=(
                    "FRD_Alert_Handling_Rev01.pdf"
                ),
                document_revision="Rev01",
                page_number=1,
                requirement_id="REQ-041",
            ),
        ]
    )

    contract = build_contract(
        context
    )

    assert contract.requirement_ids == (
        "REQ-041",
    )

    assert "REQ-042" not in (
        contract.engineering_id_allowlist
    )

    traceability_prohibitions = tuple(
        claim
        for claim
        in contract.prohibited_claims
        if claim.category
        == ClaimCategory.TRACEABILITY
    )

    assert traceability_prohibitions


def test_contract_dict_is_json_safe() -> None:
    """Contract should serialize into plain JSON-compatible values."""

    contract = build_contract(
        obs_023_context()
    )

    payload = contract.to_dict()

    assert payload[
        "decision"
    ][
        "assessment"
    ] == "SUPPORTED"

    assert payload[
        "grounding"
    ][
        "passed"
    ] is True

    assert payload[
        "citation_allowlist"
    ]
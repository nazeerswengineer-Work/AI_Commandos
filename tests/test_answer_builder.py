"""
Tests for Phase 13D deterministic answer packages.
"""

from __future__ import annotations

import json

import pytest

from src.answer_builder import (
    AnswerPackageError,
    build_answer_package,
    normalize_excerpt,
)
from src.answer_contract import (
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
    "d"
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
    """Create citation-ready evidence."""

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
    """Create a complete supported context."""

    return make_context(
        [
            make_evidence(
                "OBS-023",
                "observation",
                (
                    "Null input caused the observed "
                    "processing failure."
                ),
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
                (
                    "data_status_t process_data(...) "
                    "{ input->value; }"
                ),
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
                (
                    "REQ-017 requires invalid input "
                    "to be rejected safely."
                ),
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
                (
                    "TC-102 Result Status: FAIL."
                ),
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


def build_package(
    context: FinalEvidenceContext,
    *,
    evaluate_closure: bool = False,
    excerpt_length: int = 500,
):
    """Build one completely grounded package."""

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

    contract = build_answer_contract(
        decision=decision,
        citations=citations,
        grounding=grounding,
    )

    return build_answer_package(
        context=context,
        contract=contract,
        maximum_excerpt_characters=(
            excerpt_length
        ),
    )


def test_supported_package_is_created() -> None:
    """A grounded supported decision should produce a package."""

    package = build_package(
        obs_023_context()
    )

    assert (
        package.decision.assessment
        == "SUPPORTED"
    )

    assert (
        package.decision.confidence
        == "HIGH"
    )

    assert package.grounding_passed is True
    assert package.evidence_record_count == 4


def test_package_contains_all_core_sections() -> None:
    """The package should contain grouped evidence sections."""

    package = build_package(
        obs_023_context()
    )

    assert package.section(
        "frd"
    ) is not None

    assert package.section(
        "observation"
    ) is not None

    assert package.section(
        "code"
    ) is not None

    assert package.section(
        "test"
    ) is not None


def test_source_excerpts_are_citation_bound() -> None:
    """Every evidence excerpt must carry an approved citation."""

    package = build_package(
        obs_023_context()
    )

    allowlist = set(
        package.citation_allowlist
    )

    assert package.all_evidence_citation_ids

    assert set(
        package.all_evidence_citation_ids
    ).issubset(
        allowlist
    )


def test_test_excerpt_preserves_fail_status() -> None:
    """Test evidence must preserve its explicit result."""

    package = build_package(
        obs_023_context()
    )

    test_section = package.section(
        "test"
    )

    assert test_section is not None
    assert len(test_section.excerpts) == 1

    excerpt = test_section.excerpts[0]

    assert excerpt.test_case_id == (
        "TC-102"
    )

    assert excerpt.result_status == (
        "FAIL"
    )

    assert "FAIL" in excerpt.excerpt


def test_code_excerpt_preserves_provenance() -> None:
    """Code evidence should preserve file, function, and lines."""

    package = build_package(
        obs_023_context()
    )

    code_section = package.section(
        "code"
    )

    assert code_section is not None

    excerpt = code_section.excerpts[0]

    assert (
        "data_processor.c"
        in excerpt.provenance_location
    )

    assert "process_data" in (
        excerpt.provenance_location
    )

    assert "26-47" in (
        excerpt.provenance_location
    )


def test_deterministic_text_contains_citations() -> None:
    """Rendered deterministic text must use citation markers."""

    package = build_package(
        obs_023_context()
    )

    assert "[CIT-001]" in (
        package.deterministic_text
    )

    assert "[CIT-002]" in (
        package.deterministic_text
    )


def test_excerpt_truncation_is_deterministic() -> None:
    """Long evidence should be truncated without generated text."""

    text = (
        "This is deterministic evidence content. "
        * 20
    )

    excerpt = normalize_excerpt(
        text,
        maximum_characters=80,
    )

    assert len(excerpt) <= 84
    assert excerpt.endswith(" ...")


def test_empty_excerpt_is_rejected() -> None:
    """Empty evidence cannot enter the package."""

    with pytest.raises(
        AnswerPackageError
    ):
        normalize_excerpt(
            "   "
        )


def test_context_citation_mismatch_is_rejected() -> None:
    """The package must reject context/citation mismatch."""

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

    contract = build_answer_contract(
        decision=decision,
        citations=citations,
        grounding=grounding,
    )

    reduced_context = FinalEvidenceContext(
        evidence=context.evidence[:-1],
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )

    with pytest.raises(
        AnswerPackageError
    ):
        build_answer_package(
            context=reduced_context,
            contract=contract,
        )


def test_package_is_json_safe() -> None:
    """The full package must serialize to JSON."""

    package = build_package(
        obs_023_context()
    )

    payload = package.to_dict()

    serialized = json.dumps(
        payload
    )

    assert serialized
    assert payload[
        "decision"
    ][
        "assessment"
    ] == "SUPPORTED"


def test_package_preserves_prohibited_claims() -> None:
    """Safety prohibitions must survive answer-package creation."""

    package = build_package(
        obs_023_context()
    )

    statements = " ".join(
        claim.statement.lower()
        for claim
        in package.prohibited_claims
    )

    assert (
        "do not change"
        in statements
    )

    assert (
        "resolved"
        in statements
    )


def test_package_contains_narrator_instructions() -> None:
    """The package must carry mandatory Phase 14 constraints."""

    package = build_package(
        obs_023_context()
    )

    instructions = " ".join(
        package.narrator_instructions
    ).lower()

    assert (
        "preserve the assessment"
        in instructions
    )

    assert (
        "do not infer facts"
        in instructions
    )
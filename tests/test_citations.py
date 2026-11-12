"""
Tests for Phase 13A citation and provenance schemas.
"""

from __future__ import annotations

import pytest

from src.citations import (
    CitationSourceType,
    CitationValidationError,
    EvidenceCitation,
    build_citation_collection,
    build_evidence_citation,
)
from src.context_builder import (
    FinalEvidenceContext,
)
from src.retrieval import (
    RetrievedEvidence,
)


VALID_HASH = (
    "a"
    * 64
)


def make_pdf_evidence(
    evidence_id: str = "FRD::document.pdf::PAGE-1::REQ-017",
    document_type: str = "frd",
    *,
    document_name: str | None = "document.pdf",
    page_number: int | None = 1,
    document_revision: str | None = "Rev01",
    requirement_id: str | None = "REQ-017",
    observation_id: str | None = None,
    test_case_id: str | None = None,
    result_status: str | None = None,
) -> RetrievedEvidence:
    """Create synthetic PDF evidence."""

    metadata: dict[str, object] = {
        "document_name": (
            document_name
        ),
        "page_number": (
            page_number
        ),
        "document_revision": (
            document_revision
        ),
        "requirement_id": (
            requirement_id
        ),
        "observation_id": (
            observation_id
        ),
        "test_case_id": (
            test_case_id
        ),
        "result_status": (
            result_status
        ),
        "content_hash": (
            VALID_HASH
        ),
        "file_path": (
            "documents/example.pdf"
        ),
    }

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type=document_type,
        content="PDF evidence content.",
        metadata=metadata,
        retrieval_method="test",
        score=1.0,
        rank=1,
    )


def make_code_evidence(
    *,
    start_line: int | None = 26,
    end_line: int | None = 47,
) -> RetrievedEvidence:
    """Create synthetic source-code evidence."""

    metadata: dict[str, object] = {
        "document_name": (
            "data_processor.c"
        ),
        "file_name": (
            "data_processor.c"
        ),
        "function_name": (
            "process_data"
        ),
        "start_line": (
            start_line
        ),
        "end_line": (
            end_line
        ),
        "language": "c",
        "requirement_id": (
            "REQ-017"
        ),
        "observation_id": (
            "OBS-023"
        ),
        "test_case_id": (
            "TC-102"
        ),
        "content_hash": (
            VALID_HASH
        ),
        "file_path": (
            "documents/code/"
            "data_processor.c"
        ),
    }

    return RetrievedEvidence(
        evidence_id=(
            "CODE::data_processor.c::"
            "process_data::26-47"
        ),
        document_type="code",
        content="Source-code evidence.",
        metadata=metadata,
        retrieval_method="test",
        score=1.0,
        rank=1,
    )


def make_context(
    evidence: list[
        RetrievedEvidence
    ],
) -> FinalEvidenceContext:
    """Create one final evidence context."""

    return FinalEvidenceContext(
        evidence=tuple(
            evidence
        ),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def test_pdf_citation_is_created() -> None:
    """PDF evidence should produce one-based page provenance."""

    citation = build_evidence_citation(
        make_pdf_evidence(),
        citation_rank=1,
    )

    assert citation.citation_id == (
        "CIT-001"
    )

    assert citation.source_type == (
        CitationSourceType.PDF
    )

    assert citation.document_name == (
        "document.pdf"
    )

    assert citation.page_number == 1

    assert citation.requirement_id == (
        "REQ-017"
    )


def test_code_citation_is_created() -> None:
    """Code evidence should preserve file and physical lines."""

    citation = build_evidence_citation(
        make_code_evidence(),
        citation_rank=1,
    )

    assert citation.source_type == (
        CitationSourceType
        .SOURCE_CODE
    )

    assert citation.file_name == (
        "data_processor.c"
    )

    assert citation.function_name == (
        "process_data"
    )

    assert citation.start_line == 26
    assert citation.end_line == 47
    assert citation.page_number is None


def test_pdf_without_page_is_rejected() -> None:
    """PDF provenance without page number is invalid."""

    with pytest.raises(
        CitationValidationError
    ):
        build_evidence_citation(
            make_pdf_evidence(
                page_number=None
            ),
            citation_rank=1,
        )


def test_code_without_start_line_is_rejected() -> None:
    """Code provenance requires one-based start line."""

    with pytest.raises(
        CitationValidationError
    ):
        build_evidence_citation(
            make_code_evidence(
                start_line=None
            ),
            citation_rank=1,
        )


def test_code_without_end_line_is_rejected() -> None:
    """Code provenance requires end line."""

    with pytest.raises(
        CitationValidationError
    ):
        build_evidence_citation(
            make_code_evidence(
                end_line=None
            ),
            citation_rank=1,
        )


def test_code_invalid_line_range_is_rejected() -> None:
    """End line cannot precede start line."""

    with pytest.raises(
        CitationValidationError
    ):
        build_evidence_citation(
            make_code_evidence(
                start_line=47,
                end_line=26,
            ),
            citation_rank=1,
        )


def test_pdf_page_is_one_based() -> None:
    """PDF page zero is invalid."""

    with pytest.raises(
        CitationValidationError
    ):
        build_evidence_citation(
            make_pdf_evidence(
                page_number=0
            ),
            citation_rank=1,
        )


def test_collection_preserves_context_order() -> None:
    """Citation order should follow final-context order."""

    context = make_context(
        [
            make_pdf_evidence(
                evidence_id="FRD-1"
            ),
            make_code_evidence(),
            make_pdf_evidence(
                evidence_id="TEST-1",
                document_type="test",
                document_name=(
                    "Test_Report_TC-102.pdf"
                ),
                test_case_id="TC-102",
                result_status="FAIL",
            ),
        ]
    )

    collection = (
        build_citation_collection(
            context
        )
    )

    assert collection.citation_ids == (
        "CIT-001",
        "CIT-002",
        "CIT-003",
    )

    assert collection.evidence_ids == (
        "FRD-1",
        (
            "CODE::data_processor.c::"
            "process_data::26-47"
        ),
        "TEST-1",
    )


def test_duplicate_evidence_id_is_rejected() -> None:
    """One final context must not cite the same evidence twice."""

    first = make_pdf_evidence(
        evidence_id="DUPLICATE"
    )

    second = make_pdf_evidence(
        evidence_id="DUPLICATE"
    )

    context = make_context(
        [
            first,
            second,
        ]
    )

    with pytest.raises(
        CitationValidationError
    ):
        build_citation_collection(
            context
        )


def test_test_result_status_is_preserved() -> None:
    """Test citation should retain explicit result status."""

    citation = build_evidence_citation(
        make_pdf_evidence(
            evidence_id=(
                "TEST::Test_Report_TC-102.pdf"
                "::PAGE-1::TC-102"
            ),
            document_type="test",
            document_name=(
                "Test_Report_TC-102.pdf"
            ),
            requirement_id="REQ-017",
            test_case_id="TC-102",
            result_status="FAIL",
        ),
        citation_rank=1,
    )

    assert citation.test_case_id == (
        "TC-102"
    )

    assert citation.result_status == (
        "FAIL"
    )


def test_invalid_content_hash_is_rejected() -> None:
    """Malformed content hash must not enter the citation contract."""

    with pytest.raises(
        CitationValidationError
    ):
        EvidenceCitation(
            citation_id="CIT-001",
            evidence_id="FRD-1",
            source_type=(
                CitationSourceType.PDF
            ),
            document_type="frd",
            document_name="document.pdf",
            page_number=1,
            content_hash="invalid",
            display_label=(
                "FRD | document.pdf | page 1"
            ),
        )


def test_collection_lookup() -> None:
    """Citation lookup should support citation and evidence IDs."""

    context = make_context(
        [
            make_pdf_evidence(
                evidence_id="FRD-1"
            )
        ]
    )

    collection = (
        build_citation_collection(
            context
        )
    )

    citation = collection.by_citation_id(
        "CIT-001"
    )

    assert citation is not None

    assert citation.evidence_id == (
        "FRD-1"
    )

    evidence_lookup = (
        collection.by_evidence_id(
            "FRD-1"
        )
    )

    assert evidence_lookup == citation
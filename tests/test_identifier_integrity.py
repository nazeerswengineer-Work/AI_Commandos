"""
Tests for explicit engineering-identifier integrity.
"""

from __future__ import annotations

import pytest

from src.context_builder import (
    FinalEvidenceContext,
)
from src.identifier_integrity import (
    ExplicitIdentifierNotFound,
    extract_explicit_identifiers,
    validate_explicit_identifier_integrity,
)
from src.retrieval import (
    RetrievedEvidence,
)


def make_evidence(
    evidence_id: str,
    document_type: str,
    *,
    requirement_id: str | None = None,
    observation_id: str | None = None,
    test_case_id: str | None = None,
) -> RetrievedEvidence:

    metadata: dict[
        str,
        object,
    ] = {}

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

    return RetrievedEvidence(
        evidence_id=evidence_id,
        document_type=document_type,
        content=(
            "Synthetic evidence content."
        ),
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

    return FinalEvidenceContext(
        evidence=tuple(
            evidence
        ),
        exact_count=0,
        traceability_count=0,
        rrf_count=0,
    )


def valid_context() -> FinalEvidenceContext:

    return make_context(
        [
            make_evidence(
                "OBS-EVIDENCE",
                "observation",
                requirement_id="REQ-017",
                observation_id="OBS-023",
                test_case_id="TC-102",
            ),
            make_evidence(
                "FRD-EVIDENCE",
                "frd",
                requirement_id="REQ-017",
            ),
            make_evidence(
                "TEST-EVIDENCE",
                "test",
                requirement_id="REQ-017",
                test_case_id="TC-102",
            ),
        ]
    )


def test_extract_explicit_ids() -> None:

    identifiers = (
        extract_explicit_identifiers(
            (
                "Compare OBS-023 with REQ_017 "
                "using TC 102."
            )
        )
    )

    assert tuple(
        identifier.value
        for identifier in identifiers
    ) == (
        "OBS-023",
        "REQ-017",
        "TC-102",
    )


def test_natural_language_without_id_passes() -> None:

    result = (
        validate_explicit_identifier_integrity(
            query=(
                "Which requirement covers "
                "safe null-input handling?"
            ),
            context=valid_context(),
            strict=True,
        )
    )

    assert result.passed is True

    assert (
        result
        .explicit_identifiers_present
        is False
    )


def test_existing_observation_passes() -> None:

    result = (
        validate_explicit_identifier_integrity(
            query="Analyze OBS-023",
            context=valid_context(),
            strict=True,
        )
    )

    assert result.passed is True

    assert result.requested_ids == (
        "OBS-023",
    )


def test_missing_observation_is_rejected() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ) as error:

        validate_explicit_identifier_integrity(
            query="Analyze OBS-999",
            context=valid_context(),
            strict=True,
        )

    assert "OBS-999" in (
        error.value.missing_ids
    )


def test_missing_requirement_is_rejected() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ) as error:

        validate_explicit_identifier_integrity(
            query="Explain REQ-999",
            context=valid_context(),
            strict=True,
        )

    assert "REQ-999" in (
        error.value.missing_ids
    )


def test_missing_test_case_is_rejected() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ) as error:

        validate_explicit_identifier_integrity(
            query="Show TC-999 result",
            context=valid_context(),
            strict=True,
        )

    assert "TC-999" in (
        error.value.missing_ids
    )


def test_mixed_valid_and_invalid_ids_are_rejected() -> None:

    with pytest.raises(
        ExplicitIdentifierNotFound
    ) as error:

        validate_explicit_identifier_integrity(
            query=(
                "Compare OBS-023 "
                "with OBS-999"
            ),
            context=valid_context(),
            strict=True,
        )

    assert error.value.missing_ids == (
        "OBS-999",
    )


def test_content_mentions_do_not_establish_identity() -> None:
    """
    Free-text mention must not satisfy the authoritative ID gate.
    """

    context = make_context(
        [
            RetrievedEvidence(
                evidence_id="UNRELATED",
                document_type="observation",
                content=(
                    "This prose mentions OBS-999, "
                    "but metadata identifies OBS-052."
                ),
                metadata={
                    "observation_id": (
                        "OBS-052"
                    ),
                    "requirement_id": (
                        "REQ-041"
                    ),
                },
                retrieval_method="test",
                score=1.0,
                rank=1,
            )
        ]
    )

    with pytest.raises(
        ExplicitIdentifierNotFound
    ):

        validate_explicit_identifier_integrity(
            query="OBS-999",
            context=context,
            strict=True,
        )